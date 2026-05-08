import hashlib
import logging
import re
import datetime
from typing import Any
from uuid import uuid4

from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models.fake_chat_models import (
    FakeMessagesListChatModel,
)
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.tools import StructuredTool
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.prebuilt.chat_agent_executor import create_react_agent

from langgraph.pregel import Pregel

# Configure audit logger
audit_logger = logging.getLogger("ai_audit")
audit_logger.setLevel(logging.INFO)
if not audit_logger.handlers:
    _handler = logging.StreamHandler()
    _handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    audit_logger.addHandler(_handler)

# Approved model registry
APPROVED_MODEL_REGISTRY = {
    "FakeFunctionChatModel": "benchmark-fake-v1",
}

# Approved tool allow list (populated at agent creation time)
_APPROVED_TOOL_NAMES: set = set()

# Dangerous patterns for input/output sanitization
_DANGEROUS_PATTERNS = re.compile(
    r"\b(eval|exec|compile|__import__|subprocess|os\.system|open|base64)\b",
    re.IGNORECASE,
)

_SHELL_COMMAND_PATTERNS = re.compile(
    r"(&&|\|\||;|`|\$\(|>\s*/|<\s*/|rm\s+-|chmod\s+|curl\s+|wget\s+|nc\s+|ncat\s+)",
    re.IGNORECASE,
)

_MAX_TOOL_RESULT_LENGTH = 500
_MAX_INPUT_LENGTH = 1000


def _validate_and_sanitize_input(text: str) -> str:
    """Validate and sanitize input to the AI model."""
    if not isinstance(text, str):
        raise ValueError("Input must be a string.")
    if len(text) > _MAX_INPUT_LENGTH:
        raise ValueError(f"Input exceeds maximum allowed length of {_MAX_INPUT_LENGTH} characters.")
    if _DANGEROUS_PATTERNS.search(text):
        raise ValueError("Input contains potentially dangerous code execution primitives.")
    if _SHELL_COMMAND_PATTERNS.search(text):
        raise ValueError("Input contains potentially malicious shell command patterns.")
    return text


def _validate_and_sanitize_output(text: str) -> str:
    """Validate and sanitize LLM output for dangerous code execution primitives."""
    if not isinstance(text, str):
        return text
    if _DANGEROUS_PATTERNS.search(text):
        raise ValueError("LLM output contains dangerous code execution primitives and has been blocked.")
    if _SHELL_COMMAND_PATTERNS.search(text):
        raise ValueError("LLM output contains shell command patterns and has been blocked.")
    return text


def _minimise_tool_result(result: str) -> str:
    """Enforce output data minimisation for tool results."""
    if len(result) > _MAX_TOOL_RESULT_LENGTH:
        result = result[:_MAX_TOOL_RESULT_LENGTH]
    return result


def _check_tool_allowlist(tool_name: str) -> None:
    """Enforce tool allow list policy."""
    if tool_name not in _APPROVED_TOOL_NAMES:
        audit_logger.warning(
            "AUDIT: Denied tool invocation for tool not in allow list: %s", tool_name
        )
        raise ValueError(f"Tool '{tool_name}' is not in the approved tool allow list.")


def _verify_model_registry(model_class_name: str) -> str:
    """Verify model is in approved registry and return pinned version."""
    if model_class_name not in APPROVED_MODEL_REGISTRY:
        raise ValueError(
            f"Model '{model_class_name}' is not in the approved model registry. "
            "Only registry-approved models may be used."
        )
    return APPROVED_MODEL_REGISTRY[model_class_name]


def _watermark_output(content: str, model_id: str, version: str) -> str:
    """Add provenance label/watermark to AI-generated output."""
    timestamp = datetime.datetime.utcnow().isoformat()
    watermark = f"[AI-GENERATED | model={model_id} | version={version} | timestamp={timestamp}]"
    return f"{content}\n{watermark}"


def react_agent(n_tools: int, checkpointer: BaseCheckpointSaver | None) -> Pregel:
    # Verify model is in approved registry
    model_class_name = "FakeFunctionChatModel"
    model_version = _verify_model_registry(model_class_name)
    model_id = f"{model_class_name}-{model_version}"

    class FakeFunctionChatModel(FakeMessagesListChatModel):
        def bind_tools(self, functions: list):
            return self

        def _generate(
            self,
            messages: list[BaseMessage],
            stop: list[str] | None = None,
            run_manager: CallbackManagerForLLMRun | None = None,
            **kwargs: Any,
        ) -> ChatResult:
            # Log LLM interaction (input)
            input_summary = [
                {"type": type(m).__name__, "content_hash": hashlib.sha256(str(m.content).encode()).hexdigest()}
                for m in messages
            ]
            audit_logger.info(
                "AUDIT: LLM invocation | model=%s | version=%s | timestamp=%s | input_hash=%s",
                model_id,
                model_version,
                datetime.datetime.utcnow().isoformat(),
                hashlib.sha256(str(input_summary).encode()).hexdigest(),
            )

            response = self.responses[self.i].copy()
            if self.i < len(self.responses) - 1:
                self.i += 1
            else:
                self.i = 0

            # Validate and sanitize LLM output content
            if hasattr(response, "content") and isinstance(response.content, str):
                sanitized_content = _validate_and_sanitize_output(response.content)
                watermarked_content = _watermark_output(sanitized_content, model_id, model_version)
                response.content = watermarked_content

            # Validate tool call arguments in LLM output
            if hasattr(response, "tool_calls") and response.tool_calls:
                for tc in response.tool_calls:
                    tool_name = tc.get("name") if isinstance(tc, dict) else getattr(tc, "name", None)
                    if tool_name:
                        _check_tool_allowlist(tool_name)
                    args = tc.get("args") if isinstance(tc, dict) else getattr(tc, "args", {})
                    for arg_val in (args.values() if isinstance(args, dict) else []):
                        if isinstance(arg_val, str):
                            _validate_and_sanitize_output(arg_val)

            # Log LLM output
            audit_logger.info(
                "AUDIT: LLM response | model=%s | version=%s | timestamp=%s | output_hash=%s",
                model_id,
                model_version,
                datetime.datetime.utcnow().isoformat(),
                hashlib.sha256(str(response).encode()).hexdigest(),
            )

            generation = ChatGeneration(message=response)
            return ChatResult(generations=[generation])

    # Create tool with a stable, allowlisted name
    tool_name = "benchmark_search_tool"
    _APPROVED_TOOL_NAMES.add(tool_name)

    def _tool_fn(query: str) -> str:
        # Validate tool input
        _validate_and_sanitize_input(query)
        raw_result = f"result for query: {query}" * 10
        # Enforce output data minimisation
        minimised = _minimise_tool_result(raw_result)
        audit_logger.info(
            "AUDIT: Tool invocation | tool=%s | timestamp=%s | query_hash=%s | result_length=%d",
            tool_name,
            datetime.datetime.utcnow().isoformat(),
            hashlib.sha256(query.encode()).hexdigest(),
            len(minimised),
        )
        return minimised

    tool = StructuredTool.from_function(
        _tool_fn,
        name=tool_name,
        description="Approved benchmark search tool.",
    )

    model = FakeFunctionChatModel(
        responses=[
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "id": str(uuid4()),
                        "name": tool.name,
                        "args": {"query": str(uuid4())},
                    }
                ],
                id=str(uuid4()),
            )
            for _ in range(n_tools)
        ]
        + [
            AIMessage(content="answer", id=str(uuid4())),
        ]
    )

    audit_logger.info(
        "AUDIT: Agent created | model=%s | version=%s | timestamp=%s | tools=%s",
        model_id,
        model_version,
        datetime.datetime.utcnow().isoformat(),
        list(_APPROVED_TOOL_NAMES),
    )

    return create_react_agent(model, [tool], checkpointer=checkpointer)


if __name__ == "__main__":
    import asyncio

    import uvloop
    from langgraph.checkpoint.memory import InMemorySaver

    graph = react_agent(100, checkpointer=InMemorySaver())

    # Validate and sanitize input before sending to agent
    raw_user_input = "hi?"
    sanitized_user_input = _validate_and_sanitize_input(raw_user_input)

    agent_input = {"messages": [HumanMessage(sanitized_user_input)]}
    config = {"configurable": {"thread_id": "1"}, "recursion_limit": 20000}

    session_id = str(uuid4())

    async def run():
        audit_logger.info(
            "AUDIT: Agent run started | session_id=%s | timestamp=%s | input_hash=%s",
            session_id,
            datetime.datetime.utcnow().isoformat(),
            hashlib.sha256(sanitized_user_input.encode()).hexdigest(),
        )
        chunks = []
        async for c in graph.astream(agent_input, config=config):
            # Validate each streamed chunk for dangerous content
            chunk_str = str(c)
            _validate_and_sanitize_output(chunk_str)
            chunks.append(c)
        audit_logger.info(
            "AUDIT: Agent run completed | session_id=%s | timestamp=%s | chunk_count=%d",
            session_id,
            datetime.datetime.utcnow().isoformat(),
            len(chunks),
        )
        return len(chunks)

    uvloop.install()
    asyncio.run(run())