import base64
import hashlib
import logging
import re
import time
from collections.abc import Callable, Sequence
from typing import (
    Any,
    Literal,
)

from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models import BaseChatModel, LanguageModelInput
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    ToolCall,
)
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.runnables import Runnable, RunnableLambda
from langchain_core.tools import BaseTool
from pydantic import BaseModel

from langgraph.prebuilt.chat_agent_executor import StructuredResponse

logger = logging.getLogger(__name__)

# Approved LLM registry
_APPROVED_LLM_REGISTRY = {"fake-tool-call-model-approved"}

# Approved tool allow list
_APPROVED_TOOL_ALLOW_LIST: set[str] = set()

# Dangerous patterns for input/output sanitization
_DANGEROUS_PATTERNS = [
    re.compile(r"\beval\s*\(", re.IGNORECASE),
    re.compile(r"\bexec\s*\(", re.IGNORECASE),
    re.compile(r"\bsubprocess\b", re.IGNORECASE),
    re.compile(r"\bos\.system\s*\(", re.IGNORECASE),
    re.compile(r"\bos\.popen\s*\(", re.IGNORECASE),
    re.compile(r"\b__import__\s*\(", re.IGNORECASE),
    re.compile(r"\bcompile\s*\(", re.IGNORECASE),
    re.compile(r"\bexecfile\s*\(", re.IGNORECASE),
    re.compile(r"\binput\s*\(", re.IGNORECASE),
    re.compile(r"<script[\s>]", re.IGNORECASE),
    re.compile(r"\bshell\s*=\s*True\b", re.IGNORECASE),
    # base64-encoded content detection
    re.compile(r"(?:[A-Za-z0-9+/]{4}){10,}(?:[A-Za-z0-9+/]{2}==|[A-Za-z0-9+/]{3}=)?"),
    # leetspeak / obfuscation patterns for common dangerous words
    re.compile(r"\b3v[a4]l\b", re.IGNORECASE),
    re.compile(r"\b[e3][x\*][e3][c\(]", re.IGNORECASE),
]


def _sanitize_content(content: str) -> str:
    """Validate and sanitize content for dangerous patterns."""
    if not isinstance(content, str):
        raise ValueError("Message content must be a string.")
    for pattern in _DANGEROUS_PATTERNS:
        if pattern.search(content):
            raise ValueError(
                f"Blocked: message content contains a forbidden pattern: {pattern.pattern!r}"
            )
    # Check for possible base64-encoded dangerous payloads
    b64_pattern = re.compile(
        r"(?:[A-Za-z0-9+/]{4}){10,}(?:[A-Za-z0-9+/]{2}==|[A-Za-z0-9+/]{3}=)?"
    )
    for match in b64_pattern.finditer(content):
        try:
            decoded = base64.b64decode(match.group()).decode("utf-8", errors="ignore")
            for pattern in _DANGEROUS_PATTERNS:
                if pattern.search(decoded):
                    raise ValueError(
                        "Blocked: base64-encoded content contains a forbidden pattern."
                    )
        except Exception as exc:
            if "Blocked" in str(exc):
                raise
    return content


def _validate_messages(messages: list[BaseMessage]) -> None:
    """Validate and sanitize all input messages before processing."""
    if not messages:
        raise ValueError("Messages list must not be empty.")
    for i, message in enumerate(messages):
        if not isinstance(message, BaseMessage):
            raise TypeError(f"Message at index {i} is not a BaseMessage instance.")
        content = message.content
        if isinstance(content, str):
            _sanitize_content(content)
        elif isinstance(content, list):
            for part in content:
                if isinstance(part, dict) and "text" in part:
                    _sanitize_content(part["text"])
                elif isinstance(part, str):
                    _sanitize_content(part)


def _minimise_output(messages_string: str) -> str:
    """Apply output data minimisation: return only a fixed placeholder instead of echoing full input."""
    # Do not echo back the full concatenated input; return a minimal acknowledgement.
    return "[response]"


def _audit_log(event: str, details: dict) -> None:
    """Write a structured audit log entry for forensic readiness."""
    entry = {
        "event": event,
        "timestamp": time.time(),
        **details,
    }
    logger.info("AUDIT: %s", entry)


class FakeToolCallingModel(BaseChatModel):
    tool_calls: list[list[ToolCall]] | None = None
    structured_response: StructuredResponse | None = None
    index: int = 0
    tool_style: Literal["openai", "anthropic"] = "openai"
    tool_allow_list: list[str] | None = None

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        """Top Level call"""
        # Validate and sanitize all input messages before processing
        _validate_messages(messages)

        # Compute input hash for audit trail
        raw_input = "-".join(
            [m.content if isinstance(m.content, str) else str(m.content) for m in messages]
        )
        input_hash = hashlib.sha256(raw_input.encode("utf-8")).hexdigest()

        tool_calls = (
            self.tool_calls[self.index % len(self.tool_calls)]
            if self.tool_calls
            else []
        )

        # Apply output data minimisation: do not echo full input content
        minimised_content = _minimise_output(raw_input)

        # Validate/sanitize the output content before returning
        _sanitize_content(minimised_content)

        message = AIMessage(
            content=minimised_content, id=str(self.index), tool_calls=tool_calls.copy()
        )

        # Audit log the AI-driven decision
        _audit_log(
            "llm_generate",
            {
                "model": self._llm_type,
                "index": self.index,
                "input_hash": input_hash,
                "num_messages": len(messages),
                "num_tool_calls": len(tool_calls),
                "output_content": minimised_content,
            },
        )

        self.index += 1
        return ChatResult(generations=[ChatGeneration(message=message)])

    @property
    def _llm_type(self) -> str:
        return "fake-tool-call-model"

    def with_structured_output(
        self, schema: type[BaseModel]
    ) -> Runnable[LanguageModelInput, StructuredResponse]:
        if self.structured_response is None:
            raise ValueError("Structured response is not set")

        return RunnableLambda(lambda x: self.structured_response)

    def bind_tools(
        self,
        tools: Sequence[dict[str, Any] | type[BaseModel] | Callable | BaseTool],
        **kwargs: Any,
    ) -> Runnable[LanguageModelInput, BaseMessage]:
        if len(tools) == 0:
            raise ValueError("Must provide at least one tool")

        tool_dicts = []
        for tool in tools:
            if isinstance(tool, dict):
                tool_name = tool.get("function", {}).get("name") or tool.get("name")
                # Enforce tool allow list if configured
                if self.tool_allow_list is not None:
                    if tool_name not in self.tool_allow_list:
                        _audit_log(
                            "tool_denied",
                            {
                                "tool_name": tool_name,
                                "reason": "not in allow list",
                                "policy": "explicit_tool_allow_list",
                            },
                        )
                        raise ValueError(
                            f"Tool '{tool_name}' is not in the approved tool allow list."
                        )
                tool_dicts.append(tool)
                continue
            if not isinstance(tool, BaseTool):
                raise TypeError(
                    "Only BaseTool and dict is supported by FakeToolCallingModel.bind_tools"
                )

            # Enforce tool allow list for BaseTool instances
            if self.tool_allow_list is not None:
                if tool.name not in self.tool_allow_list:
                    _audit_log(
                        "tool_denied",
                        {
                            "tool_name": tool.name,
                            "reason": "not in allow list",
                            "policy": "explicit_tool_allow_list",
                        },
                    )
                    raise ValueError(
                        f"Tool '{tool.name}' is not in the approved tool allow list."
                    )

            # NOTE: this is a simplified tool spec for testing purposes only
            if self.tool_style == "openai":
                tool_dicts.append(
                    {
                        "type": "function",
                        "function": {
                            "name": tool.name,
                        },
                    }
                )
            elif self.tool_style == "anthropic":
                tool_dicts.append(
                    {
                        "name": tool.name,
                    }
                )

        _audit_log(
            "tools_bound",
            {
                "model": self._llm_type,
                "num_tools": len(tool_dicts),
                "tool_names": [
                    t.get("function", {}).get("name") or t.get("name")
                    for t in tool_dicts
                ],
            },
        )

        return self.bind(tools=tool_dicts)