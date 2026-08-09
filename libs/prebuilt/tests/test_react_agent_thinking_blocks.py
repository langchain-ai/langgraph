from typing import Any

from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.tools import tool
from langgraph.checkpoint.memory import InMemorySaver
from pydantic import Field

from langgraph.prebuilt import create_react_agent


class RecordingThinkingModel(BaseChatModel):
    """Fake chat model that records every messages list passed to it.

    On the first call it returns an AIMessage with both a `thinking` content
    block and a tool call. On subsequent calls it returns a plain text reply.
    """

    invocations: list[list[BaseMessage]] = Field(default_factory=list)

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        self.invocations.append(list(messages))
        if len(self.invocations) == 1:
            message = AIMessage(
                content=[
                    {"type": "thinking", "thinking": "let me think"},
                    {"type": "text", "text": "I'll call the tool"},
                ],
                tool_calls=[
                    {
                        "name": "echo",
                        "args": {"message": "hello"},
                        "id": "call-1",
                        "type": "tool_call",
                    }
                ],
            )
        else:
            message = AIMessage(content="Done")
        return ChatResult(generations=[ChatGeneration(message=message)])

    def bind_tools(self, tools: list[Any], **kwargs: Any) -> Any:
        return self

    @property
    def _llm_type(self) -> str:
        return "recording-thinking-model"


def _run_agent(model: RecordingThinkingModel) -> tuple[Any, dict]:
    @tool
    def echo(message: str) -> str:
        """Echo the given message back."""
        return message

    agent = create_react_agent(model, [echo], checkpointer=InMemorySaver())
    thread = {"configurable": {"thread_id": "1"}}
    agent.invoke({"messages": [HumanMessage(content="hello")]}, thread)
    return agent, thread


async def _arun_agent(model: RecordingThinkingModel) -> None:
    @tool
    def echo(message: str) -> str:
        """Echo the given message back."""
        return message

    agent = create_react_agent(model, [echo], checkpointer=InMemorySaver())
    thread = {"configurable": {"thread_id": "1"}}
    await agent.ainvoke({"messages": [HumanMessage(content="hello")]}, thread)
    await agent.ainvoke({"messages": [HumanMessage(content="hello again")]}, thread)


def _has_thinking_block(messages: list[BaseMessage]) -> bool:
    for msg in messages:
        if isinstance(msg, AIMessage) and isinstance(msg.content, list):
            for block in msg.content:
                if isinstance(block, dict) and block.get("type") == "thinking":
                    return True
    return False


def test_react_agent_does_not_echo_thinking_blocks_to_model() -> None:
    model = RecordingThinkingModel()
    _run_agent(model)

    assert len(model.invocations) == 2
    second_input = model.invocations[1]
    assert not _has_thinking_block(second_input), (
        f"Second model call received a thinking block that must be filtered: "
        f"{second_input}"
    )
    text_blocks = [
        block
        for msg in second_input
        if isinstance(msg, AIMessage) and isinstance(msg.content, list)
        for block in msg.content
        if isinstance(block, dict) and block.get("type") == "text"
    ]
    assert {"type": "text", "text": "I'll call the tool"} in text_blocks, (
        "Filtering thinking blocks must not strip the other content blocks from "
        "the model input"
    )


async def test_react_agent_does_not_echo_thinking_blocks_to_model_async() -> None:
    model = RecordingThinkingModel()
    await _arun_agent(model)

    assert len(model.invocations) >= 2
    for i, invocation in enumerate(model.invocations):
        if i == 0:
            continue
        assert not _has_thinking_block(invocation), (
            f"Model call {i} received a thinking block that must be filtered: "
            f"{invocation}"
        )

    history_call = next(
        invocation
        for invocation in model.invocations
        if any(
            isinstance(m, AIMessage) and isinstance(m.content, list) for m in invocation
        )
    )
    text_blocks = [
        block
        for msg in history_call
        if isinstance(msg, AIMessage) and isinstance(msg.content, list)
        for block in msg.content
        if isinstance(block, dict) and block.get("type") == "text"
    ]
    assert {"type": "text", "text": "I'll call the tool"} in text_blocks, (
        "Filtering thinking blocks must not strip the other content blocks from "
        "the model input"
    )


def test_react_agent_preserves_thinking_blocks_in_state() -> None:
    model = RecordingThinkingModel()
    agent, thread = _run_agent(model)

    state = agent.get_state(thread)
    messages = state.values["messages"]
    first_ai = next(
        m for m in messages if isinstance(m, AIMessage) and isinstance(m.content, list)
    )
    assert any(
        isinstance(b, dict) and b.get("type") == "thinking" for b in first_ai.content
    ), "Thinking block must be preserved in the persisted state"


def test_react_agent_thinking_filter_keeps_tool_calls() -> None:
    model = RecordingThinkingModel()
    _run_agent(model)

    second_input = model.invocations[1]
    tool_call_ids = [
        tc["id"]
        for m in second_input
        if isinstance(m, AIMessage)
        for tc in m.tool_calls
    ]
    assert "call-1" in tool_call_ids, "Filtered AIMessage must keep its tool_calls"
    assert any(
        isinstance(m, ToolMessage) and m.tool_call_id == "call-1" for m in second_input
    ), "ToolMessage pairing must be preserved in the model input"
