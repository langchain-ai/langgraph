"""create_agent-based example exercising the v3 `tools` channel.

`thread.tool_calls` and the underlying `tools` channel only emit
events when an actual model issues a tool call through langchain's
agent stack. The synthetic `streaming_graph.py` hand-builds
`AIMessage(tool_calls=[...])` and a `ToolMessage` via the messages
reducer — that gets persisted in state but never produces tool-call
telemetry on the wire. This graph fixes that by going through
`create_agent` with a real tool, driven by a hermetic fake chat model
(no `ANTHROPIC_API_KEY` required).

Flow on `run.start`:

1. Supervisor model returns an `AIMessage(tool_calls=[search(query="v3")])`.
2. langchain's tool node executes `search` and produces a `ToolMessage`.
3. Supervisor model returns a final `AIMessage("done.")` to terminate.

The v3 streaming layer surfaces this as `messages` + `tools` channel
events at root namespace.
"""

from __future__ import annotations

from typing import Any

from langchain.agents import create_agent
from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
from langchain_core.messages import AIMessage, BaseMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.tools import tool


@tool
def search(query: str) -> str:
    """Look up `query` in a fake search index."""
    return f"result for {query!r}"


class _ToolBindingFakeChatModel(FakeMessagesListChatModel):
    """Stateless fake chat model driving a single `search` tool call.

    `create_agent` calls `model.bind_tools(tools)` to attach the tool
    schema (`langchain/agents/factory.py:1284`). The base
    `FakeMessagesListChatModel` inherits `BaseChatModel.bind_tools`,
    which raises `NotImplementedError`, so `bind_tools` is overridden as
    a no-op (the reply is hand-built and already carries `tool_calls`).

    The reply is derived from conversation state rather than a cycling
    response list: the `search` tool call is issued until a `ToolMessage`
    appears, then a terminating `AIMessage`. This avoids the response-index
    parity flake where `FakeMessagesListChatModel.responses` is shared
    process-wide and cycles `0 -> 1 -> 0`; a run that started mid-cycle
    (e.g. on a reused server worker) would reply `"done."` first and emit
    no tool call. Being order-independent, every run emits exactly one
    tool call regardless of how many times the model was previously called.

    `FakeMessagesListChatModel` is subclassed (rather than
    `GenericFakeChatModel`) because the latter's `_stream` breaks the
    message into content chunks and drops `tool_calls` when content is
    empty, causing the v2 streaming path inside `create_agent` to raise
    `RuntimeError("v2 stream finished without producing a message")`.
    The inherited `_stream` yields the whole message in one chunk,
    preserving `tool_calls`.
    """

    def bind_tools(self, tools: Any, **kwargs: Any) -> _ToolBindingFakeChatModel:
        return self

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: Any = None,
        **kwargs: Any,
    ) -> ChatResult:
        if any(isinstance(m, ToolMessage) for m in messages):
            response = AIMessage(content="done.", id="ai-tools-done")
        else:
            response = AIMessage(
                content="",
                id="ai-tools-call",
                tool_calls=[{"id": "tc-1", "name": "search", "args": {"query": "v3"}}],
            )
        return ChatResult(generations=[ChatGeneration(message=response)])


# `responses` is a required field on `FakeMessagesListChatModel`, but the
# overridden `_generate` derives its reply from state and never reads it.
_supervisor_model = _ToolBindingFakeChatModel(responses=[])


graph = create_agent(
    model=_supervisor_model,
    tools=[search],
    system_prompt="You are a research assistant. Use the search tool when asked.",
    name="v3_tools_agent",
)
