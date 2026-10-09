"""Example graph exercising the full v3 streaming surface.

Topology:

    __start__ -> stream_message -> call_tool -> ask_human -> subgraph -> __end__

Each node is designed to surface a specific v3 channel:

- `stream_message` yields token-by-token AI message chunks (`messages`).
- `call_tool` invokes a tool and emits a tool-call lifecycle (`tools`).
- `ask_human` raises an `interrupt(...)` to test `thread.interrupted` /
  `thread.run.respond(...)` (`lifecycle` / `input`).
- `subgraph` is a nested `StateGraph` invoked once so `thread.subgraphs` has
  exactly one direct child (`tasks` + `messages` under a namespace).

Extensions: every node calls `get_stream_writer()("progress", {...})` so
`thread.extensions["progress"]` produces deterministic events.

No real LLM is used — message streaming is simulated by yielding a list of
`AIMessageChunk`s from the node. This keeps the integration suite
hermetic.
"""

from __future__ import annotations

import operator
from collections.abc import AsyncIterator, Iterator
from typing import Annotated, Any, TypedDict

from langchain_core.callbacks import (
    AsyncCallbackManagerForLLMRun,
    CallbackManagerForLLMRun,
)
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, AIMessageChunk, BaseMessage, ToolMessage
from langchain_core.outputs import ChatGenerationChunk
from langchain_core.tools import tool
from langgraph.config import get_stream_writer
from langgraph.graph import StateGraph
from langgraph.graph.message import add_messages
from langgraph.stream.transformers import CustomTransformer, UpdatesTransformer
from langgraph.types import interrupt


class _StreamingFakeChatModel(BaseChatModel):
    """Fake ``BaseChatModel`` that streams ``AIMessageChunk``s.

    Implements ``_stream`` / ``_astream`` so the v3 chat-model
    callback chain (``_aiter_v2_events`` in
    ``langchain_core/language_models/chat_models.py``) fires
    ``run_manager.on_stream_event(...)`` per normalized protocol
    event. ``StreamMessagesHandlerV2`` -- attached by the langgraph
    runtime when ``"messages"`` is in stream_modes -- catches those
    callbacks and surfaces them on the v3 wire ``messages`` channel
    at root namespace.

    The base ``FakeMessagesListChatModel`` would have worked for
    ``ainvoke`` but raises ``NotImplementedError`` from ``_stream``,
    so it can't drive the streaming-callback path. ``GenericFakeChatModel``
    implements ``_stream`` but takes an ``Iterator`` that gets
    exhausted across invocations.
    """

    text: str = "Hello, world!"
    message_id: str = "ai-msg-1"

    @property
    def _llm_type(self) -> str:
        return "streaming-fake-chat-model"

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        from langchain_core.outputs import ChatGeneration, ChatResult

        return ChatResult(
            generations=[
                ChatGeneration(message=AIMessage(content=self.text, id=self.message_id))
            ]
        )

    def _stream(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: object,
    ) -> Iterator[ChatGenerationChunk]:
        # Yield content as space-separated word chunks so deltas are
        # observable. The final chunk's ``chunk_position="last"`` tells
        # the callback chain to emit ``message-finish``.
        parts = self.text.split(" ")
        for i, part in enumerate(parts):
            content = part if i == 0 else " " + part
            chunk = AIMessageChunk(content=content, id=self.message_id)
            if i == len(parts) - 1:
                chunk.chunk_position = "last"
            yield ChatGenerationChunk(message=chunk)

    async def _astream(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: AsyncCallbackManagerForLLMRun | None = None,
        **kwargs: object,
    ) -> AsyncIterator[ChatGenerationChunk]:
        for chunk in self._stream(messages, stop=stop, **kwargs):
            yield chunk


_stream_model = _StreamingFakeChatModel()


class AgentState(TypedDict):
    """Top-level state for the agent.

    `messages` accumulates AI/tool/user messages via the standard `add_messages`
    reducer. `value` is a simple scalar to test the `values` channel.
    `items` accumulates list-append updates via `operator.add` so each node
    contributes a marker and the terminal state reflects the full path
    rather than only the last node's return.
    """

    messages: Annotated[list[BaseMessage], add_messages]
    value: str
    items: Annotated[list[str], operator.add]


@tool
def search(query: str) -> str:
    """Look up `query` in a fake search index."""
    return f"result for {query!r}"


# ---------------------------------------------------------------------------
# Nodes
# ---------------------------------------------------------------------------


async def stream_message(state: AgentState) -> dict[str, Any]:
    """Stream an AI message via a fake chat model.

    Awaiting ``model.ainvoke(...)`` drives langgraph's chat-model
    streaming callbacks (``StreamMessagesHandlerV2`` ->
    ``MessagesTransformer``), so the v3 ``messages`` channel emits the
    normalized delta lifecycle (``message-start`` ->
    ``content-block-start`` -> ``content-block-delta`` ->
    ``content-block-finish`` -> ``message-finish``) at root namespace.
    Returning the resolved ``AIMessage`` via the messages reducer also
    keeps the existing ``values`` snapshots intact.
    """
    writer = get_stream_writer()

    writer({"name": "progress", "step": "stream_message", "phase": "start"})

    # ``astream_events(version="v3")`` drives the chat model's
    # ``_aiter_v2_events`` path (``BaseChatModel`` in
    # ``langchain_core/language_models/chat_models.py``), which fires
    # ``run_manager.on_stream_event(...)`` per normalized protocol
    # event (``message-start`` / ``content-block-delta`` /
    # ``message-finish``). ``StreamMessagesHandlerV2`` -- attached by
    # the langgraph runtime when ``"messages"`` is in stream_modes --
    # catches those callbacks and surfaces them on the v3 wire
    # ``messages`` channel at root namespace. Plain ``astream(...)``
    # does NOT route through this handler.
    text_parts: list[str] = []
    message_id = "ai-msg-1"
    # ``astream_events(version="v3")`` returns an awaitable that resolves
    # to the async iterator.
    stream = await _stream_model.astream_events([], version="v3")
    async for event in stream:
        if event.get("event") == "content-block-delta":
            delta = event.get("delta") or {}
            t = delta.get("text") if isinstance(delta, dict) else None
            if isinstance(t, str):
                text_parts.append(t)
        elif event.get("event") == "message-start":
            mid = event.get("id")
            if isinstance(mid, str):
                message_id = mid
    final = AIMessage(content="".join(text_parts), id=message_id)

    writer({"name": "progress", "step": "stream_message", "phase": "end"})
    return {"messages": [final], "value": "x", "items": ["streamed"]}


def call_tool(state: AgentState) -> dict[str, Any]:
    """Invoke a tool and emit its result as a tool message.

    A tool call here exercises the `tools` channel in v3.
    """
    writer = get_stream_writer()
    writer({"name": "progress", "step": "call_tool", "phase": "start"})

    # Hand-roll a tool call so we don't need a model to issue it.
    tool_call_id = "tc-1"
    ai_with_tool = AIMessage(
        content="",
        id="ai-msg-2",
        tool_calls=[
            {
                "id": tool_call_id,
                "name": "search",
                "args": {"query": "v3"},
            }
        ],
    )
    result = search.invoke({"query": "v3"})
    tool_msg = ToolMessage(content=result, tool_call_id=tool_call_id)

    writer({"name": "progress", "step": "call_tool", "phase": "end"})
    return {
        "messages": [ai_with_tool, tool_msg],
        "items": ["tool"],
    }


def ask_human(state: AgentState) -> dict[str, Any]:
    """Pause the graph and wait for a `thread.run.respond(...)`.

    `interrupt(value)` raises a special exception that the runtime catches;
    the v3 lifecycle emits `input.requested` with this `value` and the
    client must call `thread.run.respond(answer)` to continue.
    """
    writer = get_stream_writer()
    writer({"name": "progress", "step": "ask_human", "phase": "start"})

    answer = interrupt("Are we good?")

    writer(
        {"name": "progress", "step": "ask_human", "phase": "end", "answer": str(answer)}
    )
    return {
        "messages": [AIMessage(content=f"Human said: {answer}", id="ai-msg-3")],
        "items": ["asked"],
    }


# ---------------------------------------------------------------------------
# Subgraph (exercises `thread.subgraphs`)
# ---------------------------------------------------------------------------


class SubState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    note: str


def sub_node(state: SubState) -> dict[str, Any]:
    """Single node in the subgraph; emits a message and a custom event."""
    writer = get_stream_writer()
    writer({"name": "progress", "step": "sub_node", "phase": "start"})
    msg = AIMessage(content="from subgraph", id="sub-msg-1")
    writer({"name": "progress", "step": "sub_node", "phase": "end"})
    return {"messages": [msg], "note": "ran"}


_sub_builder = StateGraph(SubState)
_sub_builder.add_node("sub", sub_node)
_sub_builder.set_entry_point("sub")
_sub_builder.set_finish_point("sub")
subgraph = _sub_builder.compile()


def run_subgraph(state: AgentState) -> dict[str, Any]:
    """Invoke the subgraph once so it appears as a direct child handle."""
    writer = get_stream_writer()
    writer({"name": "progress", "step": "run_subgraph", "phase": "start"})
    sub_state = subgraph.invoke({"messages": [], "note": ""})
    writer({"name": "progress", "step": "run_subgraph", "phase": "end"})
    return {
        "messages": sub_state["messages"],
        "items": ["sub"],
    }


# ---------------------------------------------------------------------------
# Top-level graph
# ---------------------------------------------------------------------------


_builder: StateGraph[AgentState, Any, Any, Any] = StateGraph(AgentState)
_builder.add_node("stream_message", stream_message)
_builder.add_node("call_tool", call_tool)
_builder.add_node("ask_human", ask_human)
_builder.add_node("run_subgraph", run_subgraph)

_builder.set_entry_point("stream_message")
_builder.add_edge("stream_message", "call_tool")
_builder.add_edge("call_tool", "ask_human")
_builder.add_edge("ask_human", "run_subgraph")
_builder.set_finish_point("run_subgraph")

graph = _builder.compile(
    name="v3_integration_agent",
    # Register transformers so ``custom`` (``get_stream_writer()``) and
    # ``updates`` channels emit on the wire. ``MessagesTransformer`` is
    # auto-registered by the v3 mux for any graph that streams a chat
    # model. ``ValuesTransformer`` / ``LifecycleTransformer`` are also
    # always-on natives.
    transformers=[CustomTransformer, UpdatesTransformer],
)
