"""create_agent-based example exercising the v3 ``tools`` channel.

`thread.tool_calls` and the underlying ``tools`` channel only emit
events when an actual model issues a tool call through langchain's
agent stack. The synthetic ``streaming_graph.py`` hand-builds
`AIMessage(tool_calls=[...])` and a `ToolMessage` via the messages
reducer — that gets persisted in state but never produces tool-call
telemetry on the wire. This graph fixes that by going through
`create_agent` with a real tool, driven by a `GenericFakeChatModel` so
the test stays hermetic (no `ANTHROPIC_API_KEY` required).

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
from langchain_core.messages import AIMessage
from langchain_core.tools import tool


@tool
def search(query: str) -> str:
    """Look up `query` in a fake search index."""
    return f"result for {query!r}"


class _ToolBindingFakeChatModel(FakeMessagesListChatModel):
    """Fake chat model that satisfies `create_agent`'s ``bind_tools`` call.

    ``create_agent`` calls ``model.bind_tools(tools)`` to attach the tool
    schema (``langchain/agents/factory.py:1284``). The base
    ``FakeMessagesListChatModel`` inherits ``BaseChatModel.bind_tools``,
    which raises ``NotImplementedError``. We don't actually need the
    bound schema — the fake replays scripted ``AIMessage``s with their
    own ``tool_calls`` field — so override ``bind_tools`` as a no-op.

    ``FakeMessagesListChatModel`` is preferred over
    ``GenericFakeChatModel`` here because the latter's ``_stream``
    breaks the message into content chunks and **drops ``tool_calls``**
    when content is empty, causing the v2 streaming path inside
    ``create_agent`` to raise ``RuntimeError("v2 stream finished
    without producing a message")``. ``FakeMessagesListChatModel``
    falls back to the default ``_stream`` that yields the whole
    message in one chunk, preserving ``tool_calls``.
    """

    def bind_tools(self, tools: Any, **kwargs: Any) -> _ToolBindingFakeChatModel:
        return self


# Two scripted turns. ``FakeMessagesListChatModel`` cycles through
# ``responses`` (resetting to index 0 after the last) so the graph
# can be run many times without restart; per run, ``create_agent``
# invokes the model exactly twice (once to issue the tool call,
# once after the tool result to produce the terminating answer).
_supervisor_responses: list[AIMessage] = [
    AIMessage(
        content="",
        id="ai-tools-1",
        tool_calls=[
            {
                "id": "tc-1",
                "name": "search",
                "args": {"query": "v3"},
            }
        ],
    ),
    AIMessage(content="done.", id="ai-tools-2"),
]


_supervisor_model = _ToolBindingFakeChatModel(responses=_supervisor_responses)


graph = create_agent(
    model=_supervisor_model,
    tools=[search],
    system_prompt="You are a research assistant. Use the search tool when asked.",
    name="v3_tools_agent",
)
