"""Deep-agent variant exercising v3 `thread.subgraphs` properly.

`create_deep_agent` builds a graph whose `task` tool dispatches to one
of its configured `SubAgent`s. When the supervisor's model issues a
`task(subagent_type="researcher", description=...)` tool call, the
sub-agent runs as a nested invocation and the v3 streaming server
emits the subagent's lifecycle, messages, and tool events under a
scoped namespace. That namespace is what `thread.subgraphs` surfaces
as a direct-child `ScopedStreamHandle`.

Both the supervisor and the researcher use `FakeMessagesListChatModel`
with pre-scripted responses so this graph is hermetic. No LLM API keys
are required, and the test is deterministic.
"""

from __future__ import annotations

from typing import Any

from deepagents import create_deep_agent
from deepagents.middleware.subagents import SubAgent
from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
from langchain_core.messages import AIMessage


class _FakeChatModelWithTools(FakeMessagesListChatModel):
    """`FakeMessagesListChatModel` that accepts `bind_tools(...)` as a no-op.

    `create_deep_agent` calls `model.bind_tools(tools)` to expose the `task`
    tool to the supervisor. The base `BaseChatModel.bind_tools` raises
    `NotImplementedError`. Pre-baked responses in `responses` already carry
    the desired `tool_calls`, so we ignore the tools list and return self.
    """

    def bind_tools(self, tools: Any, **kwargs: Any) -> _FakeChatModelWithTools:
        return self


# Supervisor turn 1: dispatch to the researcher via the `task` tool.
# Supervisor turn 2: emit a final assistant message (no more tool calls),
# which closes the agent loop.
_supervisor_model = _FakeChatModelWithTools(
    responses=[
        AIMessage(
            content="",
            id="sup-1",
            tool_calls=[
                {
                    "id": "tc-task-1",
                    "name": "task",
                    "args": {
                        "subagent_type": "researcher",
                        "description": "research v3 streaming",
                    },
                }
            ],
        ),
        AIMessage(content="Research complete.", id="sup-2"),
    ]
)


# Researcher turn 1: final message, no tool calls. Closes the subagent loop.
_researcher_model = _FakeChatModelWithTools(
    responses=[
        AIMessage(
            content="v3 streaming is event-typed and thread-centric.", id="res-1"
        ),
    ]
)


_researcher: SubAgent = {
    "name": "researcher",
    "description": (
        "Looks up notes on a topic and returns a short summary. "
        "Use this when the user wants to research something."
    ),
    "system_prompt": (
        "You are a research assistant. Reply with one or two sentences "
        "summarising what the user asked about. Do not call any tools."
    ),
    "model": _researcher_model,
}


graph = create_deep_agent(
    model=_supervisor_model,
    system_prompt=(
        "You are a supervisor coordinating a researcher subagent. "
        "When the user asks to research anything, call the `task` tool "
        "with subagent_type='researcher'."
    ),
    subagents=[_researcher],
    name="v3_deep_agent",
)
