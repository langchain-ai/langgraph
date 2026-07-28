"""Reading a nested subgraph's state must hydrate its `DeltaChannel`s.

A subgraph obtained through `get_subgraphs()` was compiled WITHOUT a
checkpointer — the parent supplies one via `CONFIG_KEY_CHECKPOINTER` when
reading. `_prepare_state_snapshot` used to hydrate channels with
`self.checkpointer` only, so for a nested subgraph the saver was `None`.

`channels_from_checkpoint` needs that saver to replay a `DeltaChannel`'s
ancestor writes; without it the channel falls through to
`from_checkpoint(MISSING)` and hydrates EMPTY. No error is raised, so a caller
reading a subgraph's history simply sees an empty channel and cannot tell it
from a subgraph that genuinely never wrote one.

Regular (non-delta) channels are unaffected, which is what makes the bug easy to
miss: only reducer channels stored as deltas — `messages` on any agent that
opts into delta storage — come back empty.
"""

from typing import Annotated, Any

import pytest
from langchain_core.messages import AIMessage, AnyMessage, HumanMessage
from langgraph.checkpoint.memory import InMemorySaver
from typing_extensions import TypedDict

from langgraph.channels.delta import DeltaChannel
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import _messages_delta_reducer

pytestmark = pytest.mark.anyio


class ChildState(TypedDict, total=False):
    messages: Annotated[list[AnyMessage], DeltaChannel(_messages_delta_reducer)]
    plain: str


class ParentState(TypedDict, total=False):
    messages: Annotated[list[AnyMessage], DeltaChannel(_messages_delta_reducer)]
    plain: str
    done: bool


def _build() -> Any:
    """Parent graph with one compiled subgraph node writing across two supersteps."""
    child = StateGraph(ChildState)
    child.add_node("first", lambda s: {"messages": [HumanMessage("one")], "plain": "p"})
    child.add_node("second", lambda s: {"messages": [AIMessage("two")]})
    child.add_edge(START, "first")
    child.add_edge("first", "second")
    child.add_edge("second", END)

    parent = StateGraph(ParentState)
    parent.add_node("child", child.compile())
    parent.add_node("finish", lambda s: {"done": True})
    parent.add_edge(START, "child")
    parent.add_edge("child", "finish")
    parent.add_edge("finish", END)
    return parent


def _child_namespace(app: Any, config: dict) -> str:
    """The child's `checkpoint_ns`, discovered the way a client would."""
    for snapshot in app.get_state_history(config):
        for task in snapshot.tasks:
            if task.name == "child" and isinstance(task.state, dict):
                return task.state["configurable"]["checkpoint_ns"]
    raise AssertionError("child namespace not found in the parent's tasks")


def test_subgraph_history_hydrates_delta_channel() -> None:
    app = _build().compile(checkpointer=InMemorySaver())
    config = {"configurable": {"thread_id": "1"}}
    app.invoke({}, config)

    child_ns = _child_namespace(app, config)
    child_config = {"configurable": {"thread_id": "1", "checkpoint_ns": child_ns}}

    best: list[AnyMessage] = []
    for snapshot in app.get_state_history(child_config):
        messages = snapshot.values.get("messages") or []
        if len(messages) > len(best):
            best = messages

    assert [m.content for m in best] == ["one", "two"]


def test_subgraph_get_state_hydrates_delta_channel() -> None:
    app = _build().compile(checkpointer=InMemorySaver())
    config = {"configurable": {"thread_id": "1"}}
    app.invoke({}, config)

    child_ns = _child_namespace(app, config)
    values = app.get_state(
        {"configurable": {"thread_id": "1", "checkpoint_ns": child_ns}}
    ).values
    assert [m.content for m in values.get("messages") or []] == ["one", "two"]
    # A non-delta channel in the same namespace was never affected — it is the
    # contrast that makes the delta-only nature of the bug explicit.
    assert values.get("plain") == "p"


async def test_subgraph_ahistory_hydrates_delta_channel() -> None:
    app = _build().compile(checkpointer=InMemorySaver())
    config = {"configurable": {"thread_id": "1"}}
    await app.ainvoke({}, config)

    child_ns = None
    async for snapshot in app.aget_state_history(config):
        for task in snapshot.tasks:
            if task.name == "child" and isinstance(task.state, dict):
                child_ns = task.state["configurable"]["checkpoint_ns"]
    assert child_ns is not None

    best: list[AnyMessage] = []
    async for snapshot in app.aget_state_history(
        {"configurable": {"thread_id": "1", "checkpoint_ns": child_ns}}
    ):
        messages = snapshot.values.get("messages") or []
        if len(messages) > len(best):
            best = messages

    assert [m.content for m in best] == ["one", "two"]


def test_root_history_still_hydrates_delta_channel() -> None:
    """Control: the root graph has its own checkpointer and always worked."""
    app = _build().compile(checkpointer=InMemorySaver())
    config = {"configurable": {"thread_id": "1"}}
    app.invoke({}, config)
    values = app.get_state(config).values
    assert [m.content for m in values.get("messages") or []] == ["one", "two"]
