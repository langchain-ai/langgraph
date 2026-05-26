"""DeltaChannel write-ordering: reducer mutations must be captured before persistence.

With durability="durable" (default), put_writes() submits checkpoint writes on a
background thread before apply_writes() runs. For DeltaChannel, the reducer runs
inside apply_writes() and may assign IDs to messages in-place. If the background
thread serializes the write first, it captures id=None; every subsequent get_state()
call replays that id=None through the reducer and gets a fresh UUID — making message
IDs unstable across calls.

The fix: defer DeltaChannel writes in put_writes() and flush them in after_tick()
*after* apply_writes() has run, so in-place mutations are always captured.
"""
from __future__ import annotations

from typing import Annotated, Any

import pytest
from langchain_core.messages import AIMessage, HumanMessage

pytestmark = pytest.mark.anyio
from langgraph.checkpoint.memory import InMemorySaver
from typing_extensions import TypedDict

import uuid

from langchain_core.messages import RemoveMessage
from langgraph.channels.delta import DeltaChannel
from langgraph.graph import END, START, StateGraph


def _reducer_with_id_assignment(
    state: list[AnyMessage], writes: list[list[AnyMessage]]
) -> list[AnyMessage]:
    """Minimal delta reducer that assigns UUIDs to id=None messages in-place.

    This mirrors the pattern used in deepagents' _messages_delta_reducer and
    any reducer that stamps IDs on arrival. The in-place mutation is what makes
    the write-ordering race observable: if the background thread serializes
    before apply_writes() runs the reducer, id=None is captured in the
    checkpoint and every replay produces a fresh UUID.
    """
    flat: list[AnyMessage] = []
    for w in writes:
        if isinstance(w, list):
            flat.extend(w)
        else:
            flat.append(w)  # type: ignore[arg-type]

    index: dict[str, int] = {}
    result: list[AnyMessage | None] = []
    for m in state:
        if m.id is None:
            m.id = str(uuid.uuid4())
        index[m.id] = len(result)
        result.append(m)
    for msg in flat:
        mid = msg.id
        if mid is None:
            msg.id = str(uuid.uuid4())  # in-place mutation — the key ingredient
            mid = msg.id
            index[mid] = len(result)
            result.append(msg)
        elif isinstance(msg, RemoveMessage):
            if mid in index:
                result[index[mid]] = None
                del index[mid]
        elif mid in index:
            result[index[mid]] = msg
        else:
            index[mid] = len(result)
            result.append(msg)
    return [m for m in result if m is not None]


def _build_graph(checkpointer: Any) -> Any:
    State = TypedDict(  # noqa: UP013
        "State",
        {"messages": Annotated[list, DeltaChannel(_reducer_with_id_assignment, snapshot_frequency=50)]},
    )  # type: ignore[call-overload]

    def agent(state: dict) -> dict:
        return {"messages": [AIMessage(content="reply", id="ai-1")]}

    return (
        StateGraph(State)
        .add_node("agent", agent)
        .add_edge(START, "agent")
        .add_edge("agent", END)
        .compile(checkpointer=checkpointer)
    )


def test_delta_channel_human_message_id_stable_across_get_state_calls() -> None:
    """get_state() must return the same HumanMessage id on every call.

    Without the fix, the default durability mode serializes the write before
    apply_writes() assigns the id, so every get_state() replays id=None through
    the reducer and gets a different UUID.
    """
    saver = InMemorySaver()
    graph = _build_graph(saver)
    config = {"configurable": {"thread_id": "id-stability"}}

    # Default durability (no argument) — this is the common case for open-source users.
    graph.invoke({"messages": [HumanMessage(content="hello")]}, config)

    ids = [
        next(
            m.id
            for m in graph.get_state(config).values["messages"]
            if isinstance(m, HumanMessage)
        )
        for _ in range(3)
    ]

    assert ids[0] is not None, "reducer should have assigned a message ID"
    assert len(set(ids)) == 1, (
        f"HumanMessage id must be stable across get_state() calls; "
        f"got different ids on each call: {ids}. "
        "This means id=None was serialized to the checkpoint and the reducer "
        "assigned a fresh UUID on every replay."
    )


async def test_delta_channel_human_message_id_stable_async() -> None:
    """Same check for the async (AsyncPregelLoop) path."""
    saver = InMemorySaver()
    graph = _build_graph(saver)
    config = {"configurable": {"thread_id": "id-stability-async"}}

    await graph.ainvoke({"messages": [HumanMessage(content="hello")]}, config)

    ids = [
        next(
            m.id
            for m in (await graph.aget_state(config)).values["messages"]
            if isinstance(m, HumanMessage)
        )
        for _ in range(3)
    ]

    assert ids[0] is not None
    assert len(set(ids)) == 1, (
        f"Async path: HumanMessage id unstable across aget_state() calls: {ids}"
    )
