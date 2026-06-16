"""ensure_message_ids() assigns stable UUIDs to id=None BaseMessages
before DeltaChannel writes are serialised to the checkpoint.

Without this, the checkpoint stores id=None and every get_state() replay
produces a different UUID — the same HumanMessage appears with a different
ID in each LangSmith trace / on every resumed invocation.
"""

from __future__ import annotations

from typing import Annotated, Any

import pytest
from langchain_core.messages import AIMessage, AnyMessage, HumanMessage
from langgraph.checkpoint.memory import InMemorySaver
from typing_extensions import TypedDict

from langgraph.channels.delta import DeltaChannel
from langgraph.graph import END, START, StateGraph

pytestmark = pytest.mark.anyio


def _append_reducer(
    state: list[AnyMessage], writes: list[list[AnyMessage]]
) -> list[AnyMessage]:
    """Simple append — no ID assignment. IDs come from ensure_message_ids()."""
    result = list(state)
    for w in writes:
        if isinstance(w, list):
            result.extend(w)
        else:
            result.append(w)  # type: ignore[arg-type]
    return result


def _build_graph(checkpointer: Any) -> Any:
    State = TypedDict(  # noqa: UP013
        "State",
        {
            "messages": Annotated[
                list, DeltaChannel(_append_reducer, snapshot_frequency=50)
            ]
        },
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


def test_delta_channel_message_gets_id_and_stays_stable() -> None:
    """Messages written with id=None must receive a stable UUID.

    ensure_message_ids() is called in put_writes() before the background
    thread serialises DeltaChannel writes. The checkpoint stores the
    assigned UUID, so every get_state() replay sees the same ID.
    """
    saver = InMemorySaver()
    graph = _build_graph(saver)
    config = {"configurable": {"thread_id": "id-stability"}}

    graph.invoke({"messages": [HumanMessage(content="hello")]}, config)

    ids = [
        next(
            m.id
            for m in graph.get_state(config).values["messages"]
            if isinstance(m, HumanMessage)
        )
        for _ in range(3)
    ]

    assert ids[0] is not None, "ensure_message_ids should have assigned a UUID"
    assert len(set(ids)) == 1, (
        f"HumanMessage id must be stable across get_state() calls; "
        f"got {ids}. The checkpoint is storing id=None."
    )


async def test_delta_channel_message_gets_id_and_stays_stable_async() -> None:
    """Same check via ainvoke (AsyncPregelLoop path)."""
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

    assert ids[0] is not None, "ensure_message_ids should have assigned a UUID"
    assert len(set(ids)) == 1, (
        f"Async path: HumanMessage id unstable across aget_state() calls: {ids}"
    )


def test_delta_channel_dict_style_message_gets_stable_id() -> None:
    """Dict-style inputs (API / over-the-wire format) must also get stable IDs.

    When the graph is invoked via the LangGraph API the input arrives as a raw
    dict {"role": "user", "content": "..."} rather than a BaseMessage object.
    ensure_message_ids() must coerce those dicts to typed BaseMessages and
    stamp a UUID so the checkpoint never stores an id-less message.
    """
    saver = InMemorySaver()
    graph = _build_graph(saver)
    config = {"configurable": {"thread_id": "dict-id-stability"}}

    # Invoke with a raw dict (the format LangGraph API sends)
    graph.invoke({"messages": [{"role": "user", "content": "hello"}]}, config)

    ids = [
        next(
            m.id
            for m in graph.get_state(config).values["messages"]
            if isinstance(m, HumanMessage)
        )
        for _ in range(3)
    ]

    assert ids[0] is not None, (
        "dict-style message should have been coerced and assigned a UUID"
    )
    assert len(set(ids)) == 1, (
        f"dict-style HumanMessage id must be stable across get_state() calls; got {ids}"
    )
