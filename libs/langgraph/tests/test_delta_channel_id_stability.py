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
from langgraph.channels.last_value import LastValue
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import _messages_delta_reducer
from langgraph.types import Command, Durability, Overwrite
from tests.raw_pregel import NodeBuilder, RawPregel

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


def _build_graph(
    checkpointer: Any,
    *,
    reducer: Any = _append_reducer,
    write: Any = lambda: [AIMessage(content="reply", id="ai-1")],
) -> Any:
    State = TypedDict(  # noqa: UP013
        "State",
        {"messages": Annotated[list, DeltaChannel(reducer, snapshot_frequency=50)]},
    )  # type: ignore[call-overload]

    def agent(state: dict) -> dict:
        return {"messages": write()}

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


@pytest.mark.parametrize("container", [list, tuple])
def test_delta_channel_message_in_an_overwrite_gets_stable_id(container: type) -> None:
    graph = _build_graph(InMemorySaver())
    config = {"configurable": {"thread_id": "overwrite-id-stability"}}

    graph.invoke(
        {"messages": Overwrite(container([HumanMessage(content="hello")]))}, config
    )

    ids = [
        next(
            m.id
            for m in graph.get_state(config).values["messages"]
            if isinstance(m, HumanMessage)
        )
        for _ in range(3)
    ]

    assert ids[0] is not None, "a message inside an Overwrite should get a UUID"
    assert len(set(ids)) == 1, (
        f"its id must be stable across get_state() calls; got {ids}"
    )


# Writes of "hello" in every shape `_messages_delta_reducer` accepts.
MESSAGE_SHAPES = [
    pytest.param(lambda: [HumanMessage("hello")], id="message"),
    pytest.param(lambda: HumanMessage("hello"), id="bare message"),
    pytest.param(lambda: [{"role": "user", "content": "hello"}], id="dict"),
    pytest.param(lambda: {"type": "human", "content": "hello"}, id="bare dict"),
    pytest.param(
        lambda: [{"role": "user", "content": "hello", "id": ""}], id="dict, empty id"
    ),
    pytest.param(lambda: [("user", "hello")], id="tuple"),
    pytest.param(lambda: ("user", "hello"), id="bare tuple"),
    pytest.param(lambda: ["hello"], id="string"),
    pytest.param(lambda: "hello", id="bare string"),
    pytest.param(lambda: Overwrite(["hello"]), id="Overwrite"),
    pytest.param(lambda: Overwrite(("hello",)), id="Overwrite of a tuple"),
    pytest.param(lambda: {"__overwrite__": [("user", "hello")]}, id="Overwrite dict"),
]


def _hello_ids(graph: Any, config: dict) -> list[str | None]:
    return [
        next(
            m.id
            for m in graph.get_state(config).values["messages"]
            if m.content == "hello"
        )
        for _ in range(2)
    ]


@pytest.mark.parametrize("write", MESSAGE_SHAPES)
def test_input_message_of_any_shape_gets_stable_id(write: Any) -> None:
    graph = _build_graph(InMemorySaver(), reducer=_messages_delta_reducer)
    config = {"configurable": {"thread_id": "t"}}

    graph.invoke({"messages": write()}, config)

    first, second = _hello_ids(graph, config)
    assert first, "the message should get a UUID"
    assert first == second, "its id must be the same on every read"


@pytest.mark.parametrize("write", MESSAGE_SHAPES)
def test_node_message_of_any_shape_gets_stable_id(write: Any) -> None:
    graph = _build_graph(InMemorySaver(), reducer=_messages_delta_reducer, write=write)
    config = {"configurable": {"thread_id": "t"}}

    graph.invoke({"messages": []}, config)

    first, second = _hello_ids(graph, config)
    assert first, "the message should get a UUID"
    assert first == second, "its id must be the same on every read"


def test_a_list_written_twice_adds_its_messages_twice() -> None:
    turn = [("user", "hello")]
    graph = _build_graph(InMemorySaver(), reducer=_messages_delta_reducer)
    config = {"configurable": {"thread_id": "t"}}

    graph.invoke({"messages": turn}, config)
    graph.invoke({"messages": turn}, config)

    hellos = [
        m.id for m in graph.get_state(config).values["messages"] if m.content == "hello"
    ]
    assert len(set(hellos)) == 2, f"each write should add a message; got {hellos}"
    assert turn == [("user", "hello")], "the caller's list should be left as it was"


def test_a_forked_command_update_gets_a_stable_id() -> None:
    graph = _build_graph(InMemorySaver(), reducer=_messages_delta_reducer)
    config = {"configurable": {"thread_id": "t"}}
    graph.invoke({"messages": [HumanMessage("a", id="a")]}, config)
    before_agent = next(
        s.config for s in graph.get_state_history(config) if s.next == ("agent",)
    )

    graph.invoke(Command(update={"messages": "hello"}), before_agent)

    first, second = _hello_ids(graph, config)
    assert first, "the message should get a UUID"
    assert first == second, "its id must be the same on every read"


def test_a_message_dict_the_reducer_rejects_fails_the_run_before_it_is_saved() -> None:
    graph = _build_graph(
        InMemorySaver(),
        reducer=_messages_delta_reducer,
        write=lambda: [{"role": "tool", "content": "x"}],
    )
    config = {"configurable": {"thread_id": "t"}}

    with pytest.raises(KeyError):
        graph.invoke({"messages": [HumanMessage("a", id="a")]}, config)

    assert [m.id for m in graph.get_state(config).values["messages"]] == ["a"]


def _pregel_graph(seen: list[list[str | None]]) -> Any:
    def look(state: dict) -> list:
        seen.append([m.id for m in state["messages"]])
        return []

    node = (
        NodeBuilder()
        .subscribe_to("go", read=False)
        .read_from("messages")
        .do(look)
        .write_to("messages")
    )
    return RawPregel(
        nodes={"look": node},
        channels={
            "messages": DeltaChannel(_messages_delta_reducer),
            "go": LastValue(int),
        },
        input_channels=["messages", "go"],
        output_channels=["messages"],
        checkpointer=InMemorySaver(),
    )


def test_pregel_input_message_gets_stable_id_on_a_new_thread(
    durability: Durability,
) -> None:
    graph = _pregel_graph([])
    config = {"configurable": {"thread_id": "t"}}

    graph.invoke(
        {"messages": [HumanMessage("hello")], "go": 1}, config, durability=durability
    )

    first, second = _hello_ids(graph, config)
    assert first, "the message should get a UUID"
    assert first == second, "its id must be the same on every read"


def test_a_node_reads_the_pregel_input_message_ids_that_are_saved() -> None:
    seen: list[list[str | None]] = []
    graph = _pregel_graph(seen)
    config = {"configurable": {"thread_id": "t"}}
    graph.invoke({"messages": [HumanMessage("first", id="first")], "go": 1}, config)

    graph.invoke({"messages": [{"role": "user", "content": "hello"}], "go": 1}, config)

    saved = [m.id for m in graph.get_state(config).values["messages"]]
    assert None not in saved
    assert seen[-1] == saved, "the node must see the ids that are saved"


@pytest.mark.parametrize(
    "value",
    [
        pytest.param({"type": "tool", "name": "search"}, id="not a message"),
        pytest.param({"role": "user", "content": "x"}, id="message-shaped"),
    ],
)
def test_other_delta_channels_keep_the_dicts_in_an_overwrite(value: dict) -> None:
    graph = _build_graph(InMemorySaver(), write=lambda: Overwrite([dict(value)]))
    config = {"configurable": {"thread_id": "t"}}

    graph.invoke({"messages": []}, config)

    assert graph.get_state(config).values["messages"] == [value]
