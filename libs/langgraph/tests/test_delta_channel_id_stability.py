"""ensure_message_ids() assigns stable UUIDs to id=None BaseMessages
before DeltaChannel writes are serialised to the checkpoint.

Without this, the checkpoint stores id=None and every get_state() replay
produces a different UUID — the same HumanMessage appears with a different
ID in each LangSmith trace / on every resumed invocation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Annotated, Any

import pytest
from langchain_core.messages import AIMessage, AnyMessage, HumanMessage, SystemMessage
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.memory import InMemorySaver
from pydantic import BaseModel
from typing_extensions import TypedDict

from langgraph.channels.delta import DeltaChannel
from langgraph.channels.last_value import LastValue
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import _messages_delta_reducer, add_messages
from langgraph.pregel import NodeBuilder, Pregel
from langgraph.types import Command, Durability, Overwrite, interrupt

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
    channel: Any = None,
) -> Any:
    if channel is None:
        channel = DeltaChannel(reducer, snapshot_frequency=50)
    State = TypedDict(  # noqa: UP013
        "State",
        {"messages": Annotated[list, channel]},
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
    pytest.param(lambda: [HumanMessage("hello", id="")], id="message, empty id"),
    pytest.param(lambda: [("user", "hello")], id="tuple"),
    pytest.param(lambda: ("user", "hello"), id="bare tuple"),
    pytest.param(lambda: ["hello"], id="string"),
    pytest.param(lambda: "hello", id="bare string"),
    pytest.param(lambda: Overwrite(["hello"]), id="Overwrite"),
    pytest.param(lambda: Overwrite(("hello",)), id="Overwrite of a tuple"),
    pytest.param(lambda: Overwrite(HumanMessage("hello")), id="Overwrite of a message"),
    pytest.param(lambda: Overwrite(("user", "hello")), id="Overwrite of a pair"),
    pytest.param(lambda: {"__overwrite__": [("user", "hello")]}, id="Overwrite dict"),
]


def _id(m: Any) -> str | None:
    return m.get("id") if isinstance(m, dict) else m.id


def _hello_ids(graph: Any, config: dict) -> list[str | None]:
    def messages() -> list:
        value = graph.get_state(config).values["messages"]
        return value if isinstance(value, list) else [value]

    return [
        next(
            _id(m)
            for m in messages()
            if (m.get("content") if isinstance(m, dict) else m.content) == "hello"
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
    return Pregel(
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


MESSAGES_CHANNELS = [
    pytest.param(lambda: DeltaChannel(_messages_delta_reducer), id="delta"),
    pytest.param(lambda: add_messages, id="add_messages"),
]


@pytest.mark.parametrize("channel", MESSAGES_CHANNELS)
@pytest.mark.parametrize(
    "write",
    [
        pytest.param(lambda: [{"role": "user", "content": "hello"}], id="dict"),
        pytest.param(lambda: [("user", "hello")], id="tuple"),
        pytest.param(lambda: "hello", id="bare string"),
    ],
)
def test_updates_carry_the_saved_message_ids(channel: Any, write: Any) -> None:
    graph = _build_graph(InMemorySaver(), channel=channel(), write=write)
    config = {"configurable": {"thread_id": "t"}}

    updates = list(graph.stream({"messages": []}, config, stream_mode="updates"))

    streamed = updates[0]["agent"]["messages"]
    streamed = streamed if isinstance(streamed, list) else [streamed]
    first, second = _hello_ids(graph, config)
    assert [_id(m) for m in streamed] == [first], "updates must show the saved id"
    assert first == second


def test_add_messages_keeps_a_message_dict_a_dict() -> None:
    written = {"role": "user", "content": "hello"}
    graph = _build_graph(InMemorySaver(), channel=add_messages, write=lambda: [written])
    config = {"configurable": {"thread_id": "t"}}

    updates = list(graph.stream({"messages": []}, config, stream_mode="updates"))

    (streamed,) = updates[0]["agent"]["messages"]
    assert streamed == {**written, "id": streamed["id"]}
    assert streamed["id"] == _hello_ids(graph, config)[0]
    assert written == {"role": "user", "content": "hello"}, (
        "the node's dict is left as is"
    )


@pytest.mark.parametrize("channel", MESSAGES_CHANNELS)
@pytest.mark.parametrize(
    "write",
    [
        pytest.param(lambda: Overwrite([HumanMessage("hello")]), id="message"),
        pytest.param(
            lambda: Overwrite([{"role": "user", "content": "hello"}]), id="dict"
        ),
        pytest.param(lambda: Overwrite([("user", "hello")]), id="tuple"),
        pytest.param(
            lambda: {"__overwrite__": [{"role": "user", "content": "hello"}]},
            id="__overwrite__ dict",
        ),
        pytest.param(
            lambda: {
                "type": "__overwrite__",
                "value": [{"role": "user", "content": "hello"}],
            },
            id="serialized Overwrite",
        ),
    ],
)
def test_an_overwrite_keeps_its_message_ids_in_a_fork(channel: Any, write: Any) -> None:
    graph = _build_graph(InMemorySaver(), channel=channel(), write=write)
    config = {"configurable": {"thread_id": "t"}}
    graph.invoke({"messages": []}, config)
    first, second = _hello_ids(graph, config)

    forked = graph.update_state(
        graph.get_state(config).config, {"messages": [AIMessage("x", id="x")]}
    )

    assert first, "the message should get a UUID"
    assert first == second, "its id must be the same on every read"
    assert _hello_ids(graph, forked) == [first, first], "a fork must keep it"


@pytest.mark.parametrize("channel", MESSAGES_CHANNELS)
@pytest.mark.parametrize(
    "write",
    [
        pytest.param(lambda: Overwrite(HumanMessage("hello")), id="message"),
        pytest.param(
            lambda: Overwrite({"role": "user", "content": "hello"}), id="dict"
        ),
        pytest.param(lambda: Overwrite("hello"), id="string"),
        pytest.param(lambda: Overwrite(("user", "hello")), id="pair"),
    ],
)
def test_one_message_in_an_overwrite_stays_one_message(
    channel: Any, write: Any
) -> None:
    graph = _build_graph(InMemorySaver(), channel=channel(), write=write)
    config = {"configurable": {"thread_id": "t"}}

    graph.invoke({"messages": []}, config)
    graph.invoke({"messages": [("user", "next")]}, config)

    value = graph.get_state(config).values["messages"]
    messages = value if isinstance(value, list) else [value]
    assert [
        m.get("content") if isinstance(m, dict) else m.content for m in messages
    ] == ["hello"]
    first, second = _hello_ids(graph, config)
    assert first, "the message should get a UUID"
    assert first == second, "its id must be the same on every read"


@pytest.mark.parametrize("channel", MESSAGES_CHANNELS)
@pytest.mark.parametrize(
    "input",
    [
        pytest.param(lambda: [HumanMessage("hello")], id="message"),
        pytest.param(lambda: [{"role": "user", "content": "hello"}], id="dict"),
        pytest.param(lambda: "hello", id="bare string"),
    ],
)
def test_a_replay_from_the_first_checkpoint_keeps_the_input_message_ids(
    channel: Any, input: Any, durability: Durability
) -> None:
    graph = _build_graph(InMemorySaver(), channel=channel())
    config = {"configurable": {"thread_id": "t"}}
    graph.invoke({"messages": input()}, config, durability=durability)
    first, _ = _hello_ids(graph, config)
    # `exit` saves no input checkpoint; a DeltaChannel anchor has no `source`.
    oldest = [s for s in graph.get_state_history(config) if "source" in s.metadata][-1]

    graph.invoke(None, oldest.config)

    assert first, "the message should get a UUID"
    assert _hello_ids(graph, config) == [first, first]


class _PydanticState(BaseModel):
    messages: Annotated[list, add_messages] = []


@dataclass
class _DataclassState:
    messages: Annotated[list, add_messages] = field(default_factory=list)


@dataclass
class _SystemPromptState:
    messages: Annotated[list, add_messages] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.messages = [SystemMessage("be brief"), *self.messages]


@dataclass(frozen=True)
class _FrozenDataclassState:
    messages: Annotated[list, add_messages] = field(default_factory=list)


@pytest.mark.parametrize(
    "schema",
    [_PydanticState, _DataclassState, _FrozenDataclassState, _SystemPromptState],
)
@pytest.mark.parametrize(
    "input",
    [
        pytest.param(lambda: [HumanMessage("hello")], id="message"),
        pytest.param(lambda: [{"role": "user", "content": "hello"}], id="dict"),
        pytest.param(lambda: [("user", "hello")], id="tuple"),
    ],
)
def test_a_state_object_input_saves_its_message_ids(schema: Any, input: Any) -> None:
    graph = (
        StateGraph(schema)
        .add_node("agent", lambda state: {"messages": [AIMessage("reply")]})
        .add_edge(START, "agent")
        .compile(checkpointer=InMemorySaver())
    )
    config = {"configurable": {"thread_id": "t"}}
    state = schema(messages=input())
    input_types = [type(m) for m in state.messages]
    graph.invoke(state, config)
    saved = graph.get_state(config).values["messages"]
    assert len(saved) == len(state.messages) + 1, "the input is saved as built"
    first, _ = _hello_ids(graph, config)
    oldest = list(graph.get_state_history(config))[-1]

    graph.invoke(None, oldest.config)

    assert first, "the message should get a UUID"
    assert _hello_ids(graph, config) == [first, first]
    assert [type(m) for m in state.messages] == input_types, "the input is left as is"


def test_a_dict_input_keeps_its_keys_and_their_order() -> None:
    State = TypedDict(  # noqa: UP013
        "State", {"topic": str, "messages": Annotated[list, add_messages]}
    )  # type: ignore[call-overload]
    graph = (
        StateGraph(State)
        .add_node("agent", lambda state: {})
        .add_edge(START, "agent")
        .compile(checkpointer=InMemorySaver())
    )
    config = {"configurable": {"thread_id": "t"}}
    input = {
        "topic": "x",
        "messages": [{"role": "user", "content": "hello"}],
        "other": 1,
    }

    graph.invoke(input, config)

    saved = list(graph.checkpointer.list(config))[-1].checkpoint["channel_values"]
    (message,) = saved["__start__"]["messages"]
    assert list(saved["__start__"]) == ["topic", "messages", "other"]
    assert saved["__start__"]["topic"] == "x" and saved["__start__"]["other"] == 1
    assert message == {"role": "user", "content": "hello", "id": message["id"]}
    assert message["id"]


@pytest.mark.parametrize(
    "input",
    [
        pytest.param(lambda: [HumanMessage("hello")], id="message"),
        pytest.param(lambda: {"role": "user", "content": "hello"}, id="dict"),
        pytest.param(lambda: "hello", id="bare string"),
    ],
)
def test_a_root_channel_input_saves_its_message_ids(input: Any) -> None:
    graph = (
        StateGraph(Annotated[list, add_messages])
        .add_node("agent", lambda state: [AIMessage("reply")])
        .add_edge(START, "agent")
        .compile(checkpointer=InMemorySaver())
    )
    config = {"configurable": {"thread_id": "t"}}

    def hello_id() -> str | None:
        return next(
            _id(m)
            for m in graph.get_state(config).values
            if (m.get("content") if isinstance(m, dict) else m.content) == "hello"
        )

    graph.invoke(input(), config)
    first = hello_id()
    oldest = list(graph.get_state_history(config))[-1]

    graph.invoke(None, oldest.config)

    assert first, "the message should get a UUID"
    assert hello_id() == first


def test_an_overwrite_on_a_formatting_add_messages_channel_gets_a_stable_id() -> None:
    graph = _build_graph(
        InMemorySaver(),
        channel=add_messages(format="langchain-openai"),
        write=lambda: Overwrite([HumanMessage("hello")]),
    )
    config = {"configurable": {"thread_id": "t"}}

    graph.invoke({"messages": []}, config)

    first, second = _hello_ids(graph, config)
    assert first, "the message should get a UUID"
    assert first == second, "its id must be the same on every read"


def test_add_messages_keeps_an_empty_id() -> None:
    graph = _build_graph(
        InMemorySaver(),
        channel=add_messages,
        write=lambda: [HumanMessage("hello", id="")],
    )
    config = {"configurable": {"thread_id": "t"}}

    graph.invoke({"messages": []}, config)

    assert _hello_ids(graph, config) == ["", ""]


def _hello_write_id(checkpointer: BaseCheckpointSaver, config: dict) -> str | None:
    for saved in checkpointer.list(config):
        for _, channel, value in saved.pending_writes or []:
            if channel != "messages":
                continue
            for m in value if isinstance(value, list) else [value]:
                content = m.get("content") if isinstance(m, dict) else m.content
                if content == "hello":
                    return _id(m)
    raise AssertionError("no saved write holds the message")


@pytest.mark.parametrize(
    "write",
    [
        pytest.param(lambda: Overwrite(["hello"]), id="Overwrite"),
        pytest.param(lambda: {"__overwrite__": ["hello"]}, id="__overwrite__ dict"),
        pytest.param(
            lambda: {"type": "__overwrite__", "value": ["hello"]},
            id="serialized Overwrite",
        ),
    ],
)
def test_an_overwrite_streams_in_the_form_it_was_written(write: Any) -> None:
    graph = _build_graph(InMemorySaver(), channel=add_messages, write=write)
    config = {"configurable": {"thread_id": "t"}}

    (update,) = graph.stream({"messages": []}, config, stream_mode="updates")

    streamed, written = update["agent"]["messages"], write()
    assert type(streamed) is type(written)
    if isinstance(written, dict):
        assert streamed.keys() == written.keys()


@pytest.mark.parametrize("channel", MESSAGES_CHANNELS)
def test_a_resumed_step_applies_the_message_id_it_saved(
    channel: Any, sync_checkpointer: BaseCheckpointSaver
) -> None:
    State = TypedDict("State", {"messages": Annotated[list, channel()]})  # noqa: UP013

    def ask(state: dict) -> dict:
        interrupt("continue?")
        return {}

    graph = (
        StateGraph(State)
        .add_node("write", lambda state: {"messages": [("user", "hello")]})
        .add_node("ask", ask)
        .add_edge(START, "write")
        .add_edge(START, "ask")
        .compile(checkpointer=sync_checkpointer)
    )
    config = {"configurable": {"thread_id": "t"}}
    graph.invoke({"messages": []}, config)
    saved = _hello_write_id(sync_checkpointer, config)

    graph.invoke(Command(resume=True), config)

    assert saved, "the write saved at the interrupt should carry an id"
    assert _hello_ids(graph, config) == [saved, saved]


@pytest.mark.parametrize("channel", MESSAGES_CHANNELS)
def test_a_command_update_applies_the_message_id_it_saved(
    channel: Any, sync_checkpointer: BaseCheckpointSaver
) -> None:
    graph = _build_graph(sync_checkpointer, channel=channel(), write=lambda: [])
    config = {"configurable": {"thread_id": "t"}}
    graph.invoke({"messages": [HumanMessage("a", id="a")]}, config)

    graph.invoke(
        Command(update={"messages": [{"role": "user", "content": "hello"}]}), config
    )

    saved = _hello_write_id(sync_checkpointer, config)
    assert saved, "the saved write should carry an id"
    assert _hello_ids(graph, config) == [saved, saved]


@pytest.mark.parametrize("channel", MESSAGES_CHANNELS)
def test_update_state_applies_the_message_id_it_saved(
    channel: Any, sync_checkpointer: BaseCheckpointSaver
) -> None:
    graph = _build_graph(sync_checkpointer, channel=channel(), write=lambda: [])
    config = {"configurable": {"thread_id": "t"}}
    graph.invoke({"messages": [HumanMessage("a", id="a")]}, config)

    updated = graph.update_state(
        config, {"messages": [{"role": "user", "content": "hello"}]}
    )

    saved = _hello_write_id(sync_checkpointer, config)
    assert saved, "the saved write should carry an id"
    assert _hello_ids(graph, updated) == [saved, saved]


async def _ahello_ids(graph: Any, config: dict) -> list[str | None]:
    return [
        next(
            _id(m)
            for m in (await graph.aget_state(config)).values["messages"]
            if (m.get("content") if isinstance(m, dict) else m.content) == "hello"
        )
        for _ in range(2)
    ]


@pytest.mark.parametrize("channel", MESSAGES_CHANNELS)
async def test_an_async_run_keeps_the_message_ids_it_saves(
    channel: Any, async_checkpointer: BaseCheckpointSaver
) -> None:
    graph = _build_graph(
        async_checkpointer,
        channel=channel(),
        write=lambda: [{"role": "assistant", "content": "reply"}],
    )
    config = {"configurable": {"thread_id": "t"}}

    updates = [
        c
        async for c in graph.astream(
            {"messages": [{"role": "user", "content": "hello"}]},
            config,
            stream_mode="updates",
        )
    ]
    (streamed,) = updates[-1]["agent"]["messages"]
    state = (await graph.aget_state(config)).values["messages"]
    assert [_id(m) for m in state][-1] == _id(streamed), "updates show the saved id"
    first, second = await _ahello_ids(graph, config)
    assert first and first == second

    oldest = [
        s async for s in graph.aget_state_history(config) if "source" in s.metadata
    ][-1]
    await graph.ainvoke(None, oldest.config)
    assert await _ahello_ids(graph, config) == [first, first], "replay keeps it"

    updated = await graph.aupdate_state(
        config, {"messages": [{"role": "user", "content": "hello again"}]}
    )
    again = [
        _id(m)
        for m in (await graph.aget_state(updated)).values["messages"]
        if (m.get("content") if isinstance(m, dict) else m.content) == "hello again"
    ]
    assert again and again[0], "aupdate_state gives the message an id"
