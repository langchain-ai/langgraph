import operator
from typing import Annotated, Any
from unittest.mock import patch

import pytest
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.checkpoint.serde.types import _DeltaSnapshot
from typing_extensions import TypedDict

from langgraph.channels.binop import BinaryOperatorAggregate
from langgraph.channels.delta import DeltaChannel
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import _messages_delta_reducer, add_messages
from langgraph.pregel._checkpoint import channels_from_checkpoint, empty_checkpoint
from langgraph.types import Overwrite

pytestmark = pytest.mark.anyio


class DeltaState(TypedDict):
    messages: Annotated[
        list, DeltaChannel(_messages_delta_reducer, snapshot_frequency=3)
    ]


class PlainState(TypedDict):
    messages: Annotated[list, add_messages]


def _app(schema: type, checkpointer: BaseCheckpointSaver) -> Any:
    builder = StateGraph(schema)
    builder.add_node(
        "chat",
        lambda state: {"messages": [AIMessage(f"reply {len(state['messages'])}")]},
    )
    builder.add_edge(START, "chat")
    builder.add_edge("chat", END)
    return builder.compile(checkpointer=checkpointer)


def _delta_thread(checkpointer: BaseCheckpointSaver) -> RunnableConfig:
    config: RunnableConfig = {"configurable": {"thread_id": "1"}}
    delta = _app(DeltaState, checkpointer)
    for i in range(4):
        delta.invoke({"messages": [HumanMessage(f"q{i}")]}, config)
    assert any(
        isinstance(t.checkpoint["channel_values"].get("messages"), _DeltaSnapshot)
        for t in checkpointer.list(config)
    ), "the thread should have a snapshot to read through"
    return config


def test_plain_reducer_reads_every_delta_written_checkpoint(
    sync_checkpointer: BaseCheckpointSaver,
) -> None:
    config = _delta_thread(sync_checkpointer)
    delta = _app(DeltaState, sync_checkpointer)
    plain = _app(PlainState, sync_checkpointer)

    assert [s.values for s in plain.get_state_history(config)] == [
        s.values for s in delta.get_state_history(config)
    ]


def test_plain_reducer_continues_delta_written_thread(
    sync_checkpointer: BaseCheckpointSaver,
) -> None:
    config = _delta_thread(sync_checkpointer)
    delta = _app(DeltaState, sync_checkpointer)
    before = delta.get_state(config).values["messages"]

    after = _app(PlainState, sync_checkpointer).invoke(
        {"messages": [HumanMessage("q4")]}, config
    )["messages"]

    assert after[:-2] == before
    assert [m.content for m in after[-2:]] == ["q4", f"reply {len(before) + 1}"]
    head = sync_checkpointer.get_tuple(config)
    assert head is not None
    assert head.checkpoint["channel_values"]["messages"] == after
    assert delta.get_state(config).values["messages"] == after


async def test_plain_reducer_reads_and_continues_delta_written_thread_async(
    async_checkpointer: BaseCheckpointSaver,
) -> None:
    config: RunnableConfig = {"configurable": {"thread_id": "1"}}
    delta = _app(DeltaState, async_checkpointer)
    for i in range(4):
        await delta.ainvoke({"messages": [HumanMessage(f"q{i}")]}, config)
    plain = _app(PlainState, async_checkpointer)

    expected = [s.values async for s in delta.aget_state_history(config)]
    assert [s.values async for s in plain.aget_state_history(config)] == expected

    after = (await plain.ainvoke({"messages": [HumanMessage("q4")]}, config))[
        "messages"
    ]
    assert after[:-2] == expected[0]["messages"]


@pytest.mark.parametrize(
    ("seed", "writes", "expected"),
    [
        (_DeltaSnapshot(["a"]), [["b"], ["c"]], ["a", "b", "c"]),
        (["a"], [["b"], Overwrite(["c"]), ["d"]], ["c", "d"]),
        (None, [["b"], Overwrite(["c"]), Overwrite(["d"]), ["e"]], ["d", "e"]),
    ],
)
def test_plain_reducer_replays_delta_history_one_write_at_a_time(
    seed: Any, writes: list[Any], expected: list[str]
) -> None:
    history: dict[str, Any] = {"writes": [("t", "items", w) for w in writes]}
    if seed is not None:
        history["seed"] = seed
    checkpoint = empty_checkpoint()
    checkpoint["channel_versions"]["items"] = 1
    saver = InMemorySaver()

    with patch.object(
        saver, "get_delta_channel_history", return_value={"items": history}
    ):
        channels, _ = channels_from_checkpoint(
            {"items": BinaryOperatorAggregate(list, operator.add)},
            checkpoint,
            saver=saver,
            config={"configurable": {"thread_id": "1"}},
        )

    assert channels["items"].get() == expected


def test_plain_thread_reads_never_walk_history() -> None:
    saver = InMemorySaver()
    config: RunnableConfig = {"configurable": {"thread_id": "1"}}
    plain = _app(PlainState, saver)

    with patch.object(saver, "get_delta_channel_history") as walk:
        plain.invoke({"messages": [HumanMessage("q0")]}, config)
        plain.update_state(config, {"messages": [HumanMessage("note")]})
        plain.invoke({"messages": [HumanMessage("q1")]}, config)
        plain.get_state(config)
        list(plain.get_state_history(config))

    walk.assert_not_called()
