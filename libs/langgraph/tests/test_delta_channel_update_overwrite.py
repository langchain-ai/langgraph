"""An `Overwrite` through `update_state` snapshots its DeltaChannel on the
checkpoint the update saves, as a node's `Overwrite` does on the loop's."""

from typing import Annotated, Any

import pytest
from langchain_core.messages import HumanMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.checkpoint.serde.types import _DeltaSnapshot
from typing_extensions import TypedDict

from langgraph.channels.delta import DeltaChannel
from langgraph.channels.last_value import LastValue
from langgraph.graph import START, StateGraph
from langgraph.graph.message import _messages_delta_reducer
from langgraph.pregel import NodeBuilder, Pregel
from langgraph.types import Overwrite

pytestmark = pytest.mark.anyio


class _State(TypedDict):
    messages: Annotated[list, DeltaChannel(_messages_delta_reducer)]


def _messages_graph(saver: InMemorySaver) -> Any:
    builder = StateGraph(_State)
    builder.add_node("model", lambda state: {})
    builder.add_edge(START, "model")
    return builder.compile(checkpointer=saver)


def _extend(current: list, writes: list) -> list:
    return [*current, *(item for write in writes for item in write)]


def _delta_input_graph(saver: InMemorySaver) -> Pregel:
    node = NodeBuilder().subscribe_only("go").do(lambda _: [2]).write_to("log")
    return Pregel(
        nodes={"n": node},
        channels={"log": DeltaChannel(_extend), "go": LastValue(int)},
        input_channels=["log", "go"],
        output_channels=["log"],
        checkpointer=saver,
    )


def test_update_state_with_an_overwrite_snapshots_the_channel() -> None:
    saver = InMemorySaver()
    graph = _messages_graph(saver)
    config = {"configurable": {"thread_id": "t"}}
    graph.invoke({"messages": [HumanMessage(content="a", id="1")]}, config)

    graph.update_state(
        config,
        {"messages": Overwrite([HumanMessage(content="b", id="2")])},
        as_node="model",
    )

    head = saver.get_tuple(config)
    assert head is not None
    assert isinstance(head.checkpoint["channel_values"].get("messages"), _DeltaSnapshot)
    assert [m.content for m in graph.get_state(config).values["messages"]] == ["b"]


async def test_aupdate_state_with_an_overwrite_snapshots_the_channel() -> None:
    saver = InMemorySaver()
    graph = _messages_graph(saver)
    config = {"configurable": {"thread_id": "t"}}
    await graph.ainvoke({"messages": [HumanMessage(content="a", id="1")]}, config)

    await graph.aupdate_state(
        config,
        {"messages": Overwrite([HumanMessage(content="b", id="2")])},
        as_node="model",
    )

    head = await saver.aget_tuple(config)
    assert head is not None
    assert isinstance(head.checkpoint["channel_values"].get("messages"), _DeltaSnapshot)
    values = (await graph.aget_state(config)).values
    assert [m.content for m in values["messages"]] == ["b"]


def test_update_state_as_input_with_an_overwrite_snapshots_the_channel() -> None:
    saver = InMemorySaver()
    graph = _delta_input_graph(saver)
    config = {"configurable": {"thread_id": "t"}}
    graph.invoke({"log": [0], "go": 1}, config)

    graph.update_state(config, {"log": Overwrite([1])}, as_node="__input__")

    head = saver.get_tuple(config)
    assert head is not None
    assert isinstance(head.checkpoint["channel_values"].get("log"), _DeltaSnapshot)
    assert graph.get_state(config).values["log"] == [1]


async def test_aupdate_state_as_input_with_an_overwrite_snapshots_the_channel() -> None:
    saver = InMemorySaver()
    graph = _delta_input_graph(saver)
    config = {"configurable": {"thread_id": "t"}}
    await graph.ainvoke({"log": [0], "go": 1}, config)

    await graph.aupdate_state(config, {"log": Overwrite([1])}, as_node="__input__")

    head = await saver.aget_tuple(config)
    assert head is not None
    assert isinstance(head.checkpoint["channel_values"].get("log"), _DeltaSnapshot)
    assert (await graph.aget_state(config)).values["log"] == [1]
