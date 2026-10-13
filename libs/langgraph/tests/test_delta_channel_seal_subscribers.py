import pytest
from langgraph.checkpoint.memory import InMemorySaver

from langgraph.channels.delta import DeltaChannel
from langgraph.channels.last_value import LastValue
from langgraph.pregel import NodeBuilder, Pregel

pytestmark = pytest.mark.anyio

CONFIG = {"configurable": {"thread_id": "t"}}


def _extend(current: list, writes: list) -> list:
    return [*current, *(item for write in writes for item in write)]


def _graph(reads: list) -> Pregel:
    writer = NodeBuilder().subscribe_only("a").do(lambda _: [1]).write_to("d")
    reader = NodeBuilder().subscribe_only("d").do(lambda d: reads.append(list(d)))
    return Pregel(
        nodes={"writer": writer, "reader": reader},
        channels={"a": LastValue(str), "d": DeltaChannel(_extend)},
        input_channels="a",
        output_channels=["d"],
        checkpointer=InMemorySaver(),
    )


def test_an_input_update_from_before_the_first_write_starts_only_the_writer() -> None:
    reads: list = []
    graph = _graph(reads)
    graph.invoke("go", CONFIG)
    first = next(
        s.config for s in graph.get_state_history(CONFIG) if s.metadata["step"] == -1
    )

    fork = graph.update_state(first, {"a": "go"}, as_node="__input__")

    assert graph.get_state(fork).next == ("writer",)
    graph.invoke(None, fork)
    assert reads == [[1], [1]]


def test_a_replay_from_before_the_first_write_forks_with_only_the_writer_next() -> None:
    reads: list = []
    graph = _graph(reads)
    graph.invoke("go", CONFIG)
    first = next(
        s.config for s in graph.get_state_history(CONFIG) if s.metadata["step"] == -1
    )

    graph.invoke(None, first, durability="sync")

    fork = next(
        s for s in graph.get_state_history(CONFIG) if s.metadata["source"] == "fork"
    )
    assert fork.next == ("writer",)
    graph.invoke(None, fork.config)
    assert reads == [[1], [1], [1]]


async def test_an_ainput_update_from_before_the_first_write_starts_only_the_writer() -> (
    None
):
    reads: list = []
    graph = _graph(reads)
    await graph.ainvoke("go", CONFIG)
    first = [
        s.config
        async for s in graph.aget_state_history(CONFIG)
        if s.metadata["step"] == -1
    ][0]

    fork = await graph.aupdate_state(first, {"a": "go"}, as_node="__input__")

    assert (await graph.aget_state(fork)).next == ("writer",)
    await graph.ainvoke(None, fork)
    assert reads == [[1], [1]]
