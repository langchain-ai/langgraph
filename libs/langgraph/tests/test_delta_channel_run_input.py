"""A run's input to a DeltaChannel input channel reads back on the checkpoints
built from it, and on no others."""

import pytest
from langgraph.checkpoint.memory import InMemorySaver

from langgraph.channels.binop import BinaryOperatorAggregate
from langgraph.channels.delta import DeltaChannel
from langgraph.channels.last_value import LastValue
from langgraph.pregel import NodeBuilder, Pregel
from langgraph.types import Durability

pytestmark = pytest.mark.anyio


def _sorted_extend(current: list, writes: list) -> list:
    return sorted([*current, *(item for write in writes for item in write)])


def _delta_input_graph() -> Pregel:
    node = NodeBuilder().subscribe_only("go").do(lambda _: [2]).write_to("log", "plain")
    return Pregel(
        nodes={"n": node},
        channels={
            "log": DeltaChannel(_sorted_extend),
            "plain": BinaryOperatorAggregate(list, lambda a, b: sorted(a + b)),
            "go": LastValue(int),
        },
        input_channels=["log", "plain", "go"],
        output_channels=["log", "plain"],
        checkpointer=InMemorySaver(),
    )


def test_each_run_input_reads_back_on_its_own_checkpoints(
    durability: Durability,
) -> None:
    graph = _delta_input_graph()
    config = {"configurable": {"thread_id": "t"}}

    graph.invoke({"log": [0], "plain": [0], "go": 1}, config, durability=durability)
    graph.invoke({"log": [5], "plain": [5], "go": 1}, config, durability=durability)

    for state in graph.get_state_history(config):
        assert state.values.get("log", []) == state.values.get("plain", [])


async def test_each_run_input_reads_back_on_its_own_checkpoints_async(
    durability: Durability,
) -> None:
    graph = _delta_input_graph()
    config = {"configurable": {"thread_id": "t"}}

    await graph.ainvoke(
        {"log": [0], "plain": [0], "go": 1}, config, durability=durability
    )
    await graph.ainvoke(
        {"log": [5], "plain": [5], "go": 1}, config, durability=durability
    )

    async for state in graph.aget_state_history(config):
        assert state.values.get("log", []) == state.values.get("plain", [])


OTHER_BRANCH_INPUTS = pytest.mark.parametrize(
    "other_branch_input",
    [{"go": 1}, {"log": [5], "plain": [5], "go": 1}],
    ids=["other-branch-without-delta-input", "other-branch-with-delta-input"],
)


@OTHER_BRANCH_INPUTS
def test_run_input_from_an_older_checkpoint_stays_out_of_its_other_branch(
    durability: Durability, other_branch_input: dict
) -> None:
    graph = _delta_input_graph()
    config = {"configurable": {"thread_id": "t"}}
    graph.invoke({"go": 1}, config, durability=durability)
    older = graph.get_state(config).config
    graph.invoke(other_branch_input, config, durability=durability)

    graph.invoke({"log": [7], "plain": [7], "go": 1}, older, durability=durability)

    for state in graph.get_state_history(config):
        assert state.values.get("log", []) == state.values.get("plain", [])


@OTHER_BRANCH_INPUTS
async def test_run_input_from_an_older_checkpoint_stays_out_of_its_other_branch_async(
    durability: Durability, other_branch_input: dict
) -> None:
    graph = _delta_input_graph()
    config = {"configurable": {"thread_id": "t"}}
    await graph.ainvoke({"go": 1}, config, durability=durability)
    older = (await graph.aget_state(config)).config
    await graph.ainvoke(other_branch_input, config, durability=durability)

    await graph.ainvoke(
        {"log": [7], "plain": [7], "go": 1}, older, durability=durability
    )

    async for state in graph.aget_state_history(config):
        assert state.values.get("log", []) == state.values.get("plain", [])
