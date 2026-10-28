"""`Pregel` is the type of compiled graphs, built only by `StateGraph` and
`@entrypoint`."""

import pytest
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.memory import InMemorySaver
from typing_extensions import TypedDict

import langgraph.pregel
from langgraph.channels.ephemeral_value import EphemeralValue
from langgraph.func import entrypoint
from langgraph.graph import START, StateGraph
from langgraph.pregel import Pregel
from langgraph.pregel._read import PregelNode


def test_pregel_cannot_be_constructed_directly() -> None:
    with pytest.raises(TypeError, match="`Pregel` cannot be constructed directly"):
        Pregel(
            nodes={"node": PregelNode(channels="a", triggers=["a"])},
            channels={"a": EphemeralValue(str)},
            input_channels="a",
            output_channels="a",
        )


def test_node_builder_is_not_exported() -> None:
    assert langgraph.pregel.__all__ == ("Pregel",)
    assert not hasattr(langgraph.pregel, "NodeBuilder")


def test_compiled_graphs_are_pregel_instances() -> None:
    class State(TypedDict):
        value: int

    def increment(state: State) -> State:
        return {"value": state["value"] + 1}

    builder = StateGraph(State)
    builder.add_node("increment", increment)
    builder.add_edge(START, "increment")
    graph = builder.compile(checkpointer=InMemorySaver())

    @entrypoint(checkpointer=InMemorySaver())
    def workflow(value: int) -> int:
        return value + 1

    assert isinstance(graph, Pregel)
    assert isinstance(workflow, Pregel)

    # `with_config` copies the graph by calling its constructor again.
    config: RunnableConfig = {"configurable": {"thread_id": "1"}}
    assert graph.with_config(recursion_limit=5).invoke({"value": 1}, config) == {
        "value": 2
    }
    assert workflow.with_config(recursion_limit=5).invoke(1, config) == 2
