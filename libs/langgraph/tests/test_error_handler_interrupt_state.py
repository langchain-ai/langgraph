from typing import TypedDict

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.errors import NodeError
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, interrupt


class State(TypedDict, total=False):
    x: int


def test_get_state_reports_interrupt_from_error_handler() -> None:
    def boom(state: State) -> State:
        raise ValueError("boom")

    def ask(state: State, error: NodeError) -> State:
        return {"x": interrupt("x")}

    builder = StateGraph(State)
    builder.add_node("boom", boom, error_handler=ask)
    builder.add_edge(START, "boom")
    builder.add_edge("boom", END)
    graph = builder.compile(checkpointer=InMemorySaver())
    config = {"configurable": {"thread_id": "1"}}

    result = graph.invoke({"x": 0}, config)
    snapshot = graph.get_state(config)

    assert snapshot.interrupts
    assert [item.value for item in snapshot.interrupts] == ["x"]
    resumed = graph.invoke(
        Command(resume={result["__interrupt__"][0].id: 7}), config
    )
    assert resumed["x"] == 7
