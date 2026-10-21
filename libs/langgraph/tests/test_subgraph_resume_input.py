from typing import TypedDict

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph


class State(TypedDict, total=False):
    x: int
    rounds: int


class Child(TypedDict):
    x: int


def test_checkpointed_subgraph_applies_new_input_after_parent_breakpoint() -> None:
    def increment(state: Child) -> Child:
        return {"x": state["x"] + 1}

    child = StateGraph(Child)
    child.add_node("increment", increment)
    child.add_edge(START, "increment")
    child.add_edge("increment", END)

    builder = StateGraph(State)
    builder.add_node("child", child.compile(checkpointer=True))
    builder.add_node("tick", lambda state: {"rounds": state["rounds"] + 1})
    builder.add_edge(START, "child")
    builder.add_edge("child", "tick")
    builder.add_conditional_edges(
        "tick",
        lambda state: "child" if state["rounds"] < 2 else END,
        ["child", END],
    )
    graph = builder.compile(checkpointer=InMemorySaver())
    config = {"configurable": {"thread_id": "1"}}
    result = graph.invoke(
        {"x": 0, "rounds": 0}, config, interrupt_before=["child"]
    )
    for _ in range(5):
        if not graph.get_state(config).next:
            break
        result = graph.invoke(None, config, interrupt_before=["child"])
    assert result["x"] == 2
