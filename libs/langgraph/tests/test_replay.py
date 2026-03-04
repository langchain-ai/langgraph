"""MRE: replay from before an interrupt node uses cached resume values."""

import operator
from typing import Annotated

from langgraph.checkpoint.memory import MemorySaver
from typing_extensions import TypedDict

from langgraph.graph import START, StateGraph
from langgraph.types import Command, interrupt


class State(TypedDict):
    value: Annotated[list[str], operator.add]


def test_replay_uses_cached_resume():
    called: list[str] = []

    def node_a(state: State) -> State:
        called.append("node_a")
        return {"value": ["a"]}

    def ask_human(state: State) -> State:
        called.append("ask_human")
        answer = interrupt("What is your input?")
        return {"value": [f"human:{answer}"]}

    def node_b(state: State) -> State:
        called.append("node_b")
        return {"value": ["b"]}

    graph = (
        StateGraph(State)
        .add_node("node_a", node_a)
        .add_node("ask_human", ask_human)
        .add_node("node_b", node_b)
        .add_edge(START, "node_a")
        .add_edge("node_a", "ask_human")
        .add_edge("ask_human", "node_b")
        .compile(checkpointer=MemorySaver())
    )

    config = {"configurable": {"thread_id": "1"}}

    # Run until interrupt
    result = graph.invoke({"value": []}, config)
    assert "__interrupt__" in result

    # Resume with answer
    result = graph.invoke(Command(resume="hello"), config)
    assert result == {"value": ["a", "human:hello", "b"]}

    # Find checkpoint before ask_human
    history = list(graph.get_state_history(config))
    before_ask = [s for s in history if s.next == ("ask_human",)][-1]

    # Replay from that checkpoint
    called.clear()
    replay_result = graph.invoke(None, before_ask.config)

    # Interrupt is NOT re-triggered — cached resume value used
    assert replay_result == {"value": ["a", "human:hello", "b"]}
    assert "__interrupt__" not in replay_result
    assert "ask_human" in called
    assert "node_b" in called
    assert "node_a" not in called


if __name__ == "__main__":
    test_replay_uses_cached_resume()
    print("PASSED")
