from typing import TypedDict

import pytest

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph


class State(TypedDict, total=False):
    out: str
    forbidden: int


def _build(validators):
    def node(state: State) -> State:
        return {"out": "done"}

    return (
        StateGraph(State)
        .add_node("n", node)
        .add_edge(START, "n")
        .add_edge("n", END)
        .compile(checkpointer=InMemorySaver(), input_validators=validators)
    )


def test_validator_can_rewrite_input():
    def upper(values: dict) -> dict:
        values["out"] = (values.get("out") or "").upper()
        return values

    graph = _build([upper])
    config = {"configurable": {"thread_id": "t1"}}
    graph.update_state(config, {"out": "hi"})
    assert graph.get_state(config).values["out"] == "HI"


def test_validator_can_reject_input():
    def reject(values: dict) -> dict:
        if "forbidden" in values:
            raise ValueError("forbidden key")
        return values

    graph = _build([reject])
    config = {"configurable": {"thread_id": "t2"}}
    with pytest.raises(ValueError, match="forbidden key"):
        graph.update_state(config, {"forbidden": 1})


def test_without_validators_input_is_untouched():
    graph = _build(None)
    config = {"configurable": {"thread_id": "t3"}}
    graph.update_state(config, {"out": "plain"})
    assert graph.get_state(config).values["out"] == "plain"
