from dataclasses import dataclass
from typing import TypedDict

from langgraph.graph import StateGraph
from langgraph.runtime import Runtime


@dataclass
class Ctx:
    v: str


class State(TypedDict, total=False):
    out: str


def _build(bound: Ctx):
    def node(state: State, runtime: Runtime[Ctx]) -> State:
        return {"out": runtime.context.v if runtime.context else None}

    return (
        StateGraph(State, context_schema=Ctx)
        .add_node("n", node)
        .set_entry_point("n")
        .set_finish_point("n")
        .compile(context=bound)
    )


def test_bound_context_used_when_invoke_omits_it():
    graph = _build(Ctx(v="bound"))
    assert graph.invoke({})["out"] == "bound"


def test_invoke_context_overrides_bound_context():
    graph = _build(Ctx(v="bound"))
    assert graph.invoke({}, context=Ctx(v="per-invoke"))["out"] == "per-invoke"
