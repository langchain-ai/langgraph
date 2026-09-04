import pytest
from typing_extensions import TypedDict

from langgraph.errors import NodeError
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command


class State(TypedDict, total=False):
    x: str
    y: str


def _build_graph(parallel: bool):
    handler_calls = {"count": 0}

    def boom(state: State) -> State:
        raise RuntimeError("test")

    def sibling(state: State) -> State:
        return {"y": "sibling"}

    def handler(state: State, error: NodeError) -> Command:
        handler_calls["count"] += 1
        return Command(update={"x": "handled"}, goto=END)

    builder = (
        StateGraph(State)
        .add_node("boom", boom, error_handler=handler)
        .add_edge(START, "boom")
    )
    if parallel:
        builder = builder.add_node("sibling", sibling).add_edge(START, "sibling")
    return builder.compile(), handler_calls


# (parallel, mode) pairs. For "custom" streams the graph emits no custom
# events, so the pass condition is simply that no exception escapes the run
# and the handler executed once. For invoke/values modes we additionally
# assert the handler's update landed in the final state.
STREAM_CASES = [
    pytest.param(False, "custom", id="solo-stream-custom"),
    pytest.param(True, "custom", id="parallel-stream-custom"),
    pytest.param(True, "values", id="parallel-stream-values"),
]

INVOKE_CASES = [
    pytest.param(False, id="solo-invoke"),
    pytest.param(True, id="parallel-invoke"),
]


@pytest.mark.parametrize("parallel", INVOKE_CASES)
def test_error_handler_suppresses_handled_error_invoke(parallel: bool) -> None:
    graph, calls = _build_graph(parallel)
    result = graph.invoke({})
    assert calls["count"] == 1
    assert result["x"] == "handled"


@pytest.mark.parametrize("parallel,mode", STREAM_CASES)
def test_error_handler_suppresses_handled_error_stream(
    parallel: bool, mode: str
) -> None:
    graph, calls = _build_graph(parallel)
    # The pass condition for stream modes is that the run completes without
    # the handled error escaping; the handler must have executed exactly once.
    list(graph.stream({}, stream_mode=mode))
    assert calls["count"] == 1
