"""Test that conditional edges handle END without explicit path_map entry.

Regression test for https://github.com/langchain-ai/langgraph/issues/6770
"""

from typing_extensions import TypedDict

from langgraph.constants import END
from langgraph.graph.state import StateGraph


class CounterState(TypedDict, total=False):
    counter: int


def test_conditional_edge_end_without_path_map_entry() -> None:
    """Router returning END should work even when path_map omits __end__."""

    def router(state: CounterState) -> str:
        if state.get("counter", 0) >= 5:
            return END
        return "worker"

    def worker(state: CounterState) -> CounterState:
        return {"counter": state.get("counter", 0) + 1}

    g = StateGraph(CounterState)
    g.add_node("entry", lambda s: s)
    g.add_node("router_node", lambda s: s)
    g.add_node("worker", worker)

    g.set_entry_point("entry")
    g.add_edge("entry", "router_node")

    # path_map intentionally lacks an END mapping â€” this must not crash
    g.add_conditional_edges(
        "router_node",
        router,
        {"worker": "worker"},
    )
    g.add_edge("worker", "router_node")

    app = g.compile()

    # counter starts at 0 â†’ loops 5 times then hits END
    result = app.invoke({"counter": 0})
    assert result == {"counter": 5}


def test_conditional_edge_end_with_path_map_entry() -> None:
    """Router returning END still works when path_map explicitly maps it."""

    def router(state: CounterState) -> str:
        if state.get("counter", 0) >= 3:
            return END
        return "worker"

    def worker(state: CounterState) -> CounterState:
        return {"counter": state.get("counter", 0) + 1}

    g = StateGraph(CounterState)
    g.add_node("entry", lambda s: s)
    g.add_node("router_node", lambda s: s)
    g.add_node("worker", worker)

    g.set_entry_point("entry")
    g.add_edge("entry", "router_node")

    # path_map explicitly includes END â€” should still work
    g.add_conditional_edges(
        "router_node",
        router,
        {"worker": "worker", END: END},
    )
    g.add_edge("worker", "router_node")

    app = g.compile()

    result = app.invoke({"counter": 0})
    assert result == {"counter": 3}


def test_conditional_edge_end_immediate() -> None:
    """Router returning END on the very first invocation terminates immediately."""

    def always_end(state: CounterState) -> str:
        return END

    g = StateGraph(CounterState)
    g.add_node("entry", lambda s: s)
    g.add_node("router_node", lambda s: s)
    g.add_node("worker", lambda s: s)

    g.set_entry_point("entry")
    g.add_edge("entry", "router_node")

    g.add_conditional_edges(
        "router_node",
        always_end,
        {"worker": "worker"},
    )

    app = g.compile()

    result = app.invoke({"counter": 42})
    assert result == {"counter": 42}
