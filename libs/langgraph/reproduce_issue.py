"""Repro for #6770: missing END mapping in conditional `path_map`."""

from __future__ import annotations

from typing_extensions import TypedDict

from langgraph.constants import END
from langgraph.graph import StateGraph


class State(TypedDict, total=False):
    counter: int


def router(state: State) -> str:
    if state.get("counter", 0) >= 2:
        return END
    return "worker"


def worker(state: State) -> State:
    return {"counter": state.get("counter", 0) + 1}


def main() -> None:
    graph = StateGraph(State)
    graph.add_node("router", lambda state: {})
    graph.add_node("worker", worker)
    graph.set_entry_point("router")
    # Intentionally omit END from path_map to trigger the bug on older versions.
    graph.add_conditional_edges("router", router, {"worker": "worker"})
    graph.add_edge("worker", "router")

    app = graph.compile()
    result = app.invoke({})
    assert result == {"counter": 2}, result
    print("Repro passed:", result)


if __name__ == "__main__":
    main()
