"""Regression for #8941: PEP 563 stringifies `RunnableConfig | None`.

This file must keep `from __future__ import annotations` so annotations
are strings, matching the production modules that hit the bug.
"""

from __future__ import annotations

import warnings
from typing_extensions import TypedDict

from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, START, StateGraph


class State(TypedDict):
    seen: list


def _node_union_none(state: State, config: RunnableConfig | None = None) -> dict:
    return {"seen": [config]}


def test_pep563_runnable_config_union_none_injects_and_does_not_warn() -> None:
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        builder = StateGraph(State)
        builder.add_node("node", _node_union_none)
        builder.add_edge(START, "node")
        builder.add_edge("node", END)
        graph = builder.compile()
        config_warnings = [
            w
            for w in caught
            if issubclass(w.category, UserWarning)
            and "config" in str(w.message).lower()
        ]
        assert config_warnings == []

    result = graph.invoke({"seen": []}, config={"tags": ["probe"]})
    injected = result["seen"][0]
    assert injected is not None
    assert "probe" in injected.get("tags", [])
