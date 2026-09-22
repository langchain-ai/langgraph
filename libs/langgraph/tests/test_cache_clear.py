import pytest
from langgraph.cache.memory import InMemoryCache
from langgraph.graph import END, START, StateGraph
from langgraph.types import CachePolicy


def _build_graph(calls: list[int]):
    def cached_node(state: dict) -> dict:
        calls.append(state["value"])
        return {"value": state["value"] + 1}

    builder = StateGraph(dict)
    builder.add_node("cached_node", cached_node, cache_policy=CachePolicy())
    builder.add_edge(START, "cached_node")
    builder.add_edge("cached_node", END)
    return builder.compile(cache=InMemoryCache())


def test_clear_cache_with_empty_nodes_is_noop():
    calls: list[int] = []
    graph = _build_graph(calls)

    graph.invoke({"value": 1})
    graph.invoke({"value": 1})
    assert calls == [1]

    graph.clear_cache([])
    graph.invoke({"value": 1})
    assert calls == [1]


@pytest.mark.anyio
async def test_aclear_cache_with_empty_nodes_is_noop():
    calls: list[int] = []
    graph = _build_graph(calls)

    await graph.ainvoke({"value": 1})
    await graph.ainvoke({"value": 1})
    assert calls == [1]

    await graph.aclear_cache([])
    await graph.ainvoke({"value": 1})
    assert calls == [1]
