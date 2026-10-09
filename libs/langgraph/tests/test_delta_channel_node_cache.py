"""A node served from the node cache saves its writes like a node that ran,
without writing them back to the cache."""

import operator
from typing import Annotated, Any

import pytest
from langgraph.cache.memory import InMemoryCache
from langgraph.checkpoint.memory import InMemorySaver
from typing_extensions import TypedDict

from langgraph.channels.delta import DeltaChannel
from langgraph.graph import START, StateGraph
from langgraph.types import CachePolicy, Durability

pytestmark = pytest.mark.anyio

INPUT = {"log": [], "plain": []}


def _append(current: list, writes: list) -> list:
    return [*current, *(item for write in writes for item in write)]


class _State(TypedDict):
    log: Annotated[list, DeltaChannel(_append)]
    plain: Annotated[list, operator.add]


class _CountsSets(InMemoryCache):
    sets = 0

    def set(self, keys: Any) -> None:
        self.sets += 1
        super().set(keys)


def _a_then_cached_b_then_c(runs: list[str], cache: InMemoryCache) -> Any:
    def node(name: str) -> Any:
        def run(state: _State) -> dict:
            runs.append(name)
            return {"log": [name], "plain": [name]}

        return run

    builder = StateGraph(_State)
    builder.add_node("a", node("a"))
    builder.add_node("b", node("b"), cache_policy=CachePolicy())
    builder.add_node("c", node("c"))
    builder.add_edge(START, "a")
    builder.add_edge("a", "b")
    builder.add_edge("b", "c")
    return builder.compile(checkpointer=InMemorySaver(), cache=cache)


def test_a_cache_hit_saves_its_writes_without_caching_them_again(
    durability: Durability,
) -> None:
    runs: list[str] = []
    cache = _CountsSets()
    graph = _a_then_cached_b_then_c(runs, cache)
    graph.invoke(INPUT, {"configurable": {"thread_id": "1"}}, durability=durability)
    config = {"configurable": {"thread_id": "2"}}

    graph.invoke(INPUT, config, durability=durability)

    assert runs == ["a", "b", "c", "a", "c"]
    assert cache.sets == 1, "the cache hit was written back to the cache"
    for state in graph.get_state_history(config):
        assert state.values.get("log", []) == state.values.get("plain", [])


async def test_a_cache_hit_saves_its_writes_without_caching_them_again_async(
    durability: Durability,
) -> None:
    runs: list[str] = []
    cache = _CountsSets()
    graph = _a_then_cached_b_then_c(runs, cache)
    await graph.ainvoke(
        INPUT, {"configurable": {"thread_id": "1"}}, durability=durability
    )
    config = {"configurable": {"thread_id": "2"}}

    await graph.ainvoke(INPUT, config, durability=durability)

    assert runs == ["a", "b", "c", "a", "c"]
    assert cache.sets == 1, "the cache hit was written back to the cache"
    async for state in graph.aget_state_history(config):
        assert state.values.get("log", []) == state.values.get("plain", [])
