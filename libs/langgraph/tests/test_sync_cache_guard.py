from dataclasses import dataclass

import pytest
from langgraph.cache.memory import InMemoryCache
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.func import entrypoint, task
from langgraph.types import CachePolicy, interrupt

pytestmark = pytest.mark.anyio


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


@dataclass
class CacheProbe:
    sync_error_calls: int = 0
    sync_interrupt_calls: int = 0
    sync_success_calls: int = 0
    async_error_calls: int = 0
    async_interrupt_calls: int = 0
    async_success_calls: int = 0


def build_sync_graphs(probe: CacheProbe):
    cache = InMemoryCache()
    checkpointer = InMemorySaver()

    @task(cache_policy=CachePolicy())
    def sync_error_task(x: int) -> int:
        probe.sync_error_calls += 1
        raise ValueError(f"boom-{probe.sync_error_calls}")

    @task(cache_policy=CachePolicy())
    def sync_interrupt_task() -> str:
        probe.sync_interrupt_calls += 1
        return interrupt("need sync input")

    @task(cache_policy=CachePolicy())
    def sync_success_task(x: int) -> int:
        probe.sync_success_calls += 1
        return x * 2

    @entrypoint(cache=cache)
    def sync_error_graph(inp: dict) -> int:
        return sync_error_task(inp["x"]).result()

    @entrypoint(cache=cache, checkpointer=checkpointer)
    def sync_interrupt_graph(_: dict) -> dict:
        return {"value": sync_interrupt_task().result()}

    @entrypoint(cache=cache)
    def sync_success_graph(inp: dict) -> int:
        return sync_success_task(inp["x"]).result()

    return cache, sync_error_graph, sync_interrupt_graph, sync_success_graph


def build_async_graphs(probe: CacheProbe):
    cache = InMemoryCache()
    checkpointer = InMemorySaver()

    @task(cache_policy=CachePolicy())
    async def async_error_task(x: int) -> int:
        probe.async_error_calls += 1
        raise ValueError(f"aboom-{probe.async_error_calls}")

    @task(cache_policy=CachePolicy())
    async def async_interrupt_task() -> str:
        probe.async_interrupt_calls += 1
        return interrupt("need async input")

    @task(cache_policy=CachePolicy())
    async def async_success_task(x: int) -> int:
        probe.async_success_calls += 1
        return x * 3

    @entrypoint(cache=cache)
    async def async_error_graph(inp: dict) -> int:
        return await async_error_task(inp["x"])

    @entrypoint(cache=cache, checkpointer=checkpointer)
    async def async_interrupt_graph(_: dict) -> dict:
        return {"value": await async_interrupt_task()}

    @entrypoint(cache=cache)
    async def async_success_graph(inp: dict) -> int:
        return await async_success_task(inp["x"])

    return cache, async_error_graph, async_interrupt_graph, async_success_graph


def assert_interrupt_payload(result: dict, expected_value: str) -> None:
    assert "__interrupt__" in result
    interrupts = result["__interrupt__"]
    assert len(interrupts) == 1
    assert interrupts[0].value == expected_value


def total_cache_entries(cache: InMemoryCache) -> int:
    return sum(len(entries) for entries in cache._cache.values())


def assert_cache_is_empty(cache: InMemoryCache) -> None:
    assert total_cache_entries(cache) == 0


def assert_cache_has_single_entry(cache: InMemoryCache) -> None:
    assert total_cache_entries(cache) == 1


def test_sync_error_writes_are_not_cached() -> None:
    probe = CacheProbe()
    cache, sync_error_graph, _, _ = build_sync_graphs(probe)

    with pytest.raises(ValueError, match="boom-1"):
        sync_error_graph.invoke({"x": 7})
    with pytest.raises(ValueError, match="boom-2"):
        sync_error_graph.invoke({"x": 7})

    assert probe.sync_error_calls == 2
    assert_cache_is_empty(cache)


def test_sync_interrupt_writes_are_not_cached() -> None:
    probe = CacheProbe()
    cache, _, sync_interrupt_graph, _ = build_sync_graphs(probe)
    config = {"configurable": {"thread_id": "sync-cache-guard"}}

    first = sync_interrupt_graph.invoke({}, config)
    second = sync_interrupt_graph.invoke({}, config)

    assert_interrupt_payload(first, "need sync input")
    assert_interrupt_payload(second, "need sync input")
    assert probe.sync_interrupt_calls == 2
    assert_cache_is_empty(cache)


def test_sync_successful_writes_still_use_cache() -> None:
    probe = CacheProbe()
    cache, _, _, sync_success_graph = build_sync_graphs(probe)

    assert sync_success_graph.invoke({"x": 5}) == 10
    assert sync_success_graph.invoke({"x": 5}) == 10

    assert probe.sync_success_calls == 1
    assert_cache_has_single_entry(cache)


async def test_async_error_writes_are_not_cached() -> None:
    probe = CacheProbe()
    cache, async_error_graph, _, _ = build_async_graphs(probe)

    with pytest.raises(ValueError, match="aboom-1"):
        await async_error_graph.ainvoke({"x": 7})
    with pytest.raises(ValueError, match="aboom-2"):
        await async_error_graph.ainvoke({"x": 7})

    assert probe.async_error_calls == 2
    assert_cache_is_empty(cache)


async def test_async_interrupt_writes_are_not_cached() -> None:
    probe = CacheProbe()
    cache, _, async_interrupt_graph, _ = build_async_graphs(probe)
    config = {"configurable": {"thread_id": "async-cache-guard"}}

    first = await async_interrupt_graph.ainvoke({}, config)
    second = await async_interrupt_graph.ainvoke({}, config)

    assert_interrupt_payload(first, "need async input")
    assert_interrupt_payload(second, "need async input")
    assert probe.async_interrupt_calls == 2
    assert_cache_is_empty(cache)


async def test_async_successful_writes_still_use_cache() -> None:
    probe = CacheProbe()
    cache, _, _, async_success_graph = build_async_graphs(probe)

    assert await async_success_graph.ainvoke({"x": 5}) == 15
    assert await async_success_graph.ainvoke({"x": 5}) == 15

    assert probe.async_success_calls == 1
    assert_cache_has_single_entry(cache)



def test_sync_success_cache_keys_remain_input_specific() -> None:
    probe = CacheProbe()
    cache, _, _, sync_success_graph = build_sync_graphs(probe)

    assert sync_success_graph.invoke({"x": 5}) == 10
    assert sync_success_graph.invoke({"x": 6}) == 12
    assert sync_success_graph.invoke({"x": 5}) == 10

    assert probe.sync_success_calls == 2
    assert total_cache_entries(cache) == 2


async def test_async_success_cache_keys_remain_input_specific() -> None:
    probe = CacheProbe()
    cache, _, _, async_success_graph = build_async_graphs(probe)

    assert await async_success_graph.ainvoke({"x": 5}) == 15
    assert await async_success_graph.ainvoke({"x": 6}) == 18
    assert await async_success_graph.ainvoke({"x": 5}) == 15

    assert probe.async_success_calls == 2
    assert total_cache_entries(cache) == 2
