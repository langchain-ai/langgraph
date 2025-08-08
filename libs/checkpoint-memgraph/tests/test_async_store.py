import asyncio
import itertools
import socket
import sys
import time
from collections.abc import AsyncGenerator, AsyncIterator, Coroutine, Iterator
from concurrent.futures import Future, ThreadPoolExecutor
from typing import Any
from urllib.parse import unquote, urlparse

import pytest
from neo4j import AsyncDriver, AsyncGraphDatabase

from langgraph.store.base import (
    GetOp,
    Item,
    ListNamespacesOp,
    PutOp,
    SearchOp,
    TTLConfig,
)
from langgraph.store.memgraph import MemgraphIndexConfig
from langgraph.store.memgraph.aio import AsyncMemgraphStore
from tests.conftest import (
    DEFAULT_MEMGRAPH_URI,
    VECTOR_TYPES,
    CharacterEmbeddings,
)

TTL_SECONDS = 6
TTL_MINUTES = TTL_SECONDS / 60


def is_memgraph_unavailable() -> bool:
    """
    Check if a Memgraph instance is unavailable.

    Returns:
        bool: True if a Memgraph instance is not available, False otherwise.
    """
    try:
        parsed_uri = urlparse(DEFAULT_MEMGRAPH_URI)
        if parsed_uri.port is None:
            return True
        with socket.create_connection(
            (parsed_uri.hostname, parsed_uri.port), timeout=1
        ):
            return False
    except (socket.timeout, ConnectionRefusedError):
        return True


pytestmark = pytest.mark.skipif(
    is_memgraph_unavailable(), reason="Memgraph instance not available"
)


async def wait_for_vector_index_drop(driver: AsyncDriver, timeout: int = 40) -> None:
    """Waits for the vector index to be dropped."""
    start_time = time.time()
    index_exists = True
    while time.time() - start_time < timeout:
        try:
            async with driver.session() as session:
                result = await session.run("SHOW VECTOR INDEX INFO")
                records = [r async for r in result]
                if not any(record["name"] == "vector_index" for record in records):
                    index_exists = False
                    break
        except Exception:
            index_exists = False
            break
        await asyncio.sleep(1.0)

    if index_exists:
        raise TimeoutError(
            "Vector index was not dropped within the timeout period. "
            "Memgraph may have crashed or is under heavy load."
        )


@pytest.fixture(scope="session")
def event_loop() -> Iterator[asyncio.AbstractEventLoop]:
    if sys.version_info < (3, 10):
        pytest.skip("Async Memgraph tests require Python 3.10+")
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture(scope="function")
async def async_driver() -> AsyncGenerator[Any, Any]:
    parsed = urlparse(DEFAULT_MEMGRAPH_URI)
    uri = f"{parsed.scheme}://{parsed.hostname}:{parsed.port or 7687}"
    auth = (unquote(parsed.username or ""), unquote(parsed.password or ""))
    async with AsyncGraphDatabase.driver(uri, auth=auth) as driver:
        await driver.verify_connectivity()
        yield driver


@pytest.fixture(scope="function")
async def store(async_driver: AsyncDriver) -> AsyncIterator[AsyncMemgraphStore]:
    ttl_config: TTLConfig = {
        "default_ttl": TTL_MINUTES,
        "refresh_on_read": True,
        "sweep_interval_minutes": TTL_MINUTES / 2,
    }
    store = AsyncMemgraphStore(async_driver, ttl=ttl_config)

    async with async_driver.session() as session:
        await session.run("MATCH (n) DETACH DELETE n")
        try:
            await session.run("DROP VECTOR INDEX vector_index")
            await wait_for_vector_index_drop(async_driver)
        except Exception:
            pass

    await store.setup()
    await store.start_ttl_sweeper()
    yield store
    await store.stop_ttl_sweeper()


async def test_no_running_loop(store: AsyncMemgraphStore) -> None:
    with pytest.raises(asyncio.InvalidStateError):
        store.put(("foo", "bar"), "baz", {"val": "baz"})
    with pytest.raises(asyncio.InvalidStateError):
        store.get(("foo", "bar"), "baz")
    with pytest.raises(asyncio.InvalidStateError):
        store.delete(("foo", "bar"), "baz")
    with pytest.raises(asyncio.InvalidStateError):
        store.search(("foo", "bar"))
    with pytest.raises(asyncio.InvalidStateError):
        store.list_namespaces(prefix=("foo",))
    with pytest.raises(asyncio.InvalidStateError):
        store.batch([PutOp(namespace=("foo", "bar"), key="baz", value={"val": "baz"})])
    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(store.put, ("foo", "bar"), "baz", {"val": "baz"})
        await asyncio.wrap_future(future)
        future = executor.submit(store.get, ("foo", "bar"), "baz")
        result = await asyncio.wrap_future(future)
        assert result is not None
        assert result.value == {"val": "baz"}
        result = await asyncio.wrap_future(
            executor.submit(store.list_namespaces, prefix=("foo",))
        )


async def test_large_batches(store: AsyncMemgraphStore) -> None:
    N = 20
    M = 4
    with ThreadPoolExecutor(max_workers=10) as executor:
        futures: list[Future] = []
        for m in range(M):
            for i in range(N):
                futures.extend(
                    [
                        executor.submit(
                            store.put,
                            ("test", "foo", "bar", "baz", str(m % 2)),
                            f"key{i}",
                            value={"foo": "bar" + str(i)},
                        ),
                        executor.submit(
                            store.get,
                            ("test", "foo", "bar", "baz", str(m % 2)),
                            f"key{i}",
                        ),
                        executor.submit(
                            store.list_namespaces, prefix=None, max_depth=m + 1
                        ),
                        executor.submit(store.search, ("test",)),
                        executor.submit(
                            store.put,
                            ("test", "foo", "bar", "baz", str(m % 2)),
                            f"key{i}",
                            value={"foo": "bar" + str(i)},
                        ),
                        executor.submit(
                            store.delete,
                            ("test", "foo", "bar", "baz", str(m % 2)),
                            f"key{i}",
                        ),
                    ]
                )
        results: list[Any] = await asyncio.gather(
            *(asyncio.wrap_future(future) for future in futures)
        )
    assert len(results) == M * N * 6


async def test_large_batches_async(store: AsyncMemgraphStore) -> None:
    N = 50
    M = 4
    CONCURRENT_REQUEST_LIMIT = 250
    coros: list[Coroutine] = []
    for m in range(M):
        for i in range(N):
            coros.extend(
                [
                    store.aput(
                        ("test", "foo", "bar", "baz", str(m % 2)),
                        f"key{i}",
                        value={"foo": "bar" + str(i)},
                    ),
                    store.aget(("test", "foo", "bar", "baz", str(m % 2)), f"key{i}"),
                    store.alist_namespaces(prefix=None, max_depth=m + 1),
                    store.asearch(("test",)),
                    store.aput(
                        ("test", "foo", "bar", "baz", str(m % 2)),
                        f"key{i}",
                        value={"foo": "bar" + str(i)},
                    ),
                    store.adelete(("test", "foo", "bar", "baz", str(m % 2)), f"key{i}"),
                ]
            )
    results = []
    for i in range(0, len(coros), CONCURRENT_REQUEST_LIMIT):
        batch = coros[i : i + CONCURRENT_REQUEST_LIMIT]
        results.extend(await asyncio.gather(*batch))
    assert len(results) == M * N * 6


async def test_abatch_order(store: AsyncMemgraphStore) -> None:
    await store.aput(("test", "foo"), "key1", {"data": "value1"})
    await store.aput(("test", "bar"), "key2", {"data": "value2"})
    ops: list[GetOp | PutOp | SearchOp | ListNamespacesOp] = [
        GetOp(namespace=("test", "foo"), key="key1"),
        PutOp(namespace=("test", "bar"), key="key2", value={"data": "value2"}),
        SearchOp(
            namespace_prefix=("test",), filter={"data": "value1"}, limit=10, offset=0
        ),
        ListNamespacesOp(match_conditions=None, max_depth=None, limit=10, offset=0),
        GetOp(namespace=("test",), key="key3"),
    ]
    results = await store.abatch(ops)
    assert len(results) == 5
    assert isinstance(results[0], Item) and results[0].value == {"data": "value1"}
    assert results[1] is None
    assert isinstance(results[2], list) and len(results[2]) == 1
    assert isinstance(results[3], list)
    assert ("test", "foo") in results[3] and ("test", "bar") in results[3]
    assert results[4] is None


async def test_batch_get_ops(store: AsyncMemgraphStore) -> None:
    await store.aput(("test",), "key1", {"data": "value1"})
    await store.aput(("test",), "key2", {"data": "value2"})
    ops: list[GetOp] = [
        GetOp(namespace=("test",), key="key1"),
        GetOp(namespace=("test",), key="key2"),
        GetOp(namespace=("test",), key="key3"),
    ]
    results = await store.abatch(ops)
    assert len(results) == 3
    assert isinstance(results[0], Item) and results[0].key == "key1"
    assert isinstance(results[1], Item) and results[1].key == "key2"
    assert results[2] is None


async def test_batch_put_ops(store: AsyncMemgraphStore) -> None:
    ops: list[PutOp] = [
        PutOp(namespace=("test",), key="key1", value={"data": "value1"}),
        PutOp(namespace=("test",), key="key2", value={"data": "value2"}),
        PutOp(namespace=("test",), key="key3", value=None),
    ]
    results = await store.abatch(ops)
    assert len(results) == 3 and all(result is None for result in results)
    items = await store.asearch(("test",), limit=10)
    assert len(items) == 2


async def test_batch_search_ops(store: AsyncMemgraphStore) -> None:
    await store.aput(("test", "foo"), "key1", {"data": "value1"})
    await store.aput(("test", "bar"), "key2", {"data": "value2"})
    ops: list[SearchOp] = [
        SearchOp(
            namespace_prefix=("test",), filter={"data": "value1"}, limit=10, offset=0
        ),
        SearchOp(namespace_prefix=("test",), filter=None, limit=5, offset=0),
    ]
    results = await store.abatch(ops)
    assert len(results) == 2
    assert isinstance(results[0], list) and len(results[0]) == 1
    assert isinstance(results[1], list) and len(results[1]) == 2


async def test_batch_list_namespaces_ops(store: AsyncMemgraphStore) -> None:
    await store.aput(("test", "namespace1"), "key1", {"data": "value1"})
    await store.aput(("test", "namespace2"), "key2", {"data": "value2"})
    ops: list[ListNamespacesOp] = [
        ListNamespacesOp(match_conditions=None, max_depth=None, limit=10, offset=0)
    ]
    results = await store.abatch(ops)
    assert len(results) == 1 and isinstance(results[0], list)
    assert ("test", "namespace1") in results[0]
    assert ("test", "namespace2") in results[0]


async def setup_vector_store(
    async_driver: AsyncDriver, index_config: MemgraphIndexConfig
) -> AsyncGenerator[AsyncMemgraphStore, None]:
    """Factory for setting up and tearing down a vector store."""
    async with async_driver.session() as session:
        await session.run("MATCH (n) DETACH DELETE n")
        try:
            await session.run("DROP VECTOR INDEX vector_index")
            await wait_for_vector_index_drop(async_driver)
        except Exception:
            pass
    await asyncio.sleep(2)  # Cooldown to prevent race condition

    store = AsyncMemgraphStore(async_driver, index=index_config)
    await store.setup()
    yield store


@pytest.fixture(
    scope="function",
    params=[
        (vector_type, distance_type)
        for vector_type in VECTOR_TYPES
        for distance_type in (
            ["hamming"] if vector_type == "bit" else ["l2", "inner_product", "cosine"]
        )
    ],
)
async def parameterized_vector_store(
    request: Any, async_driver: AsyncDriver, fake_embeddings: CharacterEmbeddings
) -> AsyncGenerator[AsyncMemgraphStore, None]:
    vector_type, distance_type = request.param
    metric_map = {"l2": "l2sq", "cosine": "cos", "inner_product": "ip"}
    metric = metric_map.get(distance_type, "l2sq")
    index_config: MemgraphIndexConfig = {
        "dimension": fake_embeddings.dims,
        "capacity": 1000,
        "embed": fake_embeddings,
        "metric": metric,
        "distance_type": distance_type,
    }
    async for store in setup_vector_store(async_driver, index_config):
        yield store


@pytest.fixture(scope="function")
async def vector_store(
    async_driver: AsyncDriver, fake_embeddings: CharacterEmbeddings
) -> AsyncGenerator[AsyncMemgraphStore, None]:
    index_config: MemgraphIndexConfig = {
        "dimension": fake_embeddings.dims,
        "capacity": 1000,
        "embed": fake_embeddings,
        "metric": "l2sq",
        "distance_type": "l2",
    }
    async for store in setup_vector_store(async_driver, index_config):
        yield store


@pytest.fixture(
    scope="function",
    params=[
        *itertools.product(["vector", "halfvec"], ["cosine", "inner_product", "l2"])
    ],
)
async def vector_store_with_path_embeddings(
    request: Any, async_driver: AsyncDriver, fake_embeddings: CharacterEmbeddings
) -> AsyncGenerator[AsyncMemgraphStore, None]:
    vector_type, distance_type = request.param
    metric_map = {"l2": "l2sq", "cosine": "cos", "inner_product": "ip"}
    metric = metric_map.get(distance_type, "l2sq")
    index_config: MemgraphIndexConfig = {
        "dimension": fake_embeddings.dims,
        "capacity": 1000,
        "embed": fake_embeddings,
        "metric": metric,
        "distance_type": distance_type,
        "fields": ["key0", "key1", "key3"],
    }
    async for store in setup_vector_store(async_driver, index_config):
        yield store


@pytest.fixture(
    scope="function",
    params=[*itertools.product(["vector"], ["l2", "cosine", "inner_product"])],
)
async def vector_store_for_sorting(
    request: Any, async_driver: AsyncDriver, fake_embeddings: CharacterEmbeddings
) -> AsyncGenerator[AsyncMemgraphStore, None]:
    vector_type, distance_type = request.param
    metric_map = {"l2": "l2sq", "cosine": "cos", "inner_product": "ip"}
    metric = metric_map.get(distance_type, "l2sq")
    index_config: MemgraphIndexConfig = {
        "dimension": fake_embeddings.dims,
        "capacity": 1000,
        "embed": fake_embeddings,
        "metric": metric,
        "distance_type": distance_type,
        "fields": ["key1"],
    }
    async for store in setup_vector_store(async_driver, index_config):
        yield store


async def test_vector_store_initialization(
    parameterized_vector_store: AsyncMemgraphStore,
    fake_embeddings: CharacterEmbeddings,
) -> None:
    assert parameterized_vector_store.index_config is not None
    assert parameterized_vector_store.index_config["dimension"] == fake_embeddings.dims


async def test_vector_insert_with_auto_embedding(
    vector_store: AsyncMemgraphStore,
) -> None:
    docs = [
        ("doc1", {"text": "short text"}),
        ("doc2", {"text": "longer text document"}),
        ("doc3", {"text": "longest text document here"}),
    ]
    for key, value in docs:
        await vector_store.aput(("test",), key, value)
    results = await vector_store.asearch(("test",), query="long text")
    assert len(results) > 0
    assert "doc2" in [r.key for r in results]


async def test_vector_update_with_embedding(vector_store: AsyncMemgraphStore) -> None:
    await vector_store.aput(("test",), "doc1", {"text": "zany zebra Xerxes"})
    results = await vector_store.asearch(("test",), query="Zany Xerxes")
    assert results and results[0].key == "doc1"
    initial_score = results[0].score
    assert initial_score is not None
    await vector_store.aput(("test",), "doc1", {"text": "new text about dogs"})
    results_after = await vector_store.asearch(("test",), query="Zany Xerxes")
    after_score = next((r.score for r in results_after if r.key == "doc1"), 0.0)
    assert (after_score or 0.0) < (initial_score or 0.0)


async def test_vector_search_with_filters(vector_store: AsyncMemgraphStore) -> None:
    docs = [
        ("doc1", {"text": "red apple", "color": "red"}),
        ("doc2", {"text": "red car", "color": "red"}),
        ("doc3", {"text": "green apple", "color": "green"}),
    ]
    for key, value in docs:
        await vector_store.aput(("test",), key, value)
    results = await vector_store.asearch(
        ("test",), query="apple", filter={"color": "red"}
    )
    assert results and results[0].key == "doc1"


async def test_vector_search_pagination(vector_store: AsyncMemgraphStore) -> None:
    for i in range(5):
        await vector_store.aput(("test",), f"doc{i}", {"text": f"doc num {i}"})
    page1 = await vector_store.asearch(("test",), query="doc", limit=2)
    page2 = await vector_store.asearch(("test",), query="doc", limit=2, offset=2)
    assert len(page1) == 2 and len(page2) == 2
    assert page1[0].key != page2[0].key


async def test_vector_search_edge_cases(vector_store: AsyncMemgraphStore) -> None:
    await vector_store.aput(("test",), "doc1", {"text": "test document"})
    results = await vector_store.asearch(("test",), query="")
    assert len(results) == 1 and results[0].score is None


async def test_embed_with_path(
    vector_store_with_path_embeddings: AsyncMemgraphStore,
) -> None:
    store = vector_store_with_path_embeddings
    await store.aput(("test",), "doc1", {"key1": "xxx", "key3": "zzz"})
    await store.aput(("test",), "doc2", {"key0": "uuu", "key3": "xxx"})
    results = await store.asearch(("test",), query="xxx")
    assert len(results) == 2
    assert results[0].score == pytest.approx(results[1].score, abs=1e-3)
    results_uuu = await store.asearch(("test",), query="uuu")
    assert len(results_uuu) == 2 and results_uuu[0].key == "doc2"
    assert results_uuu[0].score is not None and results_uuu[1].score is not None
    assert results_uuu[0].score > results_uuu[1].score


async def test_search_sorting(
    vector_store_for_sorting: AsyncMemgraphStore,
) -> None:
    store = vector_store_for_sorting
    await store.aput(("test", "M"), "M", {"key1": "mmm"})
    for i in range(15):
        await store.aput(("test", "A"), f"A{i}", {"key1": "no"})
    results = await store.asearch(("test",), query="mmm", limit=10)
    assert len(results) == 10 and results[0].key == "M"
    assert results[0].score is not None and results[1].score is not None
    assert results[0].score > results[1].score


async def test_store_ttl(store: AsyncMemgraphStore) -> None:
    ns = ("foo",)
    await store.start_ttl_sweeper()
    await store.aput(ns, "item1", {"foo": "bar"}, ttl=TTL_MINUTES)
    await asyncio.sleep(TTL_SECONDS - 2)
    assert await store.aget(ns, "item1", refresh_ttl=True) is not None
    await asyncio.sleep(TTL_SECONDS - 2)
    assert len(await store.asearch(ns, query="foo", refresh_ttl=True)) == 1
    await asyncio.sleep(TTL_SECONDS - 2)
    assert await store.aget(ns, "item1", refresh_ttl=False) is not None
    await asyncio.sleep(TTL_SECONDS)
    assert len(await store.asearch(ns, query="bar", refresh_ttl=False)) == 0
