# type: ignore
from __future__ import annotations

import asyncio
import itertools
import uuid
from collections.abc import AsyncIterator
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from typing import Any

import pytest
from langchain_core.embeddings import Embeddings
from langgraph.store.base import (
    GetOp,
    Item,
    ListNamespacesOp,
    PutOp,
    SearchOp,
)
from psycopg import AsyncConnection

from langgraph.store.postgres import AsyncPostgresStore
from tests.conftest import (
    DEFAULT_URI,
    VECTOR_TYPES,
    CharacterEmbeddings,
)

TTL_SECONDS = 6
TTL_MINUTES = TTL_SECONDS / 60


@pytest.fixture(scope="function", params=["default", "pipe", "pool"])
async def store(request) -> AsyncIterator[AsyncPostgresStore]:
    database = f"test_{uuid.uuid4().hex[:16]}"
    uri_parts = DEFAULT_URI.split("/")
    uri_base = "/".join(uri_parts[:-1])
    query_params = ""
    if "?" in uri_parts[-1]:
        db_name, query_params = uri_parts[-1].split("?", 1)
        query_params = "?" + query_params

    conn_string = f"{uri_base}/{database}{query_params}"
    admin_conn_string = DEFAULT_URI
    ttl_config = {
        "default_ttl": TTL_MINUTES,
        "refresh_on_read": True,
        "sweep_interval_minutes": TTL_MINUTES / 2,
    }
    async with await AsyncConnection.connect(
        admin_conn_string, autocommit=True
    ) as conn:
        await conn.execute(f"CREATE DATABASE {database}")
    try:
        async with AsyncPostgresStore.from_conn_string(
            conn_string, ttl=ttl_config
        ) as store:
            store.MIGRATIONS = [
                (
                    mig.replace("ttl_minutes INT;", "ttl_minutes FLOAT;")
                    if isinstance(mig, str)
                    else mig
                )
                for mig in store.MIGRATIONS
            ]
            await store.setup()
            async with store._cursor() as cur:
                # drop the migration index
                await cur.execute("DROP TABLE IF EXISTS store_migrations")
            await store.setup()  # Will fail if migrations aren't idempotent

        if request.param == "pipe":
            async with AsyncPostgresStore.from_conn_string(
                conn_string, pipeline=True, ttl=ttl_config
            ) as store:
                await store.start_ttl_sweeper()
                yield store
                await store.stop_ttl_sweeper()
        elif request.param == "pool":
            async with AsyncPostgresStore.from_conn_string(
                conn_string, pool_config={"min_size": 1, "max_size": 10}, ttl=ttl_config
            ) as store:
                await store.start_ttl_sweeper()
                yield store
                await store.stop_ttl_sweeper()
        else:  # default
            async with AsyncPostgresStore.from_conn_string(
                conn_string, ttl=ttl_config
            ) as store:
                await store.start_ttl_sweeper()
                yield store
                await store.stop_ttl_sweeper()
    finally:
        async with await AsyncConnection.connect(
            admin_conn_string, autocommit=True
        ) as conn:
            await conn.execute(f"DROP DATABASE {database}")


async def test_no_running_loop(store: AsyncPostgresStore) -> None:
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
        result = await asyncio.wrap_future(future)
        assert result is None
        future = executor.submit(store.get, ("foo", "bar"), "baz")
        result = await asyncio.wrap_future(future)
        assert result.value == {"val": "baz"}
        result = await asyncio.wrap_future(
            executor.submit(store.list_namespaces, prefix=("foo",))
        )


async def test_large_batches(request: Any, store: AsyncPostgresStore) -> None:
    N = 100  # less important that we are performant here
    M = 10

    with ThreadPoolExecutor(max_workers=10) as executor:
        futures = []
        for m in range(M):
            for i in range(N):
                futures += [
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
                        store.list_namespaces,
                        prefix=None,
                        max_depth=m + 1,
                    ),
                    executor.submit(
                        store.search,
                        ("test",),
                    ),
                    executor.submit(
                        store.put,
                        ("test", "foo", "bar", "baz", str(m % 2)),
                        f"key{i}",
                        value={"foo": "bar" + str(i)},
                    ),
                    executor.submit(
                        store.put,
                        ("test", "foo", "bar", "baz", str(m % 2)),
                        f"key{i}",
                        None,
                    ),
                ]

        results = await asyncio.gather(
            *(asyncio.wrap_future(future) for future in futures)
        )
    assert len(results) == M * N * 6


async def test_large_batches_async(store: AsyncPostgresStore) -> None:
    N = 1000
    M = 10
    coros = []
    for m in range(M):
        for i in range(N):
            coros.append(
                store.aput(
                    ("test", "foo", "bar", "baz", str(m % 2)),
                    f"key{i}",
                    value={"foo": "bar" + str(i)},
                )
            )
            coros.append(
                store.aget(
                    ("test", "foo", "bar", "baz", str(m % 2)),
                    f"key{i}",
                )
            )
            coros.append(
                store.alist_namespaces(
                    prefix=None,
                    max_depth=m + 1,
                )
            )
            coros.append(
                store.asearch(
                    ("test",),
                )
            )
            coros.append(
                store.aput(
                    ("test", "foo", "bar", "baz", str(m % 2)),
                    f"key{i}",
                    value={"foo": "bar" + str(i)},
                )
            )
            coros.append(
                store.adelete(
                    ("test", "foo", "bar", "baz", str(m % 2)),
                    f"key{i}",
                )
            )

    results = await asyncio.gather(*coros)
    assert len(results) == M * N * 6


async def test_abatch_order(store: AsyncPostgresStore) -> None:
    # Setup test data
    await store.aput(("test", "foo"), "key1", {"data": "value1"})
    await store.aput(("test", "bar"), "key2", {"data": "value2"})

    ops = [
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
    assert isinstance(results[0], Item)
    assert isinstance(results[0].value, dict)
    assert results[0].value == {"data": "value1"}
    assert results[0].key == "key1"
    assert results[1] is None
    assert isinstance(results[2], list)
    assert len(results[2]) == 1
    assert isinstance(results[3], list)
    assert ("test", "foo") in results[3] and ("test", "bar") in results[3]
    assert results[4] is None

    ops_reordered = [
        SearchOp(namespace_prefix=("test",), filter=None, limit=5, offset=0),
        GetOp(namespace=("test", "bar"), key="key2"),
        ListNamespacesOp(match_conditions=None, max_depth=None, limit=5, offset=0),
        PutOp(namespace=("test",), key="key3", value={"data": "value3"}),
        GetOp(namespace=("test", "foo"), key="key1"),
    ]

    results_reordered = await store.abatch(ops_reordered)
    assert len(results_reordered) == 5
    assert isinstance(results_reordered[0], list)
    assert len(results_reordered[0]) == 2
    assert isinstance(results_reordered[1], Item)
    assert results_reordered[1].value == {"data": "value2"}
    assert results_reordered[1].key == "key2"
    assert isinstance(results_reordered[2], list)
    assert ("test", "foo") in results_reordered[2] and (
        "test",
        "bar",
    ) in results_reordered[2]
    assert results_reordered[3] is None
    assert isinstance(results_reordered[4], Item)
    assert results_reordered[4].value == {"data": "value1"}
    assert results_reordered[4].key == "key1"


async def test_batch_get_ops(store: AsyncPostgresStore) -> None:
    # Setup test data
    await store.aput(("test",), "key1", {"data": "value1"})
    await store.aput(("test",), "key2", {"data": "value2"})

    ops = [
        GetOp(namespace=("test",), key="key1"),
        GetOp(namespace=("test",), key="key2"),
        GetOp(namespace=("test",), key="key3"),
    ]

    results = await store.abatch(ops)

    assert len(results) == 3
    assert results[0] is not None
    assert results[1] is not None
    assert results[2] is None
    assert results[0].key == "key1"
    assert results[1].key == "key2"


async def test_batch_put_ops(store: AsyncPostgresStore) -> None:
    ops = [
        PutOp(namespace=("test",), key="key1", value={"data": "value1"}),
        PutOp(namespace=("test",), key="key2", value={"data": "value2"}),
        PutOp(namespace=("test",), key="key3", value=None),
    ]

    results = await store.abatch(ops)

    assert len(results) == 3
    assert all(result is None for result in results)

    # Verify the puts worked
    items = await store.asearch(["test"], limit=10)
    assert len(items) == 2  # key3 had None value so wasn't stored


async def test_batch_search_ops(store: AsyncPostgresStore) -> None:
    # Setup test data
    await store.aput(("test", "foo"), "key1", {"data": "value1"})
    await store.aput(("test", "bar"), "key2", {"data": "value2"})

    ops = [
        SearchOp(
            namespace_prefix=("test",), filter={"data": "value1"}, limit=10, offset=0
        ),
        SearchOp(namespace_prefix=("test",), filter=None, limit=5, offset=0),
    ]

    results = await store.abatch(ops)

    assert len(results) == 2
    assert len(results[0]) == 1  # Filtered results
    assert len(results[1]) == 2  # All results


async def test_batch_list_namespaces_ops(store: AsyncPostgresStore) -> None:
    # Setup test data
    await store.aput(("test", "namespace1"), "key1", {"data": "value1"})
    await store.aput(("test", "namespace2"), "key2", {"data": "value2"})

    ops = [ListNamespacesOp(match_conditions=None, max_depth=None, limit=10, offset=0)]

    results = await store.abatch(ops)

    assert len(results) == 1
    assert len(results[0]) == 2
    assert ("test", "namespace1") in results[0]
    assert ("test", "namespace2") in results[0]


@asynccontextmanager
async def _create_vector_store(
    vector_type: str,
    distance_type: str,
    fake_embeddings: CharacterEmbeddings,
    text_fields: list[str] | None = None,
) -> AsyncIterator[AsyncPostgresStore]:
    """Create a store with vector search enabled."""

    database = f"test_{uuid.uuid4().hex[:16]}"
    uri_parts = DEFAULT_URI.split("/")
    uri_base = "/".join(uri_parts[:-1])
    query_params = ""
    if "?" in uri_parts[-1]:
        db_name, query_params = uri_parts[-1].split("?", 1)
        query_params = "?" + query_params

    conn_string = f"{uri_base}/{database}{query_params}"
    admin_conn_string = DEFAULT_URI

    index_config = {
        "dims": fake_embeddings.dims,
        "embed": fake_embeddings,
        "ann_index_config": {
            "vector_type": vector_type,
        },
        "distance_type": distance_type,
        "fields": text_fields,
    }

    async with await AsyncConnection.connect(
        admin_conn_string, autocommit=True
    ) as conn:
        await conn.execute(f"CREATE DATABASE {database}")
    try:
        async with AsyncPostgresStore.from_conn_string(
            conn_string,
            index=index_config,
        ) as store:
            await store.setup()
            yield store
    finally:
        async with await AsyncConnection.connect(
            admin_conn_string, autocommit=True
        ) as conn:
            await conn.execute(f"DROP DATABASE {database}")


@pytest.fixture(
    scope="function",
    params=[
        (vector_type, distance_type)
        for vector_type in VECTOR_TYPES
        for distance_type in (
            ["hamming"] if vector_type == "bit" else ["l2", "inner_product", "cosine"]
        )
    ],
    ids=lambda p: f"{p[0]}_{p[1]}",
)
async def vector_store(
    request,
    fake_embeddings: CharacterEmbeddings,
) -> AsyncIterator[AsyncPostgresStore]:
    """Create a store with vector search enabled."""
    vector_type, distance_type = request.param
    async with _create_vector_store(
        vector_type, distance_type, fake_embeddings
    ) as store:
        yield store


async def test_vector_store_initialization(
    vector_store: AsyncPostgresStore, fake_embeddings: CharacterEmbeddings
) -> None:
    """Test store initialization with embedding config."""
    assert vector_store.index_config is not None
    assert vector_store.index_config["dims"] == fake_embeddings.dims
    if isinstance(vector_store.index_config["embed"], Embeddings):
        assert vector_store.index_config["embed"] == fake_embeddings


async def test_vector_insert_with_auto_embedding(
    vector_store: AsyncPostgresStore,
) -> None:
    """Test inserting items that get auto-embedded."""
    docs = [
        ("doc1", {"text": "short text"}),
        ("doc2", {"text": "longer text document"}),
        ("doc3", {"text": "longest text document here"}),
        ("doc4", {"description": "text in description field"}),
        ("doc5", {"content": "text in content field"}),
        ("doc6", {"body": "text in body field"}),
    ]

    for key, value in docs:
        await vector_store.aput(("test",), key, value)

    results = await vector_store.asearch(("test",), query="long text")
    assert len(results) > 0

    doc_order = [r.key for r in results]
    assert "doc2" in doc_order
    assert "doc3" in doc_order


async def test_vector_update_with_embedding(vector_store: AsyncPostgresStore) -> None:
    """Test that updating items properly updates their embeddings."""
    await vector_store.aput(("test",), "doc1", {"text": "zany zebra Xerxes"})
    await vector_store.aput(("test",), "doc2", {"text": "something about dogs"})
    await vector_store.aput(("test",), "doc3", {"text": "text about birds"})

    results_initial = await vector_store.asearch(("test",), query="Zany Xerxes")
    assert len(results_initial) > 0
    assert results_initial[0].key == "doc1"
    initial_score = results_initial[0].score

    await vector_store.aput(("test",), "doc1", {"text": "new text about dogs"})

    results_after = await vector_store.asearch(("test",), query="Zany Xerxes")
    after_score = next((r.score for r in results_after if r.key == "doc1"), 0.0)
    assert after_score < initial_score

    results_new = await vector_store.asearch(("test",), query="new text about dogs")
    for r in results_new:
        if r.key == "doc1":
            assert r.score > after_score

    # Don't index this one
    await vector_store.aput(
        ("test",), "doc4", {"text": "new text about dogs"}, index=False
    )
    results_new = await vector_store.asearch(
        ("test",), query="new text about dogs", limit=3
    )
    assert not any(r.key == "doc4" for r in results_new)


async def test_vector_search_with_filters(vector_store: AsyncPostgresStore) -> None:
    """Test combining vector search with filters."""
    docs = [
        ("doc1", {"text": "red apple", "color": "red", "score": 4.5}),
        ("doc2", {"text": "red car", "color": "red", "score": 3.0}),
        ("doc3", {"text": "green apple", "color": "green", "score": 4.0}),
        ("doc4", {"text": "blue car", "color": "blue", "score": 3.5}),
    ]

    for key, value in docs:
        await vector_store.aput(("test",), key, value)

    results = await vector_store.asearch(
        ("test",), query="apple", filter={"color": "red"}
    )
    assert len(results) == 2
    assert results[0].key == "doc1"

    results = await vector_store.asearch(
        ("test",), query="car", filter={"color": "red"}
    )
    assert len(results) == 2
    assert results[0].key == "doc2"

    results = await vector_store.asearch(
        ("test",), query="bbbbluuu", filter={"score": {"$gt": 3.2}}
    )
    assert len(results) == 3
    assert results[0].key == "doc4"

    results = await vector_store.asearch(
        ("test",), query="apple", filter={"score": {"$gte": 4.0}, "color": "green"}
    )
    assert len(results) == 1
    assert results[0].key == "doc3"


async def test_vector_search_pagination(vector_store: AsyncPostgresStore) -> None:
    """Test pagination with vector search."""
    for i in range(5):
        await vector_store.aput(
            ("test",), f"doc{i}", {"text": f"test document number {i}"}
        )

    results_page1 = await vector_store.asearch(("test",), query="test", limit=2)
    results_page2 = await vector_store.asearch(
        ("test",), query="test", limit=2, offset=2
    )

    assert len(results_page1) == 2
    assert len(results_page2) == 2
    assert results_page1[0].key != results_page2[0].key

    all_results = await vector_store.asearch(("test",), query="test", limit=10)
    assert len(all_results) == 5


async def test_vector_search_edge_cases(vector_store: AsyncPostgresStore) -> None:
    """Test edge cases in vector search."""
    await vector_store.aput(("test",), "doc1", {"text": "test document"})

    perfect_match = await vector_store.asearch(("test",), query="text test document")
    perfect_score = perfect_match[0].score

    results = await vector_store.asearch(("test",), query="")
    assert len(results) == 1
    assert results[0].score is None

    results = await vector_store.asearch(("test",), query=None)
    assert len(results) == 1
    assert results[0].score is None

    long_query = "foo " * 100
    results = await vector_store.asearch(("test",), query=long_query)
    assert len(results) == 1
    assert results[0].score < perfect_score

    special_query = "test!@#$%^&*()"
    results = await vector_store.asearch(("test",), query=special_query)
    assert len(results) == 1
    assert results[0].score < perfect_score


@pytest.mark.parametrize(
    "vector_type,distance_type",
    [
        *itertools.product(["vector", "halfvec"], ["cosine", "inner_product", "l2"]),
    ],
)
async def test_embed_with_path(
    request: Any,
    fake_embeddings: CharacterEmbeddings,
    vector_type: str,
    distance_type: str,
) -> None:
    """Test vector search with specific text fields in Postgres store."""
    async with _create_vector_store(
        vector_type,
        distance_type,
        fake_embeddings,
        text_fields=["key0", "key1", "key3"],
    ) as store:
        # This will have 2 vectors representing it
        doc1 = {
            # Omit key0 - check it doesn't raise an error
            "key1": "xxx",
            "key2": "yyy",
            "key3": "zzz",
        }
        # This will have 3 vectors representing it
        doc2 = {
            "key0": "uuu",
            "key1": "vvv",
            "key2": "www",
            "key3": "xxx",
        }
        await store.aput(("test",), "doc1", doc1)
        await store.aput(("test",), "doc2", doc2)

        # doc2.key3 and doc1.key1 both would have the highest score
        results = await store.asearch(("test",), query="xxx")
        assert len(results) == 2
        assert results[0].key != results[1].key
        ascore = results[0].score
        bscore = results[1].score
        assert ascore == pytest.approx(bscore, abs=1e-3)

        results = await store.asearch(("test",), query="uuu")
        assert len(results) == 2
        assert results[0].key != results[1].key
        assert results[0].key == "doc2"
        assert results[0].score > results[1].score
        assert ascore == pytest.approx(results[0].score, abs=1e-3)

        # Un-indexed - will have low results for both. Not zero (because we're projecting)
        # but less than the above.
        results = await store.asearch(("test",), query="www")
        assert len(results) == 2
        assert results[0].score < ascore
        assert results[1].score < ascore


@pytest.mark.parametrize(
    "vector_type,distance_type",
    [
        *itertools.product(["vector", "halfvec"], ["cosine", "inner_product", "l2"]),
    ],
)
async def test_search_sorting(
    request: Any,
    fake_embeddings: CharacterEmbeddings,
    vector_type: str,
    distance_type: str,
) -> None:
    """Test operation-level field configuration for vector search."""
    async with _create_vector_store(
        vector_type,
        distance_type,
        fake_embeddings,
        text_fields=["key1"],  # Default fields that won't match our test data
    ) as store:
        amatch = {
            "key1": "mmm",
        }

        await store.aput(("test", "M"), "M", amatch)
        N = 100
        for i in range(N):
            await store.aput(("test", "A"), f"A{i}", {"key1": "no"})
        for i in range(N):
            await store.aput(("test", "Z"), f"Z{i}", {"key1": "no"})

        results = await store.asearch(("test",), query="mmm", limit=10)
        assert len(results) == 10
        assert len(set(r.key for r in results)) == 10
        assert results[0].key == "M"
        assert results[0].score > results[1].score


async def test_store_ttl(store):
    # Assumes a TTL of 1 minute = 60 seconds
    ns = ("foo",)
    await store.start_ttl_sweeper()
    await store.aput(
        ns,
        key="item1",
        value={"foo": "bar"},
        ttl=TTL_MINUTES,  # type: ignore
    )
    await asyncio.sleep(TTL_SECONDS - 2)
    res = await store.aget(ns, key="item1", refresh_ttl=True)
    assert res is not None
    await asyncio.sleep(TTL_SECONDS - 2)
    results = await store.asearch(ns, query="foo", refresh_ttl=True)
    assert len(results) == 1
    await asyncio.sleep(TTL_SECONDS - 2)
    res = await store.aget(ns, key="item1", refresh_ttl=False)
    assert res is not None
    await asyncio.sleep(TTL_SECONDS - 1)
    # Now has been (TTL_SECONDS-2)*2 > TTL_SECONDS + TTL_SECONDS/2
    results = await store.asearch(ns, query="bar", refresh_ttl=False)
    assert len(results) == 0
