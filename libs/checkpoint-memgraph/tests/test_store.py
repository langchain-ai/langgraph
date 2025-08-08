from __future__ import annotations

import socket
import time
from collections.abc import Generator
from contextlib import contextmanager
from typing import Any, Union, cast
from urllib.parse import unquote, urlparse

import pytest
from neo4j import Driver, GraphDatabase

from langgraph.store.base import (
    GetOp,
    Item,
    ListNamespacesOp,
    MatchCondition,
    PutOp,
    SearchItem,
    SearchOp,
    TTLConfig,
)
from langgraph.store.memgraph import MemgraphIndexConfig, MemgraphStore
from tests.conftest import (
    DEFAULT_MEMGRAPH_URI,
    CharacterEmbeddings,
)

TTL_SECONDS = 2
TTL_MINUTES = TTL_SECONDS / 60


def is_memgraph_unavailable() -> bool:
    """
    Check if a Memgraph instance is unavailable.

    Returns:
        bool: True if a Memgraph instance is not available, False otherwise.
    """
    try:
        # Try to create a connection to the default Memgraph port.
        parsed_uri = urlparse(DEFAULT_MEMGRAPH_URI)
        if parsed_uri.port is None:
            return True
        with socket.create_connection(
            (parsed_uri.hostname, parsed_uri.port), timeout=1
        ):
            return False
    except (socket.timeout, ConnectionRefusedError):
        return True


# Skip all tests in this module if Memgraph is not available.
pytestmark = pytest.mark.skipif(
    is_memgraph_unavailable(), reason="Memgraph instance not available"
)


@pytest.fixture(scope="session")
def driver() -> Generator[Driver, Any, None]:
    parsed = urlparse(DEFAULT_MEMGRAPH_URI)
    uri = f"{parsed.scheme}://{parsed.hostname}:{parsed.port or 7687}"
    auth = (unquote(parsed.username or ""), unquote(parsed.password or ""))
    with GraphDatabase.driver(uri, auth=auth) as driver:
        # Ensure the driver is connected and ready
        driver.verify_connectivity()
        yield driver


@pytest.fixture(scope="function")
def store(driver: Driver) -> Generator[MemgraphStore, Any, None]:
    """A MemgraphStore fixture that cleans the database before each test, without TTL."""
    with driver.session() as session:
        session.run("MATCH (n) DETACH DELETE n").consume()
    # Using a shared driver, so we instantiate the store directly.
    # No TTL sweeper is started for these basic tests to reduce overhead.
    store = MemgraphStore(driver)
    store.setup()
    yield store


def test_from_conn_string() -> None:
    namespace = ("test_conn_string",)
    key = "key1"
    with MemgraphStore.from_conn_string(DEFAULT_MEMGRAPH_URI) as store:
        store.delete(namespace, key)
        store.put(namespace, key, {"data": "value1"})
        item = store.get(namespace, key)
        assert item is not None
        assert item.value == {"data": "value1"}
        store.delete(namespace, key)
        item_after_delete = store.get(namespace, key)
        assert item_after_delete is None


def test_batch_order(store: MemgraphStore) -> None:
    # Setup test data
    store.put(("test", "foo"), "key1", {"data": "value1"})
    store.put(("test", "bar"), "key2", {"data": "value2"})
    ops: list[Union[GetOp, PutOp, SearchOp, ListNamespacesOp]] = [
        GetOp(namespace=("test", "foo"), key="key1"),
        PutOp(namespace=("test", "bar"), key="key2", value={"data": "value2"}),
        SearchOp(
            namespace_prefix=("test",), filter={"data": "value1"}, limit=10, offset=0
        ),
        ListNamespacesOp(match_conditions=None, max_depth=None, limit=10, offset=0),
        GetOp(namespace=("test",), key="key3"),
    ]
    results = store.batch(ops)
    assert len(results) == 5
    assert isinstance(results[0], Item)
    assert isinstance(results[0].value, dict)
    assert results[0].value == {"data": "value1"}
    assert results[0].key == "key1"
    assert results[1] is None  # Put operation returns None
    assert isinstance(results[2], list)
    assert len(results[2]) == 1
    assert isinstance(results[3], list)
    assert len(results[3]) >= 2
    assert results[4] is None  # Non-existent key returns None
    # Test reordered operations
    ops_reordered: list[Union[SearchOp, GetOp, ListNamespacesOp, PutOp]] = [
        SearchOp(namespace_prefix=("test",), filter=None, limit=5, offset=0),
        GetOp(namespace=("test", "bar"), key="key2"),
        ListNamespacesOp(match_conditions=None, max_depth=None, limit=5, offset=0),
        PutOp(namespace=("test",), key="key3", value={"data": "value3"}),
        GetOp(namespace=("test", "foo"), key="key1"),
    ]
    results_reordered = store.batch(ops_reordered)
    assert len(results_reordered) == 5
    assert isinstance(results_reordered[0], list)
    assert len(results_reordered[0]) >= 2
    assert isinstance(results_reordered[1], Item)
    assert results_reordered[1].value == {"data": "value2"}
    assert results_reordered[1].key == "key2"
    assert isinstance(results_reordered[2], list)
    assert len(results_reordered[2]) > 0
    assert results_reordered[3] is None
    assert isinstance(results_reordered[4], Item)
    assert results_reordered[4].value == {"data": "value1"}


def test_batch_get_ops(store: MemgraphStore) -> None:
    # Setup test data
    store.put(("test",), "key1", {"data": "value1"})
    store.put(("test",), "key2", {"data": "value2"})
    ops = [
        GetOp(namespace=("test",), key="key1"),
        GetOp(namespace=("test",), key="key2"),
        GetOp(namespace=("test",), key="key3"),  # Non-existent key
    ]
    results = store.batch(ops)
    assert len(results) == 3
    assert results[0] is not None
    assert results[1] is not None
    assert results[2] is None
    assert isinstance(results[0], Item)
    assert results[0].key == "key1"
    assert isinstance(results[1], Item)
    assert results[1].key == "key2"


def test_batch_put_ops(store: MemgraphStore) -> None:
    ops = [
        PutOp(namespace=("test",), key="key1", value={"data": "value1"}),
        PutOp(namespace=("test",), key="key2", value={"data": "value2"}),
        PutOp(namespace=("test",), key="key3", value=None),  # Delete operation
    ]
    results = store.batch(ops)
    assert len(results) == 3
    assert all(result is None for result in results)
    # Verify the puts worked
    item1 = store.get(("test",), "key1")
    item2 = store.get(("test",), "key2")
    item3 = store.get(("test",), "key3")
    assert item1 and item1.value == {"data": "value1"}
    assert item2 and item2.value == {"data": "value2"}
    assert item3 is None


def test_batch_search_ops(store: MemgraphStore) -> None:
    # Setup test data
    test_data = [
        (("test", "foo"), "key1", {"data": "value1", "tag": "a"}),
        (("test", "bar"), "key2", {"data": "value2", "tag": "a"}),
        (("test", "baz"), "key3", {"data": "value3", "tag": "b"}),
    ]
    for namespace, key, value in test_data:
        store.put(namespace, key, value)
    ops = [
        SearchOp(namespace_prefix=("test",), filter={"tag": "a"}, limit=10, offset=0),
        SearchOp(namespace_prefix=("test",), filter=None, limit=2, offset=0),
        SearchOp(namespace_prefix=("test", "foo"), filter=None, limit=10, offset=0),
    ]
    results = store.batch(ops)
    assert len(results) == 3
    # First search should find items with tag "a"
    assert isinstance(results[0], list)
    assert len(results[0]) == 2
    assert all(item.value["tag"] == "a" for item in cast(list[SearchItem], results[0]))
    # Second search should return first 2 items (order by updated_at desc)
    assert isinstance(results[1], list)
    assert len(results[1]) == 2
    # Third search should only find items in test.foo namespace
    assert isinstance(results[2], list)
    assert len(results[2]) == 1
    assert cast(list[SearchItem], results[2])[0].namespace == ("test", "foo")


def test_batch_list_namespaces_ops(store: MemgraphStore) -> None:
    # Setup test data with various namespaces
    test_data = [
        (("test", "documents", "public"), "doc1", {"content": "public doc"}),
        (("test", "documents", "private"), "doc2", {"content": "private doc"}),
        (("test", "images", "public"), "img1", {"content": "public image"}),
        (("prod", "documents", "public"), "doc3", {"content": "prod doc"}),
    ]
    for namespace, key, value in test_data:
        store.put(namespace, key, value)
    ops = [
        ListNamespacesOp(match_conditions=None, max_depth=None, limit=10, offset=0),
        ListNamespacesOp(match_conditions=None, max_depth=2, limit=10, offset=0),
        ListNamespacesOp(
            match_conditions=(MatchCondition(match_type="suffix", path=("public",)),),
            max_depth=None,
            limit=10,
            offset=0,
        ),
    ]
    results = store.batch(ops)
    assert len(results) == 3
    assert isinstance(results[0], list)
    assert len(results[0]) >= len(test_data)
    assert isinstance(results[1], list)
    assert all(len(ns) <= 2 for ns in cast(list[tuple[str, ...]], results[1]))
    assert isinstance(results[2], list)
    assert all(ns[-1] == "public" for ns in cast(list[tuple[str, ...]], results[2]))


def test_basic_store_ops(store: MemgraphStore) -> None:
    namespace = ("test", "documents")
    item_id = "doc1"
    item_value = {"title": "Test Document", "content": "Hello, World!"}
    store.put(namespace, item_id, item_value)
    item = store.get(namespace, item_id)
    assert item
    assert item.namespace == namespace
    assert item.key == item_id
    assert item.value == item_value
    # Test update
    updated_value = {"title": "Updated Document", "content": "Hello, Updated!"}
    store.put(namespace, item_id, updated_value)
    updated_item = store.get(namespace, item_id)
    assert updated_item is not None
    assert updated_item.value == updated_value
    assert updated_item.updated_at > item.updated_at
    # Test get from non-existent namespace
    different_namespace = ("test", "other_documents")
    item_in_different_namespace = store.get(different_namespace, item_id)
    assert item_in_different_namespace is None
    # Test delete
    store.delete(namespace, item_id)
    deleted_item = store.get(namespace, item_id)
    assert deleted_item is None


def test_list_namespaces(store: MemgraphStore) -> None:
    # Create test data with various namespaces
    test_namespaces = [
        ("test", "documents", "public"),
        ("test", "documents", "private"),
        ("test", "images", "public"),
        ("test", "images", "private"),
        ("prod", "documents", "public"),
        ("prod", "documents", "private"),
    ]
    for namespace in test_namespaces:
        store.put(namespace, "dummy", {"content": "dummy"})
    all_namespaces = store.list_namespaces()
    assert len(all_namespaces) >= len(test_namespaces)
    test_prefix_namespaces = store.list_namespaces(prefix=("test",))
    assert len(test_prefix_namespaces) == 4
    assert all(ns[0] == "test" for ns in test_prefix_namespaces)
    public_namespaces = store.list_namespaces(suffix=("public",))
    assert len(public_namespaces) == 3
    assert all(ns[-1] == "public" for ns in public_namespaces)
    depth_2_namespaces = store.list_namespaces(max_depth=2)
    assert all(len(ns) <= 2 for ns in depth_2_namespaces)
    paginated_namespaces = store.list_namespaces(limit=3)
    assert len(paginated_namespaces) == 3
    for namespace in test_namespaces:
        store.delete(namespace, "dummy")


def test_search(store: MemgraphStore) -> None:
    # Create test data
    test_data = [
        (
            ("test", "docs"),
            "doc1",
            {"title": "First Doc", "author": "Alice", "tags": ["important"]},
        ),
        (
            ("test", "docs"),
            "doc2",
            {"title": "Second Doc", "author": "Bob", "tags": ["draft"]},
        ),
        (
            ("test", "images"),
            "img1",
            {"title": "Image 1", "author": "Alice", "tags": ["final"]},
        ),
    ]
    for namespace, key, value in test_data:
        store.put(namespace, key, value)
    all_items = store.search(("test",))
    assert len(all_items) == 3
    docs_items = store.search(("test", "docs"))
    assert len(docs_items) == 2
    assert all(item.namespace == ("test", "docs") for item in docs_items)
    alice_items = store.search(("test",), filter={"author": "Alice"})
    assert len(alice_items) == 2
    assert all(cast(dict, item.value).get("author") == "Alice" for item in alice_items)
    paginated_items = store.search(("test",), limit=2)
    assert len(paginated_items) == 2
    offset_items = store.search(("test",), offset=2)
    assert len(offset_items) == 1
    for namespace, key, _ in test_data:
        store.delete(namespace, key)


@pytest.fixture(
    scope="function",
    params=["l2sq", "cos", "ip"],
)
def vector_store(
    driver: Driver,
    request: Any,
    fake_embeddings: CharacterEmbeddings,
) -> Generator[MemgraphStore, Any, None]:
    metric = request.param
    index_config: MemgraphIndexConfig = {
        "dimension": fake_embeddings.dims,
        "capacity": 1000,
        "embed": fake_embeddings,
        "metric": metric,
        "fields": ["text"],
    }
    # Clean the database before each test. This is critical for vector tests
    # that are sensitive to index state.
    with driver.session() as session:
        session.run("MATCH (n) DETACH DELETE n").consume()
        try:
            # Drop the vector index to ensure a clean slate for each test run,
            # especially since parameters like the metric can change.
            session.run("DROP VECTOR INDEX vector_index").consume()
        except Exception:
            pass  # Fails if the index does not exist, which is fine.

    # The sleep is unfortunate but appears necessary to prevent race conditions
    # with Memgraph's index creation/deletion. Its duration is tuned to balance
    # stability and test speed.
    time.sleep(0.2)

    store = MemgraphStore(driver, index=index_config, ttl=None)
    store.setup()
    yield store


@contextmanager
def _create_vector_store_with_text_fields(
    driver: Driver,
    metric: str,
    fake_embeddings: CharacterEmbeddings,
    text_fields: list[str] | None = None,
) -> Generator[MemgraphStore, Any, None]:
    index_config: MemgraphIndexConfig = {
        "dimension": fake_embeddings.dims,
        "capacity": 1000,
        "embed": fake_embeddings,
        "metric": metric,
        "fields": text_fields,
    }
    with driver.session() as session:
        session.run("MATCH (n) DETACH DELETE n").consume()
        try:
            session.run("DROP VECTOR INDEX vector_index").consume()
        except Exception:
            pass  # Fails if the index does not exist, which is fine.

    # A short sleep to allow the database to stabilize after cleanup.
    time.sleep(0.1)

    store = MemgraphStore(driver, index=index_config)
    store.setup()
    yield store


def test_vector_store_initialization(
    vector_store: MemgraphStore, fake_embeddings: CharacterEmbeddings
) -> None:
    assert vector_store.index_config is not None
    assert vector_store.index_config["dimension"] == fake_embeddings.dims
    assert vector_store.index_config["embed"] == fake_embeddings


def test_vector_insert_with_auto_embedding(vector_store: MemgraphStore) -> None:
    docs = [
        ("doc1", {"text": "short text"}),
        ("doc2", {"text": "longer text document"}),
        ("doc3", {"text": "longest text document here"}),
    ]
    for key, value in docs:
        vector_store.put(("test",), key, value, index=["text"])
    results = vector_store.search(("test",), query="long text")
    assert len(results) > 0
    doc_order = [r.key for r in results]
    assert "doc2" in doc_order
    assert "doc3" in doc_order


def test_vector_update_with_embedding(vector_store: MemgraphStore) -> None:
    vector_store.put(("test",), "doc1", {"text": "zany zebra Xerxes"}, index=["text"])
    vector_store.put(
        ("test",), "doc2", {"text": "something about dogs"}, index=["text"]
    )
    results_initial = vector_store.search(("test",), query="Zany Xerxes")
    assert len(results_initial) > 0
    assert results_initial[0].key == "doc1"
    initial_score = results_initial[0].score
    assert initial_score is not None
    vector_store.put(("test",), "doc1", {"text": "new text about dogs"}, index=["text"])
    results_after = vector_store.search(("test",), query="Zany Xerxes")
    after_score = next((r.score for r in results_after if r.key == "doc1"), None)
    assert after_score is not None
    assert after_score < initial_score
    results_new = vector_store.search(("test",), query="new text about dogs")
    for r in results_new:
        if r.key == "doc1":
            assert r.score is not None
            assert r.score > after_score
    # Don't index this one
    vector_store.put(("test",), "doc4", {"text": "new text about dogs"}, index=False)
    results_no_index = vector_store.search(
        ("test",), query="new text about dogs", limit=3
    )
    assert not any(r.key == "doc4" for r in results_no_index)


@pytest.mark.parametrize("refresh_ttl", [True, False])
def test_vector_search_with_filters(
    vector_store: MemgraphStore, refresh_ttl: bool
) -> None:
    docs = [
        ("doc1", {"text": "red apple", "color": "red"}),
        ("doc2", {"text": "red car", "color": "red"}),
        ("doc3", {"text": "green apple", "color": "green"}),
    ]
    for key, value in docs:
        vector_store.put(("test",), key, value, index=["text"])

    results = vector_store.search(
        ("test",), query="apple", filter={"color": "red"}, refresh_ttl=refresh_ttl
    )
    assert len(results) >= 1
    assert results[0].key == "doc1"


def test_vector_search_pagination(vector_store: MemgraphStore) -> None:
    for i in range(5):
        vector_store.put(
            ("test",), f"doc{i}", {"text": f"test document number {i}"}, index=["text"]
        )
    results_page1 = vector_store.search(("test",), query="test", limit=2)
    results_page2 = vector_store.search(("test",), query="test", limit=2, offset=2)
    assert len(results_page1) == 2
    assert len(results_page2) == 2
    assert results_page1[0].key != results_page2[0].key
    all_results = vector_store.search(("test",), query="test", limit=10)
    assert len(all_results) == 5


def test_vector_search_edge_cases(vector_store: MemgraphStore) -> None:
    vector_store.put(("test",), "doc1", {"text": "test document"}, index=["text"])
    results = vector_store.search(("test",), query="")
    assert len(results) == 1
    results = vector_store.search(("test",), query=None)
    assert len(results) == 1


def test_embed_with_path_sync(
    driver: Driver,
    fake_embeddings: CharacterEmbeddings,
) -> None:
    with _create_vector_store_with_text_fields(
        driver,
        "cos",
        fake_embeddings,
        text_fields=["key0", "key1", "key3"],
    ) as store:
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
        store.put(("test",), "doc1", doc1)
        store.put(("test",), "doc2", doc2)
        # doc2.key3 and doc1.key1 both would have the highest score
        results: list[SearchItem] = store.search(("test",), query="xxx")
        assert len(results) == 2
        assert results[0].key != results[1].key
        ascore = results[0].score
        bscore = results[1].score
        assert ascore is not None
        assert bscore is not None
        assert ascore == pytest.approx(bscore, abs=1e-3)
        # ~Only match doc2
        results = store.search(("test",), query="uuu")
        assert len(results) == 2
        assert results[0].key != results[1].key
        assert results[0].key == "doc2"
        assert results[0].score is not None
        assert results[1].score is not None
        assert results[0].score > results[1].score
        assert ascore == pytest.approx(results[0].score, abs=1e-3)
        # ~Only match doc1
        results = store.search(("test",), query="zzz")
        assert len(results) == 2
        assert results[0].key != results[1].key
        assert results[0].key == "doc1"
        assert results[0].score is not None
        assert results[1].score is not None
        assert results[0].score > results[1].score
        assert ascore == pytest.approx(results[0].score, abs=1e-3)
        # Un-indexed - will have low results for both. Not zero (because we're projecting)
        # but less than the above.
        results = store.search(("test",), query="www")
        assert len(results) == 2
        assert results[0].key != results[1].key
        assert results[0].score is not None
        assert results[1].score is not None
        assert results[0].score < ascore
        assert results[1].score < ascore


def test_embed_with_path_operation_config(
    driver: Driver,
    fake_embeddings: CharacterEmbeddings,
) -> None:
    with _create_vector_store_with_text_fields(
        driver, "cos", fake_embeddings, text_fields=["key17"]
    ) as store:
        doc3 = {
            "key0": "aaa",
            "key1": "bbb",
            "key2": "ccc",
            "key3": "ddd",
        }
        doc4 = {
            "key0": "eee",
            "key1": "bbb",  # Same as doc3.key1
            "key2": "fff",
            "key3": "ggg",
        }
        store.put(("test",), "doc3", doc3, index=["key0", "key1"])
        store.put(("test",), "doc4", doc4, index=["key1", "key3"])
        results: list[SearchItem] = store.search(("test",), query="aaa")
        assert len(results) == 2
        assert results[0].key == "doc3"
        assert len(set(r.key for r in results)) == 2
        assert results[0].score is not None
        assert results[1].score is not None
        assert results[0].score > results[1].score
        results = store.search(("test",), query="ggg")
        assert len(results) == 2
        assert results[0].key == "doc4"
        assert results[0].score is not None
        assert results[1].score is not None
        assert results[0].score > results[1].score
        results = store.search(("test",), query="bbb")
        assert len(results) == 2
        assert results[0].key != results[1].key
        assert results[0].score is not None
        assert results[1].score is not None
        assert results[0].score == pytest.approx(results[1].score, abs=1e-3)
        results = store.search(("test",), query="ccc")
        assert len(results) == 2
        assert all(
            r.score is not None and r.score < 0.9 for r in results
        )  # Unindexed field should have low scores
        # Test index=False behavior
        doc5 = {
            "key0": "hhh",
            "key1": "iii",
        }
        store.put(("test",), "doc5", doc5, index=False)
        results = store.search(("test",))
        assert len(results) == 3
        assert all(r.score is None for r in results), f"{results}"
        assert any(r.key == "doc5" for r in results)
        store.search(("test",), query="hhh")


def _cosine_similarity(X: list[float], Y: list[list[float]]) -> list[float]:
    """
    Compute cosine similarity between a vector X and a matrix Y.
    Lazy import numpy for efficiency.
    """

    similarities = []
    for y in Y:
        dot_product = sum(a * b for a, b in zip(X, y))
        norm1 = sum(a * a for a in X) ** 0.5
        norm2 = sum(a * a for a in y) ** 0.5
        similarity = dot_product / (norm1 * norm2) if norm1 > 0 and norm2 > 0 else 0.0
        similarities.append(similarity)

    return similarities


def _inner_product(X: list[float], Y: list[list[float]]) -> list[float]:
    """
    Compute inner product between a vector X and a matrix Y.
    Lazy import numpy for efficiency.
    """

    similarities = []
    for y in Y:
        similarity = sum(a * b for a, b in zip(X, y))
        similarities.append(similarity)

    return similarities


def _neg_l2_distance(X: list[float], Y: list[list[float]]) -> list[float]:
    """
    Compute l2 distance between a vector X and a matrix Y.
    Lazy import numpy for efficiency.
    """
    similarities = []
    for y in Y:
        similarity = sum((a - b) ** 2 for a, b in zip(X, y)) ** 0.5
        similarities.append(-similarity)
    return similarities


@pytest.mark.parametrize("metric", ["cos", "ip", "l2sq"])
@pytest.mark.parametrize("query", ["aaa", "abcd", "poisson"])
def test_scores(
    driver: Driver,
    fake_embeddings: CharacterEmbeddings,
    metric: str,
    query: str,
) -> None:
    with _create_vector_store_with_text_fields(
        driver, metric, fake_embeddings, text_fields=["key0"]
    ) as store:
        doc = {"key0": "aaa"}
        store.put(("test",), "doc", doc, index=["key0", "key1"])

        results = store.search((), query=query)
        vec0 = fake_embeddings.embed_query(doc["key0"])
        vec1 = fake_embeddings.embed_query(query)

        if metric == "cos":
            similarities = _cosine_similarity(vec1, [vec0])
        elif metric == "ip":
            similarities = _inner_product(vec1, [vec0])
        else:  # l2sq
            similarities = _neg_l2_distance(vec1, [vec0])

        assert len(results) == 1
        assert results[0].score == pytest.approx(similarities[0], abs=1e-3)


def test_nonnull_migrations() -> None:
    for migration in MemgraphStore.MIGRATIONS:
        assert migration.strip()


def test_store_ttl(driver: Driver) -> None:
    """Tests TTL functionality with a dedicated store instance."""
    ttl_config: TTLConfig = {
        "default_ttl": TTL_MINUTES,
        "refresh_on_read": True,
        # The sweep interval is aggressive to make the test run faster.
        "sweep_interval_minutes": TTL_MINUTES / 2,
    }
    # This test requires a store with a TTL sweeper.
    # We create it here to avoid running the sweeper for all other tests.
    store = MemgraphStore(driver, ttl=ttl_config)
    store.setup()
    store.start_ttl_sweeper()
    try:
        ns = ("foo",)
        store.put(ns, key="item1", value={"foo": "bar"}, ttl=TTL_MINUTES)
        time.sleep(TTL_SECONDS + 0.1)
        # Item should have expired
        res = store.get(ns, key="item1")
        assert res is None
        # Test refresh on read
        store.put(ns, key="item2", value={"foo": "baz"}, ttl=TTL_MINUTES)
        time.sleep(TTL_SECONDS / 2)
        res = store.get(ns, key="item2", refresh_ttl=True)
        assert res is not None
        time.sleep(TTL_SECONDS / 2 + 0.1)
        # TTL was refreshed, so it should still exist
        res = store.get(ns, key="item2", refresh_ttl=False)
        assert res is not None
        time.sleep(TTL_SECONDS / 2 + 0.1)
        # Now it should have expired
        res = store.get(ns, key="item2", refresh_ttl=False)
        assert res is None
    finally:
        store.stop_ttl_sweeper()


def test_vector_store_with_ttl(
    driver: Driver, fake_embeddings: CharacterEmbeddings
) -> None:
    """Dedicated test for TTL functionality with a vector-indexed store."""
    index_config: MemgraphIndexConfig = {
        "dimension": fake_embeddings.dims,
        "capacity": 1000,
        "embed": fake_embeddings,
        "metric": "cos",
        "fields": ["text"],
    }
    ttl_config: TTLConfig = {
        "default_ttl": TTL_MINUTES,
        "refresh_on_read": False,
        "sweep_interval_minutes": TTL_MINUTES / 2,
    }
    with driver.session() as session:
        session.run("MATCH (n) DETACH DELETE n").consume()
        try:
            session.run("DROP VECTOR INDEX vector_index").consume()
        except Exception:
            pass
    time.sleep(0.1)
    store = MemgraphStore(driver, index=index_config, ttl=ttl_config)
    store.setup()
    store.start_ttl_sweeper()
    try:
        ns = ("vector_ttl",)
        value = {"text": "This document should expire"}
        store.put(ns, key="item1", value=value, ttl=TTL_MINUTES, index=["text"])
        # Verify it's searchable
        results = store.search(ns, query="expire")
        assert len(results) == 1
        assert results[0].key == "item1"
        # Wait for it to expire
        time.sleep(TTL_SECONDS + 0.1)
        # Verify it's gone from search
        results_after = store.search(ns, query="expire")
        assert len(results_after) == 0
        # Verify it's gone from get
        item_after = store.get(ns, "item1")
        assert item_after is None
    finally:
        store.stop_ttl_sweeper()