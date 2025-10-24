# mypy: disable-error-code="union-attr,arg-type,index,operator"
import os
import re
import tempfile
import uuid
from collections.abc import Generator, Iterable
from contextlib import contextmanager
from typing import Any, Literal, cast

import pytest
from langchain_core.embeddings import Embeddings
from langgraph.store.base import (
    GetOp,
    Item,
    ListNamespacesOp,
    MatchCondition,
    PutOp,
    SearchOp,
)

from langgraph.store.sqlite import SqliteStore
from langgraph.store.sqlite.base import SqliteIndexConfig


# Local embeddings implementation for testing vector search
class CharacterEmbeddings(Embeddings):
    """Simple character-frequency based embeddings using random projections."""

    def __init__(self, dims: int = 50, seed: int = 42):
        """Initialize with embedding dimensions and random seed."""
        import math
        import random
        from collections import defaultdict

        self._rng = random.Random(seed)
        self.dims = dims
        # Create projection vector for each character lazily
        self._char_projections: dict[str, list[float]] = defaultdict(
            lambda: [
                self._rng.gauss(0, 1 / math.sqrt(self.dims)) for _ in range(self.dims)
            ]
        )

    def _embed_one(self, text: str) -> list[float]:
        """Embed a single text."""
        import math
        from collections import Counter

        counts = Counter(text)
        total = sum(counts.values())

        if total == 0:
            return [0.0] * self.dims

        embedding = [0.0] * self.dims
        for char, count in counts.items():
            weight = count / total
            char_proj = self._char_projections[char]
            for i, proj in enumerate(char_proj):
                embedding[i] += weight * proj

        norm = math.sqrt(sum(x * x for x in embedding))
        if norm > 0:
            embedding = [x / norm for x in embedding]

        return embedding

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Embed a list of documents."""
        return [self._embed_one(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        """Embed a query string."""
        return self._embed_one(text)

    def __eq__(self, other: Any) -> bool:
        return isinstance(other, CharacterEmbeddings) and self.dims == other.dims


@pytest.fixture(scope="function", params=["memory", "file"])
def store(request: Any) -> Generator[SqliteStore, None, None]:
    """Create a SqliteStore for testing."""
    if request.param == "memory":
        # In-memory store
        with SqliteStore.from_conn_string(":memory:") as store:
            store.setup()
            yield store
    else:
        # Temporary file store
        temp_file = tempfile.NamedTemporaryFile(delete=False)
        temp_file.close()
        try:
            with SqliteStore.from_conn_string(temp_file.name) as store:
                store.setup()
                yield store
        finally:
            os.unlink(temp_file.name)


@pytest.fixture(scope="function")
def fake_embeddings() -> CharacterEmbeddings:
    """Create fake embeddings for testing."""
    return CharacterEmbeddings(dims=500)


# Define vector types and distance types for parametrized tests
VECTOR_TYPES = ["cosine"]  # SQLite only supports cosine similarity


@contextmanager
def create_vector_store(
    fake_embeddings: CharacterEmbeddings,
    text_fields: list[str] | None = None,
    distance_type: str = "cosine",
    conn_type: Literal["memory", "file"] = "memory",
) -> Generator[SqliteStore, None, None]:
    """Create a SqliteStore with vector search enabled."""
    index_config: SqliteIndexConfig = {
        "dims": fake_embeddings.dims,
        "embed": fake_embeddings,
        "text_fields": text_fields,
        "distance_type": distance_type,  # This is for API consistency but SQLite only supports cosine
    }
    if conn_type == "memory":
        conn_str = ":memory:"
    else:
        temp_file = tempfile.NamedTemporaryFile(delete=False)
        temp_file.close()
        conn_str = temp_file.name

    try:
        with SqliteStore.from_conn_string(conn_str, index=index_config) as store:
            store.setup()
            yield store
    finally:
        if conn_type == "file":
            os.unlink(conn_str)


def test_batch_order(store: SqliteStore) -> None:
    # Setup test data
    store.put(("test", "foo"), "key1", {"data": "value1"})
    store.put(("test", "bar"), "key2", {"data": "value2"})

    ops = [
        GetOp(namespace=("test", "foo"), key="key1"),
        PutOp(namespace=("test", "bar"), key="key2", value={"data": "value2"}),
        SearchOp(
            namespace_prefix=("test",), filter={"data": "value1"}, limit=10, offset=0
        ),
        ListNamespacesOp(match_conditions=None, max_depth=None, limit=10, offset=0),
        GetOp(namespace=("test",), key="key3"),
    ]

    results = store.batch(
        cast(Iterable[GetOp | PutOp | SearchOp | ListNamespacesOp], ops)
    )
    assert len(results) == 5
    assert isinstance(results[0], Item)
    assert isinstance(results[0].value, dict)
    assert results[0].value == {"data": "value1"}
    assert results[0].key == "key1"
    assert results[0].namespace == ("test", "foo")
    assert results[1] is None  # Put operation returns None
    assert isinstance(results[2], list)
    assert len(results[2]) == 1
    assert results[2][0].key == "key1"
    assert results[2][0].value == {"data": "value1"}
    assert isinstance(results[3], list)
    assert len(results[3]) > 0  # Should contain at least our test namespaces
    assert ("test", "foo") in results[3]
    assert ("test", "bar") in results[3]
    assert results[4] is None  # Non-existent key returns None

    # Test reordered operations
    ops_reordered = [
        SearchOp(namespace_prefix=("test",), filter=None, limit=5, offset=0),
        GetOp(namespace=("test", "bar"), key="key2"),
        ListNamespacesOp(match_conditions=None, max_depth=None, limit=5, offset=0),
        PutOp(namespace=("test",), key="key3", value={"data": "value3"}),
        GetOp(namespace=("test", "foo"), key="key1"),
    ]

    results_reordered = store.batch(
        cast(Iterable[GetOp | PutOp | SearchOp | ListNamespacesOp], ops_reordered)
    )
    assert len(results_reordered) == 5
    assert isinstance(results_reordered[0], list)
    assert len(results_reordered[0]) >= 2  # Should find at least our two test items
    assert isinstance(results_reordered[1], Item)
    assert results_reordered[1].value == {"data": "value2"}
    assert results_reordered[1].key == "key2"
    assert results_reordered[1].namespace == ("test", "bar")
    assert isinstance(results_reordered[2], list)
    assert len(results_reordered[2]) > 0
    assert results_reordered[3] is None  # Put operation returns None
    assert isinstance(results_reordered[4], Item)
    assert results_reordered[4].value == {"data": "value1"}
    assert results_reordered[4].key == "key1"
    assert results_reordered[4].namespace == ("test", "foo")

    # Verify the put worked
    item3 = store.get(("test",), "key3")
    assert item3 is not None
    assert item3.value == {"data": "value3"}


def test_batch_get_ops(store: SqliteStore) -> None:
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
    assert results[0].key == "key1"
    assert results[1].key == "key2"


def test_batch_put_ops(store: SqliteStore) -> None:
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


def test_batch_search_ops(store: SqliteStore) -> None:
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
    assert len(results[0]) == 2
    assert all(item.value["tag"] == "a" for item in results[0])

    # Second search should return first 2 items
    assert len(results[1]) == 2

    # Third search should only find items in test/foo namespace
    assert len(results[2]) == 1
    assert results[2][0].namespace == ("test", "foo")


def test_batch_list_namespaces_ops(store: SqliteStore) -> None:
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
            match_conditions=tuple([MatchCondition("suffix", ("public",))]),
            max_depth=None,
            limit=10,
            offset=0,
        ),
    ]

    results = store.batch(
        cast(Iterable[GetOp | PutOp | SearchOp | ListNamespacesOp], ops)
    )
    assert len(results) == 3

    # First operation should list all namespaces
    assert len(results[0]) == len(test_data)

    # Second operation should only return namespaces up to depth 2
    assert all(len(ns) <= 2 for ns in results[1])

    # Third operation should only return namespaces ending with "public"
    assert all(ns[-1] == "public" for ns in results[2])


class TestSqliteStore:
    def test_basic_store_ops(self) -> None:
        with SqliteStore.from_conn_string(":memory:") as store:
            store.setup()
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
            # Small delay to ensure the updated timestamp is different
            import time

            time.sleep(0.01)

            updated_value = {"title": "Updated Document", "content": "Hello, Updated!"}
            store.put(namespace, item_id, updated_value)
            updated_item = store.get(namespace, item_id)

            assert updated_item.value == updated_value
            # Don't check timestamps because SQLite execution might be too fast
            # assert updated_item.updated_at > item.updated_at

            # Test get from non-existent namespace
            different_namespace = ("test", "other_documents")
            item_in_different_namespace = store.get(different_namespace, item_id)
            assert item_in_different_namespace is None

            # Test delete
            store.delete(namespace, item_id)
            deleted_item = store.get(namespace, item_id)
            assert deleted_item is None

    def test_list_namespaces(self) -> None:
        with SqliteStore.from_conn_string(":memory:") as store:
            store.setup()
            # Create test data with various namespaces
            test_namespaces = [
                ("test", "documents", "public"),
                ("test", "documents", "private"),
                ("test", "images", "public"),
                ("test", "images", "private"),
                ("prod", "documents", "public"),
                ("prod", "documents", "private"),
            ]

            # Insert test data
            for namespace in test_namespaces:
                store.put(namespace, "dummy", {"content": "dummy"})

            # Test listing with various filters
            all_namespaces = store.list_namespaces()
            assert len(all_namespaces) == len(test_namespaces)

            # Test prefix filtering
            test_prefix_namespaces = store.list_namespaces(prefix=["test"])
            assert len(test_prefix_namespaces) == 4
            assert all(ns[0] == "test" for ns in test_prefix_namespaces)

            # Test suffix filtering
            public_namespaces = store.list_namespaces(suffix=["public"])
            assert len(public_namespaces) == 3
            assert all(ns[-1] == "public" for ns in public_namespaces)

            # Test max depth
            depth_2_namespaces = store.list_namespaces(max_depth=2)
            assert all(len(ns) <= 2 for ns in depth_2_namespaces)

            # Test pagination
            paginated_namespaces = store.list_namespaces(limit=3)
            assert len(paginated_namespaces) == 3

            # Cleanup
            for namespace in test_namespaces:
                store.delete(namespace, "dummy")

    def test_search(self) -> None:
        with SqliteStore.from_conn_string(":memory:") as store:
            store.setup()
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

            # Test basic search
            all_items = store.search(["test"])
            assert len(all_items) == 3

            # Test namespace filtering
            docs_items = store.search(["test", "docs"])
            assert len(docs_items) == 2
            assert all(item.namespace == ("test", "docs") for item in docs_items)

            # Test value filtering
            alice_items = store.search(["test"], filter={"author": "Alice"})
            assert len(alice_items) == 2
            assert all(item.value["author"] == "Alice" for item in alice_items)

            # Test pagination
            paginated_items = store.search(["test"], limit=2)
            assert len(paginated_items) == 2

            offset_items = store.search(["test"], offset=2)
            assert len(offset_items) == 1

            # Cleanup
            for namespace, key, _ in test_data:
                store.delete(namespace, key)


def test_vector_store_initialization(fake_embeddings: CharacterEmbeddings) -> None:
    """Test store initialization with embedding config."""
    # Basic initialization
    with create_vector_store(fake_embeddings) as store:
        assert store.index_config is not None
        assert store.embeddings == fake_embeddings
        assert store.index_config["dims"] == fake_embeddings.dims
        assert store.index_config.get("text_fields") is None

    # With text fields specified
    text_fields = ["content", "title"]
    with create_vector_store(fake_embeddings, text_fields=text_fields) as store:
        assert store.index_config is not None
        assert store.embeddings == fake_embeddings
        assert store.index_config["dims"] == fake_embeddings.dims
        assert store.index_config["text_fields"] == text_fields

    # Ensure store setup properly creates the vector tables
    with create_vector_store(fake_embeddings) as store:
        # Check if vector tables exist
        cursor = store.conn.cursor()
        cursor.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name LIKE '%vector%'"
        )
        tables = cursor.fetchall()
        assert len(tables) >= 1, "Vector tables were not created"


@pytest.mark.parametrize("distance_type", VECTOR_TYPES)
@pytest.mark.parametrize("conn_type", ["memory", "file"])
def test_vector_insert_with_auto_embedding(
    fake_embeddings: CharacterEmbeddings,
    distance_type: str,
    conn_type: Literal["memory", "file"],
) -> None:
    """Test inserting items that get auto-embedded."""
    with create_vector_store(
        fake_embeddings, distance_type=distance_type, conn_type=conn_type
    ) as store:
        docs = [
            ("doc1", {"text": "short text"}),
            ("doc2", {"text": "longer text document"}),
            ("doc3", {"text": "longest text document here"}),
            ("doc4", {"description": "text in description field"}),
            ("doc5", {"content": "text in content field"}),
            ("doc6", {"body": "text in body field"}),
        ]

        for key, value in docs:
            store.put(("test",), key, value)

        results = store.search(("test",), query="long text")
        assert len(results) > 0

        doc_order = [r.key for r in results]
        assert "doc2" in doc_order
        assert "doc3" in doc_order


@pytest.mark.parametrize("distance_type", VECTOR_TYPES)
@pytest.mark.parametrize("conn_type", ["memory", "file"])
def test_vector_update_with_embedding(
    fake_embeddings: CharacterEmbeddings,
    distance_type: str,
    conn_type: Literal["memory", "file"],
) -> None:
    """Test that updating items properly updates their embeddings."""
    with create_vector_store(
        fake_embeddings, distance_type=distance_type, conn_type=conn_type
    ) as store:
        store.put(("test",), "doc1", {"text": "zany zebra Xerxes"})
        store.put(("test",), "doc2", {"text": "something about dogs"})
        store.put(("test",), "doc3", {"text": "text about birds"})

        results_initial = store.search(("test",), query="Zany Xerxes")
        assert len(results_initial) > 0
        assert results_initial[0].key == "doc1"
        initial_score = results_initial[0].score

        store.put(("test",), "doc1", {"text": "new text about dogs"})

        results_after = store.search(("test",), query="Zany Xerxes")
        after_score = next((r.score for r in results_after if r.key == "doc1"), 0.0)
        assert after_score < initial_score

        results_new = store.search(("test",), query="new text about dogs")
        for r in results_new:
            if r.key == "doc1":
                assert r.score > after_score

        # Don't index this one
        store.put(("test",), "doc4", {"text": "new text about dogs"}, index=False)
        results_new = store.search(("test",), query="new text about dogs", limit=3)
        assert not any(r.key == "doc4" for r in results_new)


@pytest.mark.parametrize("distance_type", VECTOR_TYPES)
def test_vector_search_with_filters(
    fake_embeddings: CharacterEmbeddings,
    distance_type: str,
) -> None:
    """Test combining vector search with filters."""
    with create_vector_store(fake_embeddings, distance_type=distance_type) as store:
        # Insert test documents
        docs = [
            ("doc1", {"text": "red apple", "color": "red", "score": 4.5}),
            ("doc2", {"text": "red car", "color": "red", "score": 3.0}),
            ("doc3", {"text": "green apple", "color": "green", "score": 4.0}),
            ("doc4", {"text": "blue car", "color": "blue", "score": 3.5}),
        ]
        for key, value in docs:
            store.put(("test",), key, value)

        results = store.search(("test",), query="apple", filter={"color": "red"})

        # Check ordering and score - verify "doc1" is first result
        assert len(results) == 2
        assert results[0].key == "doc1"

        results = store.search(("test",), query="car", filter={"color": "red"})
        # Check ordering - verify "doc2" is first result
        assert len(results) > 0
        assert results[0].key == "doc2"

        results = store.search(
            ("test",), query="bbbbluuu", filter={"score": {"$gt": 3.2}}
        )
        # There should be 3 documents with score > 3.2
        assert len(results) == 3
        # Check that the blue car is the most similar to "bbbbluuu" query
        assert results[0].key == "doc4"  # The blue car should be the most relevant
        # Verify remaining docs are ordered by appropriate similarity
        high_score_keys = [r.key for r in results]
        assert "doc1" in high_score_keys  # score 4.5
        assert "doc3" in high_score_keys  # score 4.0

        # Multiple filters
        results = store.search(
            ("test",), query="apple", filter={"score": {"$gte": 4.0}, "color": "green"}
        )
        # Check that doc3 is the top result
        assert len(results) > 0
        assert results[0].key == "doc3"


@pytest.mark.parametrize("distance_type", VECTOR_TYPES)
def test_vector_search_pagination(
    fake_embeddings: CharacterEmbeddings,
    distance_type: str,
) -> None:
    """Test pagination with vector search."""
    with create_vector_store(fake_embeddings, distance_type=distance_type) as store:
        # Insert multiple similar documents
        for i in range(5):
            store.put(("test",), f"doc{i}", {"text": f"test document number {i}"})

        # Test with different page sizes
        results_page1 = store.search(("test",), query="test", limit=2)
        results_page2 = store.search(("test",), query="test", limit=2, offset=2)

        assert len(results_page1) == 2
        assert len(results_page2) == 2
        # Make sure different pages have different results
        assert results_page1[0].key != results_page2[0].key
        assert results_page1[1].key != results_page2[0].key
        assert results_page1[0].key != results_page2[1].key
        assert results_page1[1].key != results_page2[1].key

        # Check scores are in descending order within each page
        assert results_page1[0].score >= results_page1[1].score
        assert results_page2[0].score >= results_page2[1].score

        # First page results should have higher scores than second page
        all_results = store.search(("test",), query="test", limit=10)
        assert len(all_results) == 5
        assert (
            all_results[0].score >= all_results[2].score
        )  # First page vs second page start


@pytest.mark.parametrize("distance_type", VECTOR_TYPES)
def test_vector_search_edge_cases(
    fake_embeddings: CharacterEmbeddings,
    distance_type: str,
) -> None:
    """Test edge cases in vector search."""
    with create_vector_store(fake_embeddings, distance_type=distance_type) as store:
        store.put(("test",), "doc1", {"text": "test document"})

        results = store.search(("test",), query="")
        assert len(results) == 1

        results = store.search(("test",), query=None)
        assert len(results) == 1

        long_query = "test " * 100
        results = store.search(("test",), query=long_query)
        assert len(results) == 1

        special_query = "test!@#$%^&*()"
        results = store.search(("test",), query=special_query)
        assert len(results) == 1


@pytest.mark.parametrize("distance_type", VECTOR_TYPES)
def test_embed_with_path(
    fake_embeddings: CharacterEmbeddings,
    distance_type: str,
) -> None:
    """Test vector search with specific text fields in SQLite store."""
    with create_vector_store(
        fake_embeddings,
        text_fields=["key0", "key1", "key3"],
        distance_type=distance_type,
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
        store.put(("test",), "doc1", doc1)
        store.put(("test",), "doc2", doc2)

        # doc2.key3 and doc1.key1 both would have the highest score
        results = store.search(("test",), query="xxx")
        assert len(results) == 2
        assert results[0].key != results[1].key
        assert results[0].score > 0.9
        assert results[1].score > 0.9

        # ~Only match doc2
        results = store.search(("test",), query="uuu")
        assert len(results) == 2
        assert results[0].key != results[1].key
        assert results[0].key == "doc2"
        assert results[0].score > results[1].score

        # ~Only match doc1
        results = store.search(("test",), query="zzz")
        assert len(results) == 2
        assert results[0].key != results[1].key
        assert results[0].key == "doc1"
        assert results[0].score > results[1].score

        # Un-indexed - will have low results for both, Not zero (because we're projecting)
        # but less than the above.
        results = store.search(("test",), query="www")
        assert len(results) == 2
        assert results[0].key != results[1].key
        assert results[0].score < 0.9
        assert results[1].score < 0.9


@pytest.mark.parametrize("distance_type", VECTOR_TYPES)
def test_embed_with_path_operation_config(
    fake_embeddings: CharacterEmbeddings,
    distance_type: str,
) -> None:
    """Test operation-level field configuration for vector search."""
    with create_vector_store(
        fake_embeddings, text_fields=["key17"], distance_type=distance_type
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

        results = store.search(("test",), query="aaa")
        assert len(results) == 2
        assert results[0].key == "doc3"
        assert len(set(r.key for r in results)) == 2
        assert results[0].score > results[1].score

        results = store.search(("test",), query="ggg")
        assert len(results) == 2
        assert results[0].key == "doc4"
        assert results[0].score > results[1].score

        results = store.search(("test",), query="bbb")
        assert len(results) == 2
        assert results[0].key != results[1].key
        assert abs(results[0].score - results[1].score) < 0.1  # Similar scores

        results = store.search(("test",), query="ccc")
        assert len(results) == 2
        assert all(
            r.score < 0.9 for r in results
        )  # Unindexed field should have low scores

        # Test index=False behavior
        doc5 = {
            "key0": "hhh",
            "key1": "iii",
        }
        store.put(("test",), "doc5", doc5, index=False)
        results = store.search(("test",))
        assert len(results) == 3
        assert any(r.key == "doc5" for r in results)


# Helper functions for vector similarity calculations
def _cosine_similarity(X: list[float], Y: list[list[float]]) -> list[float]:
    """
    Compute cosine similarity between a vector X and a matrix Y.
    Lazy import numpy for efficiency.
    """

    similarities = []
    for y in Y:
        dot_product = sum(a * b for a, b in zip(X, y, strict=False))
        norm1 = sum(a * a for a in X) ** 0.5
        norm2 = sum(a * a for a in y) ** 0.5
        similarity = dot_product / (norm1 * norm2) if norm1 > 0 and norm2 > 0 else 0.0
        similarities.append(similarity)

    return similarities


@pytest.mark.parametrize("query", ["aaa", "bbb", "ccc", "abcd", "poisson"])
@pytest.mark.parametrize("conn_type", ["memory", "file"])
def test_scores(
    fake_embeddings: CharacterEmbeddings,
    query: str,
    conn_type: Literal["memory", "file"],
) -> None:
    """Test operation-level field configuration for vector search."""
    with create_vector_store(
        fake_embeddings,
        text_fields=["key0"],
        distance_type="cosine",
        conn_type=conn_type,
    ) as store:
        doc = {
            "key0": "aaa",
        }
        store.put(("test",), "doc", doc, index=["key0", "key1"])

        results = store.search((), query=query)
        vec0 = fake_embeddings.embed_query(doc["key0"])
        vec1 = fake_embeddings.embed_query(query)

        # SQLite uses cosine similarity by default
        similarities = _cosine_similarity(vec1, [vec0])

        assert len(results) == 1
        assert results[0].score == pytest.approx(similarities[0], abs=1e-3)


def test_nonnull_migrations() -> None:
    """Test that all migration statements are non-null."""
    _leading_comment_remover = re.compile(r"^/\*.*?\*/")
    for migration in SqliteStore.MIGRATIONS:
        statement = _leading_comment_remover.sub("", migration).split()[0]
        assert statement.strip(), f"Empty migration statement found: {migration}"


def test_basic_store_operations(
    fake_embeddings: CharacterEmbeddings,
) -> None:
    """Test basic store operations with SQLite store."""
    with create_vector_store(
        fake_embeddings, text_fields=["key0", "key1", "key3"]
    ) as store:
        uid = uuid.uuid4().hex
        namespace = (uid, "test", "documents")
        item_id = "doc1"
        item_value = {"title": "Test Document", "content": "Hello, World!"}
        results = store.search((uid,))
        assert len(results) == 0

        store.put(namespace, item_id, item_value)
        item = store.get(namespace, item_id)

        assert item is not None
        assert item.namespace == namespace
        assert item.key == item_id
        assert item.value == item_value
        assert item.created_at is not None
        assert item.updated_at is not None

        updated_value = {
            "title": "Updated Test Document",
            "content": "Hello, LangGraph!",
        }
        store.put(namespace, item_id, updated_value)
        updated_item = store.get(namespace, item_id)
        assert updated_item is not None

        assert updated_item.value == updated_value
        assert updated_item.updated_at >= item.updated_at

        different_namespace = (uid, "test", "other_documents")
        item_in_different_namespace = store.get(different_namespace, item_id)
        assert item_in_different_namespace is None

        new_item_id = "doc2"
        new_item_value = {"title": "Another Document", "content": "Greetings!"}
        store.put(namespace, new_item_id, new_item_value)

        items = store.search((uid, "test"), limit=10)
        assert len(items) == 2
        assert any(item.key == item_id for item in items)
        assert any(item.key == new_item_id for item in items)

        namespaces = store.list_namespaces(prefix=(uid, "test"))
        assert (uid, "test", "documents") in namespaces

        store.delete(namespace, item_id)
        store.delete(namespace, new_item_id)
        deleted_item = store.get(namespace, item_id)
        assert deleted_item is None

        deleted_item = store.get(namespace, new_item_id)
        assert deleted_item is None

        empty_search_results = store.search((uid, "test"), limit=10)
        assert len(empty_search_results) == 0


def test_list_namespaces_operations(
    fake_embeddings: CharacterEmbeddings,
) -> None:
    """Test list namespaces functionality with various filters."""
    with create_vector_store(
        fake_embeddings, text_fields=["key0", "key1", "key3"]
    ) as store:
        test_pref = str(uuid.uuid4())
        test_namespaces = [
            (test_pref, "test", "documents", "public", test_pref),
            (test_pref, "test", "documents", "private", test_pref),
            (test_pref, "test", "images", "public", test_pref),
            (test_pref, "test", "images", "private", test_pref),
            (test_pref, "prod", "documents", "public", test_pref),
            (test_pref, "prod", "documents", "some", "nesting", "public", test_pref),
            (test_pref, "prod", "documents", "private", test_pref),
        ]

        # Add test data
        for namespace in test_namespaces:
            store.put(namespace, "dummy", {"content": "dummy"})

        # Test prefix filtering
        prefix_result = store.list_namespaces(prefix=(test_pref, "test"))
        assert len(prefix_result) == 4
        assert all(ns[1] == "test" for ns in prefix_result)

        # Test specific prefix
        specific_prefix_result = store.list_namespaces(
            prefix=(test_pref, "test", "documents")
        )
        assert len(specific_prefix_result) == 2
        assert all(ns[1:3] == ("test", "documents") for ns in specific_prefix_result)

        # Test suffix filtering
        suffix_result = store.list_namespaces(suffix=("public", test_pref))
        assert len(suffix_result) == 4
        assert all(ns[-2] == "public" for ns in suffix_result)

        # Test combined prefix and suffix
        prefix_suffix_result = store.list_namespaces(
            prefix=(test_pref, "test"), suffix=("public", test_pref)
        )
        assert len(prefix_suffix_result) == 2
        assert all(
            ns[1] == "test" and ns[-2] == "public" for ns in prefix_suffix_result
        )

        # Test wildcard in prefix
        wildcard_prefix_result = store.list_namespaces(
            prefix=(test_pref, "*", "documents")
        )
        assert len(wildcard_prefix_result) == 5
        assert all(ns[2] == "documents" for ns in wildcard_prefix_result)

        # Test wildcard in suffix
        wildcard_suffix_result = store.list_namespaces(
            suffix=("*", "public", test_pref)
        )
        assert len(wildcard_suffix_result) == 4
        assert all(ns[-2] == "public" for ns in wildcard_suffix_result)

        wildcard_single = store.list_namespaces(
            suffix=("some", "*", "public", test_pref)
        )
        assert len(wildcard_single) == 1
        assert wildcard_single[0] == (
            test_pref,
            "prod",
            "documents",
            "some",
            "nesting",
            "public",
            test_pref,
        )

        # Test max depth
        max_depth_result = store.list_namespaces(max_depth=3)
        assert all(len(ns) <= 3 for ns in max_depth_result)

        max_depth_result = store.list_namespaces(
            max_depth=4, prefix=(test_pref, "*", "documents")
        )
        assert len(set(res for res in max_depth_result)) == len(max_depth_result) == 5

        # Test pagination
        limit_result = store.list_namespaces(prefix=(test_pref,), limit=3)
        assert len(limit_result) == 3

        offset_result = store.list_namespaces(prefix=(test_pref,), offset=3)
        assert len(offset_result) == len(test_namespaces) - 3

        empty_prefix_result = store.list_namespaces(prefix=(test_pref,))
        assert len(empty_prefix_result) == len(test_namespaces)
        assert set(empty_prefix_result) == set(test_namespaces)

        # Clean up
        for namespace in test_namespaces:
            store.delete(namespace, "dummy")


def test_search_items(
    fake_embeddings: CharacterEmbeddings,
) -> None:
    """Test search_items functionality by calling store methods directly."""
    base = "test_search_items"
    test_namespaces = [
        (base, "documents", "user1"),
        (base, "documents", "user2"),
        (base, "reports", "department1"),
        (base, "reports", "department2"),
    ]
    test_items = [
        {"title": "Doc 1", "author": "John Doe", "tags": ["important"]},
        {"title": "Doc 2", "author": "Jane Smith", "tags": ["draft"]},
        {"title": "Report A", "author": "John Doe", "tags": ["final"]},
        {"title": "Report B", "author": "Alice Johnson", "tags": ["draft"]},
    ]

    with create_vector_store(
        fake_embeddings, text_fields=["key0", "key1", "key3"]
    ) as store:
        # Insert test data
        for ns, item in zip(test_namespaces, test_items, strict=False):
            key = f"item_{ns[-1]}"
            store.put(ns, key, item)

        # 1. Search documents
        docs = store.search((base, "documents"))
        assert len(docs) == 2
        assert all(item.namespace[1] == "documents" for item in docs)

        # 2. Search reports
        reports = store.search((base, "reports"))
        assert len(reports) == 2
        assert all(item.namespace[1] == "reports" for item in reports)

        # 3. Pagination
        first_page = store.search((base,), limit=2, offset=0)
        second_page = store.search((base,), limit=2, offset=2)
        assert len(first_page) == 2
        assert len(second_page) == 2
        keys_page1 = {item.key for item in first_page}
        keys_page2 = {item.key for item in second_page}
        assert keys_page1.isdisjoint(keys_page2)
        all_items = store.search((base,))
        assert len(all_items) == 4

        john_items = store.search((base,), filter={"author": "John Doe"})
        assert len(john_items) == 2
        assert all(item.value["author"] == "John Doe" for item in john_items)

        draft_items = store.search((base,), filter={"tags": ["draft"]})
        assert len(draft_items) == 2
        assert all("draft" in item.value["tags"] for item in draft_items)

        for ns in test_namespaces:
            key = f"item_{ns[-1]}"
            store.delete(ns, key)


def test_sql_injection_vulnerability(store: SqliteStore) -> None:
    """Test that SQL injection via malicious filter keys is prevented."""
    # Add public and private documents
    store.put(("docs",), "public", {"access": "public", "data": "public info"})
    store.put(
        ("docs",), "private", {"access": "private", "data": "secret", "password": "123"}
    )

    # Normal query - returns 1 public document
    normal = store.search(("docs",), filter={"access": "public"})
    assert len(normal) == 1
    assert normal[0].value["access"] == "public"

    # SQL injection attempt via malicious key should raise ValueError
    malicious_key = "access') = 'public' OR '1'='1' OR json_extract(value, '$."

    with pytest.raises(ValueError, match="Invalid filter key"):
        store.search(("docs",), filter={malicious_key: "dummy"})


@pytest.mark.parametrize("distance_type", VECTOR_TYPES)
def test_non_ascii(
    fake_embeddings: CharacterEmbeddings,
    distance_type: str,
) -> None:
    """Test support for non-ascii characters"""
    with create_vector_store(fake_embeddings, distance_type=distance_type) as store:
        store.put(("user_123", "memories"), "1", {"text": "这是中文"})  # Chinese
        store.put(
            ("user_123", "memories"), "2", {"text": "これは日本語です"}
        )  # Japanese
        store.put(("user_123", "memories"), "3", {"text": "이건 한국어야"})  # Korean
        store.put(("user_123", "memories"), "4", {"text": "Это русский"})  # Russian
        store.put(("user_123", "memories"), "5", {"text": "यह रूसी है"})  # Hindi

        result1 = store.search(("user_123", "memories"), query="这是中文")
        result2 = store.search(("user_123", "memories"), query="これは日本語です")
        result3 = store.search(("user_123", "memories"), query="이건 한국어야")
        result4 = store.search(("user_123", "memories"), query="Это русский")
        result5 = store.search(("user_123", "memories"), query="यह रूसी है")

        assert result1[0].key == "1"
        assert result2[0].key == "2"
        assert result3[0].key == "3"
        assert result4[0].key == "4"
        assert result5[0].key == "5"
