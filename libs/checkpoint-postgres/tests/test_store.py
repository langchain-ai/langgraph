# type: ignore

from uuid import uuid4

import pytest
from conftest import DEFAULT_URI  # type: ignore
from psycopg import Connection
from psycopg_pool import ConnectionPool

from langgraph.store.base import (
    GetOp,
    Item,
    ListNamespacesOp,
    MatchCondition,
    PutOp,
    SearchOp,
)
from langgraph.store.postgres import PostgresStore


@pytest.fixture(scope="function", params=["default", "pipe", "pool"])
def store(request) -> PostgresStore:
    database = f"test_{uuid4().hex[:16]}"
    uri_parts = DEFAULT_URI.split("/")
    uri_base = "/".join(uri_parts[:-1])
    query_params = ""
    if "?" in uri_parts[-1]:
        db_name, query_params = uri_parts[-1].split("?", 1)
        query_params = "?" + query_params

    conn_string = f"{uri_base}/{database}{query_params}"
    admin_conn_string = DEFAULT_URI

    with Connection.connect(admin_conn_string, autocommit=True) as conn:
        conn.execute(f"CREATE DATABASE {database}")
    try:
        with PostgresStore.from_conn_string(conn_string) as store:
            store.setup()

        if request.param == "pipe":
            with PostgresStore.from_conn_string(conn_string, pipeline=True) as store:
                yield store
        elif request.param == "pool":
            with ConnectionPool(
                conn_string, max_size=10, kwargs={"autocommit": True}
            ) as pool:
                store = PostgresStore(pool)
                yield store
        else:  # default
            with PostgresStore.from_conn_string(conn_string) as store:
                yield store
    finally:
        with Connection.connect(admin_conn_string, autocommit=True) as conn:
            conn.execute(f"DROP DATABASE {database}")


def test_batch_order(store: PostgresStore) -> None:
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
    assert len(results[3]) > 0  # Should contain at least our test namespaces
    assert results[4] is None  # Non-existent key returns None

    # Test reordered operations
    ops_reordered = [
        SearchOp(namespace_prefix=("test",), filter=None, limit=5, offset=0),
        GetOp(namespace=("test", "bar"), key="key2"),
        ListNamespacesOp(match_conditions=None, max_depth=None, limit=5, offset=0),
        PutOp(namespace=("test",), key="key3", value={"data": "value3"}),
        GetOp(namespace=("test", "foo"), key="key1"),
    ]

    results_reordered = store.batch(ops_reordered)
    assert len(results_reordered) == 5
    assert isinstance(results_reordered[0], list)
    assert len(results_reordered[0]) >= 2  # Should find at least our two test items
    assert isinstance(results_reordered[1], Item)
    assert results_reordered[1].value == {"data": "value2"}
    assert results_reordered[1].key == "key2"
    assert isinstance(results_reordered[2], list)
    assert len(results_reordered[2]) > 0
    assert results_reordered[3] is None  # Put operation returns None
    assert isinstance(results_reordered[4], Item)
    assert results_reordered[4].value == {"data": "value1"}
    assert results_reordered[4].key == "key1"


def test_batch_get_ops(store: PostgresStore) -> None:
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


def test_batch_put_ops(store: PostgresStore) -> None:
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


def test_batch_search_ops(store: PostgresStore) -> None:
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


def test_batch_list_namespaces_ops(store: PostgresStore) -> None:
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
            match_conditions=[MatchCondition("suffix", "public")],
            max_depth=None,
            limit=10,
            offset=0,
        ),
    ]

    results = store.batch(ops)
    assert len(results) == 3

    # First operation should list all namespaces
    assert len(results[0]) == len(test_data)

    # Second operation should only return namespaces up to depth 2
    assert all(len(ns) <= 2 for ns in results[1])

    # Third operation should only return namespaces ending with "public"
    assert all(ns[-1] == "public" for ns in results[2])


class TestPostgresStore:
    @pytest.fixture(autouse=True)
    def setup(self) -> None:
        with PostgresStore.from_conn_string(DEFAULT_URI) as store:
            store.setup()

    def test_basic_store_ops(self) -> None:
        with PostgresStore.from_conn_string(DEFAULT_URI) as store:
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

    def test_list_namespaces(self) -> None:
        with PostgresStore.from_conn_string(DEFAULT_URI) as store:
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
        with PostgresStore.from_conn_string(DEFAULT_URI) as store:
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
