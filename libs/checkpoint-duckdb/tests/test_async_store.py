# type: ignore
import uuid
from datetime import datetime
from typing import Any
from unittest.mock import MagicMock

import pytest

from langgraph.store.base import GetOp, Item, ListNamespacesOp, PutOp, SearchOp
from langgraph.store.duckdb import AsyncDuckDBStore


class MockCursor:
    def __init__(self, fetch_result: Any) -> None:
        self.fetch_result = fetch_result
        self.execute = MagicMock()
        self.fetchall = MagicMock(return_value=self.fetch_result)


class MockConnection:
    def __init__(self) -> None:
        self.cursor = MagicMock()


@pytest.fixture
def mock_connection() -> MockConnection:
    return MockConnection()


@pytest.fixture
async def store(mock_connection: MockConnection) -> AsyncDuckDBStore:
    duck_db_store = AsyncDuckDBStore(mock_connection)
    await duck_db_store.setup()
    return duck_db_store


async def test_abatch_order(store: AsyncDuckDBStore) -> None:
    mock_connection = store.conn
    mock_get_cursor = MockCursor(
        [
            (
                "test.foo",
                "key1",
                '{"data": "value1"}',
                datetime.now(),
                datetime.now(),
            ),
            (
                "test.bar",
                "key2",
                '{"data": "value2"}',
                datetime.now(),
                datetime.now(),
            ),
        ]
    )
    mock_search_cursor = MockCursor(
        [
            (
                "test.foo",
                "key1",
                '{"data": "value1"}',
                datetime.now(),
                datetime.now(),
            ),
        ]
    )
    mock_list_namespaces_cursor = MockCursor(
        [
            ("test",),
        ]
    )

    failures = []

    def cursor_side_effect() -> Any:
        cursor = MagicMock()

        def execute_side_effect(query: str, *params: Any) -> None:
            # My super sophisticated database.
            if "WHERE prefix = ? AND key" in query:
                cursor.fetchall = mock_get_cursor.fetchall
            elif "SELECT prefix, key, value" in query:
                cursor.fetchall = mock_search_cursor.fetchall
            elif "SELECT DISTINCT ON (truncated_prefix)" in query:
                cursor.fetchall = mock_list_namespaces_cursor.fetchall
            elif "INSERT INTO " in query:
                pass
            else:
                e = ValueError(f"Unmatched query: {query}")
                failures.append(e)
                raise e

        cursor.execute = MagicMock(side_effect=execute_side_effect)
        return cursor

    mock_connection.cursor.side_effect = cursor_side_effect  # type: ignore

    ops = [
        GetOp(namespace=("test",), key="key1"),
        PutOp(namespace=("test",), key="key2", value={"data": "value2"}),
        SearchOp(
            namespace_prefix=("test",), filter={"data": "value1"}, limit=10, offset=0
        ),
        ListNamespacesOp(match_conditions=None, max_depth=None, limit=10, offset=0),
        GetOp(namespace=("test",), key="key3"),
    ]
    results = await store.abatch(ops)
    assert not failures
    assert len(results) == 5
    assert isinstance(results[0], Item)
    assert isinstance(results[0].value, dict)
    assert results[0].value == {"data": "value1"}
    assert results[0].key == "key1"
    assert results[1] is None
    assert isinstance(results[2], list)
    assert len(results[2]) == 1
    assert isinstance(results[3], list)
    assert results[3] == [("test",)]
    assert results[4] is None

    ops_reordered = [
        SearchOp(namespace_prefix=("test",), filter=None, limit=5, offset=0),
        GetOp(namespace=("test",), key="key2"),
        ListNamespacesOp(match_conditions=None, max_depth=None, limit=5, offset=0),
        PutOp(namespace=("test",), key="key3", value={"data": "value3"}),
        GetOp(namespace=("test",), key="key1"),
    ]

    results_reordered = await store.abatch(ops_reordered)
    assert not failures
    assert len(results_reordered) == 5
    assert isinstance(results_reordered[0], list)
    assert len(results_reordered[0]) == 1
    assert isinstance(results_reordered[1], Item)
    assert results_reordered[1].value == {"data": "value2"}
    assert results_reordered[1].key == "key2"
    assert isinstance(results_reordered[2], list)
    assert results_reordered[2] == [("test",)]
    assert results_reordered[3] is None
    assert isinstance(results_reordered[4], Item)
    assert results_reordered[4].value == {"data": "value1"}
    assert results_reordered[4].key == "key1"


async def test_batch_get_ops(store: AsyncDuckDBStore) -> None:
    mock_connection = store.conn
    mock_cursor = MockCursor(
        [
            (
                "test.foo",
                "key1",
                '{"data": "value1"}',
                datetime.now(),
                datetime.now(),
            ),
            (
                "test.bar",
                "key2",
                '{"data": "value2"}',
                datetime.now(),
                datetime.now(),
            ),
        ]
    )
    mock_connection.cursor.return_value = mock_cursor

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


async def test_batch_put_ops(store: AsyncDuckDBStore) -> None:
    mock_connection = store.conn
    mock_cursor = MockCursor([])
    mock_connection.cursor.return_value = mock_cursor

    ops = [
        PutOp(namespace=("test",), key="key1", value={"data": "value1"}),
        PutOp(namespace=("test",), key="key2", value={"data": "value2"}),
        PutOp(namespace=("test",), key="key3", value=None),
    ]

    results = await store.abatch(ops)

    assert len(results) == 3
    assert all(result is None for result in results)
    assert mock_cursor.execute.call_count == 2


async def test_batch_search_ops(store: AsyncDuckDBStore) -> None:
    mock_connection = store.conn
    mock_cursor = MockCursor(
        [
            (
                "test.foo",
                "key1",
                '{"data": "value1"}',
                datetime.now(),
                datetime.now(),
            ),
            (
                "test.bar",
                "key2",
                '{"data": "value2"}',
                datetime.now(),
                datetime.now(),
            ),
        ]
    )
    mock_connection.cursor.return_value = mock_cursor

    ops = [
        SearchOp(
            namespace_prefix=("test",), filter={"data": "value1"}, limit=10, offset=0
        ),
        SearchOp(namespace_prefix=("test",), filter=None, limit=5, offset=0),
    ]

    results = await store.abatch(ops)

    assert len(results) == 2
    assert len(results[0]) == 2
    assert len(results[1]) == 2


async def test_batch_list_namespaces_ops(store: AsyncDuckDBStore) -> None:
    mock_connection = store.conn
    mock_cursor = MockCursor([("test.namespace1",), ("test.namespace2",)])
    mock_connection.cursor.return_value = mock_cursor

    ops = [ListNamespacesOp(match_conditions=None, max_depth=None, limit=10, offset=0)]

    results = await store.abatch(ops)

    assert len(results) == 1
    assert results[0] == [("test", "namespace1"), ("test", "namespace2")]


# The following use the actual DB connection


async def test_basic_store_ops() -> None:
    async with AsyncDuckDBStore.from_conn_string(":memory:") as store:
        await store.setup()
        namespace = ("test", "documents")
        item_id = "doc1"
        item_value = {"title": "Test Document", "content": "Hello, World!"}

        await store.aput(namespace, item_id, item_value)
        item = await store.aget(namespace, item_id)

        assert item
        assert item.namespace == namespace
        assert item.key == item_id
        assert item.value == item_value

        updated_value = {
            "title": "Updated Test Document",
            "content": "Hello, LangGraph!",
        }
        await store.aput(namespace, item_id, updated_value)
        updated_item = await store.aget(namespace, item_id)

        assert updated_item.value == updated_value
        assert updated_item.updated_at > item.updated_at
        different_namespace = ("test", "other_documents")
        item_in_different_namespace = await store.aget(different_namespace, item_id)
        assert item_in_different_namespace is None

        new_item_id = "doc2"
        new_item_value = {"title": "Another Document", "content": "Greetings!"}
        await store.aput(namespace, new_item_id, new_item_value)

        search_results = await store.asearch(["test"], limit=10)
        items = search_results
        assert len(items) == 2
        assert any(item.key == item_id for item in items)
        assert any(item.key == new_item_id for item in items)

        namespaces = await store.alist_namespaces(prefix=["test"])
        assert ("test", "documents") in namespaces

        await store.adelete(namespace, item_id)
        await store.adelete(namespace, new_item_id)
        deleted_item = await store.aget(namespace, item_id)
        assert deleted_item is None

        deleted_item = await store.aget(namespace, new_item_id)
        assert deleted_item is None

        empty_search_results = await store.asearch(["test"], limit=10)
        assert len(empty_search_results) == 0


async def test_list_namespaces() -> None:
    async with AsyncDuckDBStore.from_conn_string(":memory:") as store:
        await store.setup()
        test_pref = str(uuid.uuid4())
        test_namespaces = [
            (test_pref, "test", "documents", "public", test_pref),
            (test_pref, "test", "documents", "private", test_pref),
            (test_pref, "test", "images", "public", test_pref),
            (test_pref, "test", "images", "private", test_pref),
            (test_pref, "prod", "documents", "public", test_pref),
            (
                test_pref,
                "prod",
                "documents",
                "some",
                "nesting",
                "public",
                test_pref,
            ),
            (test_pref, "prod", "documents", "private", test_pref),
        ]

        for namespace in test_namespaces:
            await store.aput(namespace, "dummy", {"content": "dummy"})

        prefix_result = await store.alist_namespaces(prefix=[test_pref, "test"])
        assert len(prefix_result) == 4
        assert all([ns[1] == "test" for ns in prefix_result])

        specific_prefix_result = await store.alist_namespaces(
            prefix=[test_pref, "test", "documents"]
        )
        assert len(specific_prefix_result) == 2
        assert all([ns[1:3] == ("test", "documents") for ns in specific_prefix_result])

        suffix_result = await store.alist_namespaces(suffix=["public", test_pref])
        assert len(suffix_result) == 4
        assert all(ns[-2] == "public" for ns in suffix_result)

        prefix_suffix_result = await store.alist_namespaces(
            prefix=[test_pref, "test"], suffix=["public", test_pref]
        )
        assert len(prefix_suffix_result) == 2
        assert all(
            ns[1] == "test" and ns[-2] == "public" for ns in prefix_suffix_result
        )

        wildcard_prefix_result = await store.alist_namespaces(
            prefix=[test_pref, "*", "documents"]
        )
        assert len(wildcard_prefix_result) == 5
        assert all(ns[2] == "documents" for ns in wildcard_prefix_result)

        wildcard_suffix_result = await store.alist_namespaces(
            suffix=["*", "public", test_pref]
        )
        assert len(wildcard_suffix_result) == 4
        assert all(ns[-2] == "public" for ns in wildcard_suffix_result)
        wildcard_single = await store.alist_namespaces(
            suffix=["some", "*", "public", test_pref]
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

        max_depth_result = await store.alist_namespaces(max_depth=3)
        assert all([len(ns) <= 3 for ns in max_depth_result])
        max_depth_result = await store.alist_namespaces(
            max_depth=4, prefix=[test_pref, "*", "documents"]
        )
        assert (
            len(set(tuple(res) for res in max_depth_result))
            == len(max_depth_result)
            == 5
        )

        limit_result = await store.alist_namespaces(prefix=[test_pref], limit=3)
        assert len(limit_result) == 3

        offset_result = await store.alist_namespaces(prefix=[test_pref], offset=3)
        assert len(offset_result) == len(test_namespaces) - 3

        empty_prefix_result = await store.alist_namespaces(prefix=[test_pref])
        assert len(empty_prefix_result) == len(test_namespaces)
        assert set(tuple(ns) for ns in empty_prefix_result) == set(
            tuple(ns) for ns in test_namespaces
        )

        for namespace in test_namespaces:
            await store.adelete(namespace, "dummy")


async def test_search():
    async with AsyncDuckDBStore.from_conn_string(":memory:") as store:
        await store.setup()
        test_namespaces = [
            ("test_search", "documents", "user1"),
            ("test_search", "documents", "user2"),
            ("test_search", "reports", "department1"),
            ("test_search", "reports", "department2"),
        ]
        test_items = [
            {"title": "Doc 1", "author": "John Doe", "tags": ["important"]},
            {"title": "Doc 2", "author": "Jane Smith", "tags": ["draft"]},
            {"title": "Report A", "author": "John Doe", "tags": ["final"]},
            {"title": "Report B", "author": "Alice Johnson", "tags": ["draft"]},
        ]
        empty = await store.asearch(
            (
                "scoped",
                "assistant_id",
                "shared",
                "6c5356f6-63ab-4158-868d-cd9fd14c736e",
            ),
            limit=10,
            offset=0,
        )
        assert len(empty) == 0

        for namespace, item in zip(test_namespaces, test_items):
            await store.aput(namespace, f"item_{namespace[-1]}", item)

        docs_result = await store.asearch(["test_search", "documents"])
        assert len(docs_result) == 2
        assert all([item.namespace[1] == "documents" for item in docs_result]), [
            item.namespace for item in docs_result
        ]

        reports_result = await store.asearch(["test_search", "reports"])
        assert len(reports_result) == 2
        assert all(item.namespace[1] == "reports" for item in reports_result)

        limited_result = await store.asearch(["test_search"], limit=2)
        assert len(limited_result) == 2
        offset_result = await store.asearch(["test_search"])
        assert len(offset_result) == 4

        offset_result = await store.asearch(["test_search"], offset=2)
        assert len(offset_result) == 2
        assert all(item not in limited_result for item in offset_result)

        john_doe_result = await store.asearch(
            ["test_search"], filter={"author": "John Doe"}
        )
        assert len(john_doe_result) == 2
        assert all(item.value["author"] == "John Doe" for item in john_doe_result)

        draft_result = await store.asearch(["test_search"], filter={"tags": ["draft"]})
        assert len(draft_result) == 2
        assert all("draft" in item.value["tags"] for item in draft_result)

        page1 = await store.asearch(["test_search"], limit=2, offset=0)
        page2 = await store.asearch(["test_search"], limit=2, offset=2)
        all_items = page1 + page2
        assert len(all_items) == 4
        assert len(set(item.key for item in all_items)) == 4
        empty = await store.asearch(
            (
                "scoped",
                "assistant_id",
                "shared",
                "again",
                "maybe",
                "some-long",
                "6be5cb0e-2eb4-42e6-bb6b-fba3c269db25",
            ),
            limit=10,
            offset=0,
        )
        assert len(empty) == 0

        # Test with a namespace beginning with a number (like a UUID)
        uuid_namespace = (str(uuid.uuid4()), "documents")
        uuid_item_id = "uuid_doc"
        uuid_item_value = {
            "title": "UUID Document",
            "content": "This document has a UUID namespace.",
        }

        # Insert the item with the UUID namespace
        await store.aput(uuid_namespace, uuid_item_id, uuid_item_value)

        # Retrieve the item to verify it was stored correctly
        retrieved_item = await store.aget(uuid_namespace, uuid_item_id)
        assert retrieved_item is not None
        assert retrieved_item.namespace == uuid_namespace
        assert retrieved_item.key == uuid_item_id
        assert retrieved_item.value == uuid_item_value

        # Search for the item using the UUID namespace
        search_result = await store.asearch([uuid_namespace[0]])
        assert len(search_result) == 1
        assert search_result[0].key == uuid_item_id
        assert search_result[0].value == uuid_item_value

        # Clean up: delete the item with the UUID namespace
        await store.adelete(uuid_namespace, uuid_item_id)

        # Verify the item was deleted
        deleted_item = await store.aget(uuid_namespace, uuid_item_id)
        assert deleted_item is None

        for namespace in test_namespaces:
            await store.adelete(namespace, f"item_{namespace[-1]}")
