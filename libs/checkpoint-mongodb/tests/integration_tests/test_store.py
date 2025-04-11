import os
from datetime import datetime
from typing import Generator

import pytest
from pymongo import MongoClient

from langgraph.store.base import Item
from langgraph.store.mongodb import MongoDBStore

MONGODB_URI = os.environ.get("MONGODB_URI", "mongodb://localhost:27017")
DB_NAME = os.environ.get("DB_NAME", "langgraph-test")
COLLECTION_NAME = "long_term_memory"


t0 = (datetime(2025, 4, 7, 17, 29, 10, 0),)


@pytest.fixture
def ttl():
    return 3600


@pytest.fixture
def store(ttl) -> Generator:
    """Create a simple store following that in base's test_list_namespaces_basic"""
    client = MongoClient(MONGODB_URI)
    collection = client[DB_NAME][COLLECTION_NAME]
    collection.delete_many({})
    collection.drop_indexes()

    mdbstore = MongoDBStore(
        collection,
        ttl=ttl,
    )

    namespaces = [
        ("a", "b", "c"),
        ("a", "b", "d", "e"),
        ("a", "b", "d", "i"),
        ("a", "b", "f"),
        ("a", "c", "f"),
        ("b", "a", "f"),
        ("users", "123"),
        ("users", "456", "settings"),
        ("admin", "users", "789"),
    ]
    for i, ns in enumerate(namespaces):
        mdbstore.put(namespace=ns, key=f"id_{i}", value={"data": f"value_{i:02d}"})

    yield mdbstore

    if client:
        client.close()


def test_list_namespaces():
    client = MongoClient(MONGODB_URI)
    collection = client[DB_NAME][COLLECTION_NAME]
    collection.delete_many({})
    collection.drop_indexes()

    with MongoDBStore.from_conn_string(
        conn_string=MONGODB_URI,
        db_name=DB_NAME,
        collection_name=COLLECTION_NAME,
        ttl=3600,  # 1 hour
    ) as store:
        # Following the data structure of test_list_namespaces_basic
        namespaces = [
            ("a", "b", "c"),
            ("a", "b", "d", "e"),
            ("a", "b", "d", "i"),
            ("a", "b", "f"),
            ("a", "c", "f"),
            ("b", "a", "f"),
            ("users", "123"),
            ("users", "456", "settings"),
            ("admin", "users", "789"),
        ]

        for i, ns in enumerate(namespaces):
            store.put(namespace=ns, key=f"id_{i}", value={"data": f"value_{i:02d}"})

        result = store.list_namespaces(prefix=("a", "b"))
        expected = [
            ("a", "b", "c"),
            ("a", "b", "d", "e"),
            ("a", "b", "d", "i"),
            ("a", "b", "f"),
        ]
        assert sorted(result) == sorted(expected)

        result = store.list_namespaces(suffix=("f",))
        expected = [
            ("a", "b", "f"),
            ("a", "c", "f"),
            ("b", "a", "f"),
        ]
        assert sorted(result) == sorted(expected)

        result = store.list_namespaces(prefix=("a",), suffix=("f",))
        expected = [
            ("a", "b", "f"),
            ("a", "c", "f"),
        ]
        assert sorted(result) == sorted(expected)

        result = store.list_namespaces(
            prefix=("a",),
            suffix=(
                "b",
                "f",
            ),
        )
        expected = [("a", "b", "f")]
        assert sorted(result) == sorted(expected)

        # Test max_depth and deduplication
        result = store.list_namespaces(prefix=("a", "b"), max_depth=3)
        expected = [
            ("a", "b", "c"),
            ("a", "b", "d"),
            ("a", "b", "f"),
        ]
        assert sorted(result) == sorted(expected)

        result = store.list_namespaces(prefix=("a", "*", "f"))
        expected = [
            ("a", "b", "f"),
            ("a", "c", "f"),
        ]
        assert sorted(result) == sorted(expected)

        result = store.list_namespaces(prefix=("*", "*", "f"))
        expected = [("a", "c", "f"), ("b", "a", "f"), ("a", "b", "f")]
        assert sorted(result) == sorted(expected)

        result = store.list_namespaces(suffix=("*", "f"))
        expected = [
            ("a", "b", "f"),
            ("a", "c", "f"),
            ("b", "a", "f"),
        ]
        assert sorted(result) == sorted(expected)

        result = store.list_namespaces(prefix=("a", "b"), suffix=("d", "i"))
        expected = [("a", "b", "d", "i")]
        assert sorted(result) == sorted(expected)

        result = store.list_namespaces(prefix=("a", "b"), suffix=("i"))
        expected = [("a", "b", "d", "i")]
        assert sorted(result) == sorted(expected)

        result = store.list_namespaces(prefix=("nonexistent",))
        assert result == []

        result = store.list_namespaces()
        assert len(result) == store._collection.count_documents({})


def test_get(store: MongoDBStore):
    result = store.get(namespace=("a", "b", "d", "i"), key="id_2")
    assert isinstance(result, Item)
    assert result.updated_at > result.created_at
    assert result.value == {"data": f"value_{2:02d}"}

    result = store.get(namespace=("a", "b", "d", "i"), key="id-2")
    assert result is None

    result = store.get(namespace=tuple(), key="id_2")
    assert result is None

    result = store.get(namespace=("a", "b", "d", "i"), key="")
    assert result is None

    # Test case: refresh_ttl is False
    expected_updated_at = store._collection.find_one(
        dict(namespace=["a", "b", "d", "i"], key="id_2")
    )["updated_at"]

    result = store.get(namespace=("a", "b", "d", "i"), key="id_2", refresh_ttl=False)
    assert result.updated_at == expected_updated_at
