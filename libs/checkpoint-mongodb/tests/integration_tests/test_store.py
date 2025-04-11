import os
from datetime import datetime
from typing import Generator

import pytest
from pymongo import MongoClient

from langgraph.store.base import Item, TTLConfig
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
        ttl_config=TTLConfig(default_ttl=ttl, refresh_on_read=True),
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


def test_list_namespaces(store):
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
    assert len(result) == store.collection.count_documents({})


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
    expected_updated_at = store.collection.find_one(
        dict(namespace=["a", "b", "d", "i"], key="id_2")
    )["updated_at"]

    result = store.get(namespace=("a", "b", "d", "i"), key="id_2", refresh_ttl=False)
    assert result.updated_at == expected_updated_at


def test_ttl():
    namespace = ("a", "b", "c", "d", "e")
    key = "thread"
    value = {"human": "What is the weather in SF?", "ai": "It's always sunny in SF."}

    # refresh_on_read is True
    with MongoDBStore.from_conn_string(
        conn_string=MONGODB_URI,
        db_name=DB_NAME,
        collection_name=COLLECTION_NAME,
        ttl_config=TTLConfig(default_ttl=3600, refresh_on_read=True)
    ) as store:
        store.collection.delete_many({})
        store.put(namespace=namespace, key=key, value=value)
        orig_updated_at = store.collection.find_one({})['updated_at']
        res = store.get(namespace=namespace, key=key)
        new_updated_at = store.collection.find_one({})['updated_at']
        assert new_updated_at > orig_updated_at
        assert res.updated_at == new_updated_at

    # refresh_on_read is False
    with MongoDBStore.from_conn_string(
        conn_string=MONGODB_URI,
        db_name=DB_NAME,
        collection_name=COLLECTION_NAME,
        ttl_config=TTLConfig(default_ttl=3600, refresh_on_read=False)
    ) as store:
        store.collection.delete_many({})
        store.put(namespace=namespace, key=key, value=value)
        orig_updated_at = store.collection.find_one({})['updated_at']
        res = store.get(namespace=namespace, key=key)
        new_updated_at = store.collection.find_one({})['updated_at']
        assert new_updated_at == orig_updated_at
        assert res.updated_at == new_updated_at

    # ttl_config is None
    with MongoDBStore.from_conn_string(
        conn_string=MONGODB_URI,
        db_name=DB_NAME,
        collection_name=COLLECTION_NAME,
        ttl_config=None
    ) as store:
        store.collection.delete_many({})
        store.put(namespace=namespace, key=key, value=value)
        orig_updated_at = store.collection.find_one({})['updated_at']
        res = store.get(namespace=namespace, key=key)
        new_updated_at = store.collection.find_one({})['updated_at']
        assert new_updated_at > orig_updated_at
        assert res.updated_at == new_updated_at

    # refresh_on_read is True but refresh_ttl=False in get()
    with MongoDBStore.from_conn_string(
        conn_string=MONGODB_URI,
        db_name=DB_NAME,
        collection_name=COLLECTION_NAME,
        ttl_config=TTLConfig(default_ttl=3600, refresh_on_read=True)
    ) as store:
        store.collection.delete_many({})
        store.put(namespace=namespace, key=key, value=value)
        orig_updated_at = store.collection.find_one({})['updated_at']
        res = store.get(refresh_ttl=False, namespace=namespace, key=key)
        new_updated_at = store.collection.find_one({})['updated_at']
        assert new_updated_at == orig_updated_at
        assert res.updated_at == new_updated_at


def test_put(store:MongoDBStore):
    n = store.collection.count_documents({})
    store.put(namespace=("a",), key=f"id_{n}", value={"data": f"value_{n:02d}"})
    assert store.collection.count_documents({}) == n + 1
    store.put(namespace=("a",), key=f"id_{n}", value={"data": f"value_{n:02d}"})
    assert store.collection.count_documents({}) == n + 1

    with pytest.raises(NotImplementedError):
        store.put(("a",), "idx", {"data": "val"}, index=['idx'])


def test_delete(store:MongoDBStore):
    n_items = store.collection.count_documents({})
    store.delete(namespace=("a", "b", "c"), key="id_0")
    assert store.collection.count_documents({}) == n_items - 1
    store.delete(namespace=("a", "b", "c"), key="id_0")
    assert store.collection.count_documents({}) == n_items - 1
