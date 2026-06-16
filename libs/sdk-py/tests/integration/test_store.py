"""`StoreClient` against the integration API.

Covers the put / get / search / list_namespaces / delete round-trip
under a unique-per-test namespace so concurrent runs don't collide.
"""

from __future__ import annotations

import uuid

import pytest

pytestmark = pytest.mark.integration


def _async_store(raw):
    from langgraph_sdk._async.http import HttpClient
    from langgraph_sdk._async.store import StoreClient

    return StoreClient(HttpClient(raw))


def _sync_store(raw):
    from langgraph_sdk._sync.http import SyncHttpClient
    from langgraph_sdk._sync.store import SyncStoreClient

    return SyncStoreClient(SyncHttpClient(raw))


def _unique_namespace(label: str) -> list[str]:
    return ["test-integration", label, uuid.uuid4().hex[:12]]


async def test_store_put_get_delete_async(async_threads) -> None:
    _, raw = async_threads
    store = _async_store(raw)
    ns = _unique_namespace("put-async")
    key = "doc-1"
    payload = {"title": "Hello", "body": "World"}

    await store.put_item(ns, key=key, value=payload)
    try:
        fetched = await store.get_item(ns, key=key)
        assert fetched["value"] == payload
        assert fetched["namespace"] == ns
        assert fetched["key"] == key
    finally:
        await store.delete_item(ns, key=key)

    missing = await store.get_item(ns, key=key)
    assert missing is None


def test_store_put_get_delete_sync(sync_threads) -> None:
    _, raw = sync_threads
    store = _sync_store(raw)
    ns = _unique_namespace("put-sync")
    key = "doc-1"
    payload = {"title": "Hello", "body": "World"}

    store.put_item(ns, key=key, value=payload)
    try:
        fetched = store.get_item(ns, key=key)
        assert fetched["value"] == payload
        assert fetched["namespace"] == ns
        assert fetched["key"] == key
    finally:
        store.delete_item(ns, key=key)

    missing = store.get_item(ns, key=key)
    assert missing is None


async def test_store_search_and_list_namespaces_async(async_threads) -> None:
    _, raw = async_threads
    store = _async_store(raw)
    ns = _unique_namespace("search-async")
    await store.put_item(ns, key="a", value={"kind": "alpha"})
    await store.put_item(ns, key="b", value={"kind": "beta"})
    try:
        search = await store.search_items(ns, limit=10)
        items = search.get("items", search) if isinstance(search, dict) else search
        keys = sorted(i["key"] for i in items)
        assert keys == ["a", "b"]

        namespaces_result = await store.list_namespaces(prefix=ns[:1], limit=100)
        namespaces = (
            namespaces_result.get("namespaces", namespaces_result)
            if isinstance(namespaces_result, dict)
            else namespaces_result
        )
        assert any(list(found) == ns for found in namespaces), (
            f"namespace {ns!r} not in list_namespaces result"
        )
    finally:
        await store.delete_item(ns, key="a")
        await store.delete_item(ns, key="b")


def test_store_search_and_list_namespaces_sync(sync_threads) -> None:
    _, raw = sync_threads
    store = _sync_store(raw)
    ns = _unique_namespace("search-sync")
    store.put_item(ns, key="a", value={"kind": "alpha"})
    store.put_item(ns, key="b", value={"kind": "beta"})
    try:
        search = store.search_items(ns, limit=10)
        items = search.get("items", search) if isinstance(search, dict) else search
        keys = sorted(i["key"] for i in items)
        assert keys == ["a", "b"]

        namespaces_result = store.list_namespaces(prefix=ns[:1], limit=100)
        namespaces = (
            namespaces_result.get("namespaces", namespaces_result)
            if isinstance(namespaces_result, dict)
            else namespaces_result
        )
        assert any(list(found) == ns for found in namespaces), (
            f"namespace {ns!r} not in list_namespaces result"
        )
    finally:
        store.delete_item(ns, key="a")
        store.delete_item(ns, key="b")
