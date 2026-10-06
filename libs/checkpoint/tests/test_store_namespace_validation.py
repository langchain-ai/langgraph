import pytest

from langgraph.store.base import BaseStore, InvalidNamespaceError
from langgraph.store.base.batch import AsyncBatchedBaseStore


class RecordingStore(BaseStore):
    def __init__(self):
        self.operations = []

    def batch(self, ops):
        self.operations.extend(ops)
        return [None for _ in self.operations]

    async def abatch(self, ops):
        return self.batch(ops)


class RecordingBatchedStore(AsyncBatchedBaseStore):
    async def abatch(self, ops):
        return [None for _ in ops]


@pytest.mark.parametrize("namespace", [("foo.bar",), ("foo", ""), (123,)])
@pytest.mark.parametrize("method", ["get", "delete", "search", "list_namespaces"])
def test_base_rejects_invalid_labels(namespace, method):
    store = RecordingStore()
    args = (namespace, "key") if method in ("get", "delete") else (namespace,)
    with pytest.raises(InvalidNamespaceError):
        if method == "list_namespaces":
            store.list_namespaces(prefix=namespace)
        else:
            getattr(store, method)(*args)
    assert store.operations == []


@pytest.mark.parametrize("store_type", [RecordingStore, RecordingBatchedStore])
@pytest.mark.parametrize("namespace", [("foo.bar",), ("foo", ""), (123,)])
@pytest.mark.parametrize("method", ["aget", "adelete", "asearch", "alist_namespaces"])
async def test_async_rejects_invalid_labels(store_type, namespace, method):
    store = store_type()
    args = (namespace, "key") if method in ("aget", "adelete") else (namespace,)
    try:
        with pytest.raises(InvalidNamespaceError):
            if method == "alist_namespaces":
                await store.alist_namespaces(suffix=namespace)
            else:
                await getattr(store, method)(*args)
        if isinstance(store, RecordingBatchedStore):
            assert store._aqueue.empty()
        else:
            assert store.operations == []
    finally:
        if isinstance(store, RecordingBatchedStore) and store._task:
            store._task.cancel()


def test_search_and_listing_preserve_prefixes():
    store = RecordingStore()
    store.search(())
    store.search(("tenant", "a_%"))
    store.list_namespaces(prefix=("tenant", "*"), suffix=("*",))
    assert store.operations[0].namespace_prefix == ()
    assert store.operations[1].namespace_prefix == ("tenant", "a_%")
    assert store.operations[2].match_conditions[0].path == ("tenant", "*")


@pytest.mark.parametrize("store_type", [RecordingStore, RecordingBatchedStore])
async def test_async_empty_search_and_listing_wildcards(store_type):
    store = store_type()
    try:
        await store.asearch(())
        await store.alist_namespaces(prefix=("tenant", "*"))
    finally:
        if isinstance(store, RecordingBatchedStore) and store._task:
            store._task.cancel()
