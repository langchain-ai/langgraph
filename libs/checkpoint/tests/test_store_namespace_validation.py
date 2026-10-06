import pytest

from langgraph.store.base import BaseStore, InvalidNamespaceError
from langgraph.store.base.batch import AsyncBatchedBaseStore

INVALID_NAMESPACES = [("foo.bar",), ("foo", ""), (123,)]
METHODS = ["get", "delete", "search", "prefix", "suffix"]


class RecordingStore(BaseStore):
    def __init__(self):
        self.operations = []

    def batch(self, ops):
        ops = list(ops)
        self.operations.extend(ops)
        return [None for _ in ops]

    async def abatch(self, ops):
        return self.batch(ops)


class RecordingBatchedStore(AsyncBatchedBaseStore):
    def __init__(self):
        super().__init__()
        self.operations = []

    async def abatch(self, ops):
        ops = list(ops)
        self.operations.extend(ops)
        return [None for _ in ops]


@pytest.mark.parametrize("namespace", INVALID_NAMESPACES)
@pytest.mark.parametrize("method", METHODS)
def test_base_rejects_invalid_labels(namespace, method):
    store = RecordingStore()
    with pytest.raises(InvalidNamespaceError):
        if method in ("prefix", "suffix"):
            store.list_namespaces(**{method: namespace})
        elif method == "search":
            store.search(namespace)
        else:
            getattr(store, method)(namespace, "key")
    assert store.operations == []


@pytest.mark.parametrize("store_type", [RecordingStore, RecordingBatchedStore])
@pytest.mark.parametrize("namespace", INVALID_NAMESPACES)
@pytest.mark.parametrize("method", METHODS)
async def test_async_rejects_invalid_labels(store_type, namespace, method):
    store = store_type()
    try:
        with pytest.raises(InvalidNamespaceError):
            if method in ("prefix", "suffix"):
                await store.alist_namespaces(**{method: namespace})
            elif method == "search":
                await store.asearch(namespace)
            else:
                await getattr(store, f"a{method}")(namespace, "key")
        if isinstance(store, RecordingBatchedStore):
            assert store._aqueue.empty()
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
    assert [c.path for c in store.operations[2].match_conditions] == [
        ("tenant", "*"),
        ("*",),
    ]


@pytest.mark.parametrize("store_type", [RecordingStore, RecordingBatchedStore])
async def test_async_empty_search_and_listing_wildcards(store_type):
    store = store_type()
    try:
        await store.asearch(())
        await store.asearch(("tenant", "a_%"))
        await store.alist_namespaces(prefix=("tenant", "*"), suffix=("*",))
    finally:
        if isinstance(store, RecordingBatchedStore) and store._task:
            store._task.cancel()
    assert store.operations[0].namespace_prefix == ()
    assert store.operations[1].namespace_prefix == ("tenant", "a_%")
    assert [c.path for c in store.operations[2].match_conditions] == [
        ("tenant", "*"),
        ("*",),
    ]
