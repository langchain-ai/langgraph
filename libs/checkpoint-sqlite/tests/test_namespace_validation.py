import pytest
from langgraph.store.base import (
    GetOp,
    InvalidNamespaceError,
    ListNamespacesOp,
    MatchCondition,
    PutOp,
    SearchOp,
)

from langgraph.store.sqlite import SqliteStore
from langgraph.store.sqlite.aio import AsyncSqliteStore


@pytest.mark.parametrize("namespace", [("foo.bar",), ("foo", ""), ("foo", 1)])
@pytest.mark.parametrize("kind", ["get", "put", "delete", "search", "prefix", "suffix"])
def test_invalid_namespace_batch(namespace, kind):
    operations = {
        "get": GetOp(namespace, "key"),
        "put": PutOp(namespace, "key", {"changed": True}),
        "delete": PutOp(namespace, "key", None),
        "search": SearchOp(namespace),
        "prefix": ListNamespacesOp((MatchCondition("prefix", namespace),)),
        "suffix": ListNamespacesOp((MatchCondition("suffix", namespace),)),
    }
    with SqliteStore.from_conn_string(":memory:") as store:
        store.put(("foo", "bar"), "key", {"original": True})
        with pytest.raises(InvalidNamespaceError):
            store.batch([PutOp(("valid",), "key", {}), operations[kind]])
        item = store.get(("foo", "bar"), "key")
        assert item is not None and item.value == {"original": True}
        assert store.get(("valid",), "key") is None


@pytest.mark.parametrize("method", ["get", "delete", "search"])
def test_invalid_namespace_public(method):
    with SqliteStore.from_conn_string(":memory:") as store:
        store.put(("foo", "bar"), "key", {"original": True})
        args = (("foo.bar",),) if method == "search" else (("foo.bar",), "key")
        with pytest.raises(InvalidNamespaceError):
            getattr(store, method)(*args)
        item = store.get(("foo", "bar"), "key")
        assert item is not None and item.value == {"original": True}


async def test_async_namespace_validation():
    async with AsyncSqliteStore.from_conn_string(":memory:") as store:
        await store.aput(("foo", "bar"), "key", {"original": True})
        for op in (
            GetOp(("foo.bar",), "key"),
            PutOp(("foo.bar",), "key", {}),
            PutOp(("foo.bar",), "key", None),
            SearchOp(("foo.bar",)),
            ListNamespacesOp((MatchCondition("prefix", ("foo.bar",)),)),
        ):
            with pytest.raises(InvalidNamespaceError):
                await store.abatch([op])
        with pytest.raises(InvalidNamespaceError):
            await store.aget(("foo.bar",), "key")
        with pytest.raises(InvalidNamespaceError):
            await store.adelete(("foo.bar",), "key")
        with pytest.raises(InvalidNamespaceError):
            await store.asearch(("foo.bar",))
        item = await store.aget(("foo", "bar"), "key")
        assert item is not None and item.value == {"original": True}


def test_hierarchical_search_and_listing_preserved():
    with SqliteStore.from_conn_string(":memory:") as store:
        namespaces = [("user_%",), ("user_%", "child"), ("userX%",), ("USER_%",)]
        for namespace in namespaces:
            store.put(namespace, "key", {})
        assert {item.namespace for item in store.search(())} == set(namespaces)
        assert {item.namespace for item in store.search(("user_%",))} == set(
            namespaces[:2]
        )
        assert store.list_namespaces(prefix=("user_%", "*")) == [("user_%", "child")]
