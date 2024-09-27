import asyncio
from datetime import datetime
from typing import Iterable

from pytest_mock import MockerFixture

from langgraph.store.base import GetOp, Item, Op, Result
from langgraph.store.batch import AsyncBatchedBaseStore
from langgraph.store.memory import InMemoryStore


async def test_async_batch_store(mocker: MockerFixture) -> None:
    abatch = mocker.stub()

    class MockStore(AsyncBatchedBaseStore):
        def batch(self, ops: Iterable[Op]) -> list[Result]:
            raise NotImplementedError

        async def abatch(self, ops: Iterable[Op]) -> list[Result]:
            assert all(isinstance(op, GetOp) for op in ops)
            abatch(ops)
            return [
                Item(
                    value={},
                    scores={},
                    id=getattr(op, "id", ""),
                    namespace=getattr(op, "namespace", ()),
                    created_at=datetime(2024, 9, 24, 17, 29, 10, 128397),
                    updated_at=datetime(2024, 9, 24, 17, 29, 10, 128397),
                )
                for op in ops
            ]

    store = MockStore()

    # concurrent calls are batched
    results = await asyncio.gather(
        store.aget(namespace=("a",), id="b"),
        store.aget(namespace=("c",), id="d"),
    )
    assert results == [
        Item(
            {},
            {},
            "b",
            ("a",),
            datetime(2024, 9, 24, 17, 29, 10, 128397),
            datetime(2024, 9, 24, 17, 29, 10, 128397),
            datetime(2024, 9, 24, 17, 29, 10, 128397),
        ),
        Item(
            {},
            {},
            "d",
            ("c",),
            datetime(2024, 9, 24, 17, 29, 10, 128397),
            datetime(2024, 9, 24, 17, 29, 10, 128397),
            datetime(2024, 9, 24, 17, 29, 10, 128397),
        ),
    ]
    assert abatch.call_count == 1
    assert [tuple(c.args[0]) for c in abatch.call_args_list] == [
        (
            GetOp(("a",), "b"),
            GetOp(("c",), "d"),
        ),
    ]


def test_list_namespaces_basic() -> None:
    store = InMemoryStore()

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
        store.put(namespace=ns, id=f"id_{i}", value={"data": f"value_{i:02d}"})

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

    # Test max_depth
    result = store.list_namespaces(prefix=("a", "b"), max_depth=3)
    expected = [
        ("a", "b", "c"),
        ("a", "b", "d"),
        ("a", "b", "f"),
    ]
    assert sorted(result) == sorted(expected)

    # Test limit and offset
    result = store.list_namespaces(prefix=("a", "b"), limit=2)
    expected = [
        ("a", "b", "c"),
        ("a", "b", "d", "e"),
    ]
    assert result == expected

    result = store.list_namespaces(prefix=("a", "b"), offset=2)
    expected = [
        ("a", "b", "d", "i"),
        ("a", "b", "f"),
    ]
    assert result == expected

    result = store.list_namespaces(prefix=("a", "*", "f"))
    expected = [
        ("a", "b", "f"),
        ("a", "c", "f"),
    ]
    assert sorted(result) == sorted(expected)

    result = store.list_namespaces(suffix=("*", "f"))
    expected = [
        ("a", "b", "f"),
        ("a", "c", "f"),
        ("b", "a", "f"),
    ]
    assert sorted(result) == sorted(expected)

    result = store.list_namespaces(prefix=("nonexistent",))
    assert result == []

    result = store.list_namespaces(prefix=("users", "123"))
    expected = [("users", "123")]
    assert result == expected


def test_list_namespaces_with_wildcards() -> None:
    store = InMemoryStore()

    namespaces = [
        ("users", "123"),
        ("users", "456"),
        ("users", "789", "settings"),
        ("admin", "users", "789"),
        ("guests", "123"),
        ("guests", "456", "preferences"),
    ]

    for i, ns in enumerate(namespaces):
        store.put(namespace=ns, id=f"id_{i}", value={"data": f"value_{i:02d}"})

    result = store.list_namespaces(prefix=("users", "*"))
    expected = [
        ("users", "123"),
        ("users", "456"),
        ("users", "789", "settings"),
    ]
    assert sorted(result) == sorted(expected)

    result = store.list_namespaces(suffix=("*", "preferences"))
    expected = [
        ("guests", "456", "preferences"),
    ]
    assert result == expected

    result = store.list_namespaces(prefix=("*", "users"), suffix=("*", "settings"))

    assert result == []

    store.put(
        namespace=("admin", "users", "settings", "789"),
        id="foo",
        value={"data": "some_val"},
    )
    expected = [
        ("admin", "users", "settings", "789"),
    ]


def test_list_namespaces_pagination() -> None:
    store = InMemoryStore()

    for i in range(20):
        ns = ("namespace", f"sub_{i:02d}")
        store.put(namespace=ns, id=f"id_{i:02d}", value={"data": f"value_{i:02d}"})

    result = store.list_namespaces(prefix=("namespace",), limit=5, offset=0)
    expected = [("namespace", f"sub_{i:02d}") for i in range(5)]
    assert result == expected

    result = store.list_namespaces(prefix=("namespace",), limit=5, offset=5)
    expected = [("namespace", f"sub_{i:02d}") for i in range(5, 10)]
    assert result == expected

    result = store.list_namespaces(prefix=("namespace",), limit=5, offset=15)
    expected = [("namespace", f"sub_{i:02d}") for i in range(15, 20)]
    assert result == expected


def test_list_namespaces_max_depth() -> None:
    store = InMemoryStore()

    namespaces = [
        ("a", "b", "c", "d"),
        ("a", "b", "c", "e"),
        ("a", "b", "f"),
        ("a", "g"),
        ("h", "i", "j", "k"),
    ]

    for i, ns in enumerate(namespaces):
        store.put(namespace=ns, id=f"id_{i}", value={"data": f"value_{i:02d}"})

    result = store.list_namespaces(max_depth=2)
    expected = [
        ("a", "b"),
        ("a", "g"),
        ("h", "i"),
    ]
    assert sorted(result) == sorted(expected)


def test_list_namespaces_no_conditions() -> None:
    store = InMemoryStore()

    namespaces = [
        ("a", "b"),
        ("c", "d"),
        ("e", "f", "g"),
    ]

    for i, ns in enumerate(namespaces):
        store.put(namespace=ns, id=f"id_{i}", value={"data": f"value_{i:02d}"})

    result = store.list_namespaces()
    expected = namespaces
    assert sorted(result) == sorted(expected)


def test_list_namespaces_empty_store() -> None:
    store = InMemoryStore()

    result = store.list_namespaces()
    assert result == []
