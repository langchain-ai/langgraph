import asyncio
from collections.abc import Iterable
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from threading import Barrier, Event

import pytest

from langgraph.store.base import (
    NOT_PROVIDED,
    BaseStore,
    InvalidNamespaceError,
    Item,
    Op,
    Result,
    SearchItem,
)
from langgraph.store.memory import InMemoryStore


class UnsupportedStore(BaseStore):
    def batch(self, ops: Iterable[Op]) -> list[Result]:
        raise AssertionError("Conditional writes must not fall back to batch")

    async def abatch(self, ops: Iterable[Op]) -> list[Result]:
        raise AssertionError("Conditional writes must not fall back to abatch")


async def test_unsupported_store_fails_closed() -> None:
    store = UnsupportedStore()
    with pytest.raises(NotImplementedError, match="Atomic conditional writes"):
        store.compare_and_swap(("docs",), "key", {}, expected_revision=None)
    with pytest.raises(NotImplementedError, match="Atomic conditional writes"):
        await store.acompare_and_swap(("docs",), "key", {}, expected_revision=None)


def test_optional_revision_preserves_legacy_constructors_and_serialization() -> None:
    now = datetime.now(timezone.utc)
    item = Item(
        value={}, key="key", namespace=("docs",), created_at=now, updated_at=now
    )
    search_item = SearchItem(("docs",), "key", {}, now, now, 0.5)
    for snapshot in (item, search_item):
        assert snapshot.revision is None
        assert "revision" not in snapshot.dict()
        snapshot.revision = "opaque"
        assert snapshot.dict()["revision"] == "opaque"
    assert Item(**item.dict()) == item
    assert SearchItem(**search_item.dict()) == search_item


def test_conditional_create_replace_delete_and_aba() -> None:
    store = InMemoryStore()
    ns = ("docs",)
    assert not store.compare_and_swap(ns, "key", {}, expected_revision="missing")
    assert store.list_namespaces() == []
    assert store.compare_and_swap(ns, "key", None, expected_revision=None)
    assert store.list_namespaces() == []
    assert store.compare_and_swap(ns, "key", {"text": "a"}, expected_revision=None)
    first = store.get(ns, "key")
    assert first is not None and first.revision is not None
    assert not store.compare_and_swap(ns, "key", {}, expected_revision=None)
    store.put(ns, "key", first.value)
    same_value = store.get(ns, "key")
    assert same_value is not None and same_value.revision != first.revision
    assert not store.compare_and_swap(ns, "key", {}, expected_revision=first.revision)
    assert store.compare_and_swap(
        ns, "key", {"text": "b"}, expected_revision=same_value.revision
    )
    second = store.get(ns, "key")
    assert second is not None and second.revision != same_value.revision
    assert not store.compare_and_swap(ns, "key", None, expected_revision=first.revision)
    assert store.compare_and_swap(ns, "key", None, expected_revision=second.revision)
    assert store.get(ns, "key") is None
    assert not store.compare_and_swap(ns, "key", {}, expected_revision=second.revision)
    store.put(ns, "key", first.value)
    recreated = store.get(ns, "key")
    assert recreated is not None
    assert recreated.revision not in (first.revision, second.revision)
    assert not store.compare_and_swap(ns, "key", {}, expected_revision=first.revision)
    assert store.get(ns, "key") == recreated


def test_snapshots_and_write_inputs_do_not_alias_storage() -> None:
    store = InMemoryStore()
    ns = ("docs",)
    value = {"nested": {"text": "original"}}
    store.put(ns, "key", value)
    snapshot = store.get(ns, "key")
    assert snapshot is not None and snapshot.revision is not None
    revision = snapshot.revision
    value["nested"]["text"] = "input mutation"
    snapshot.value["nested"]["text"] = "get mutation"
    snapshot.revision = "changed by caller"
    found = store.search(ns)[0]
    assert found.revision == revision
    found.value["nested"]["text"] = "search mutation"
    replacement = {"nested": {"text": "replacement"}}
    assert store.compare_and_swap(ns, "key", replacement, expected_revision=revision)
    replacement["nested"]["text"] = "conditional input mutation"
    current = store.get(ns, "key")
    assert current is not None
    assert current.value == {"nested": {"text": "replacement"}}
    assert snapshot.value == {"nested": {"text": "get mutation"}}


async def test_invalid_conditions_cannot_bypass_comparison() -> None:
    store = InMemoryStore()
    for expected in (NOT_PROVIDED, 1):
        with pytest.raises(TypeError, match="expected_revision"):
            store.compare_and_swap(("docs",), "key", {}, expected_revision=expected)
        with pytest.raises(TypeError, match="expected_revision"):
            await store.acompare_and_swap(
                ("docs",), "key", {}, expected_revision=expected
            )
    with pytest.raises(InvalidNamespaceError):
        store.compare_and_swap(("bad.label",), "key", {}, expected_revision=None)
    with pytest.raises(InvalidNamespaceError):
        await store.acompare_and_swap((), "key", {}, expected_revision=None)


def test_threaded_contenders_have_one_winner() -> None:
    store = InMemoryStore()
    store.put(("docs",), "key", {"count": 0})
    snapshot = store.get(("docs",), "key")
    assert snapshot is not None
    barrier = Barrier(8, timeout=5)

    def update(count: int) -> bool:
        barrier.wait()
        return store.compare_and_swap(
            ("docs",), "key", {"count": count}, expected_revision=snapshot.revision
        )

    with ThreadPoolExecutor(max_workers=8) as pool:
        results = list(pool.map(update, range(1, 9)))
    assert sum(results) == 1


def test_sync_embedding_rechecks_after_concurrent_async_write() -> None:
    started, resume = Event(), Event()

    def embed(texts: list[str]) -> list[list[float]]:
        started.set()
        assert resume.wait(5)
        return [[1.0, 0.0] for _ in texts]

    store = InMemoryStore(index={"dims": 2, "embed": embed, "fields": ["text"]})
    store.put(("docs",), "key", {"text": "initial"}, index=False)
    snapshot = store.get(("docs",), "key")
    assert snapshot is not None
    with ThreadPoolExecutor(max_workers=1) as pool:
        pending = pool.submit(
            store.compare_and_swap,
            ("docs",),
            "key",
            {"text": "paused"},
            expected_revision=snapshot.revision,
        )
        try:
            assert started.wait(5)
            asyncio.run(store.aput(("docs",), "key", {"text": "winner"}, index=False))
            winner = store.get(("docs",), "key")
        finally:
            resume.set()
        assert pending.result(timeout=5) is False
    assert store.get(("docs",), "key") == winner
    assert store._vectors[("docs",)]["key"] == {}


@pytest.mark.parametrize("ordinary_pending", [False, True])
async def test_async_embedding_commits_value_revision_and_vectors_together(
    ordinary_pending: bool,
) -> None:
    started, resume = asyncio.Event(), asyncio.Event()

    async def embed(texts: list[str]) -> list[list[float]]:
        if texts == ["paused"]:
            started.set()
            await resume.wait()
        return [[1.0, 0.0] for _ in texts]

    store = InMemoryStore(index={"dims": 2, "embed": embed, "fields": ["text"]})
    store.put(("docs",), "key", {"text": "initial"}, index=False)
    snapshot = store.get(("docs",), "key")
    assert snapshot is not None
    value = {"text": "paused"}
    index = ["text"]
    pending = asyncio.create_task(
        store.aput(("docs",), "key", value, index=index)
        if ordinary_pending
        else store.acompare_and_swap(
            ("docs",), "key", value, index=index, expected_revision=snapshot.revision
        )
    )
    try:
        await asyncio.wait_for(started.wait(), timeout=5)
        value["text"] = "mutated during embedding"
        index.clear()
        if ordinary_pending:
            assert store.compare_and_swap(
                ("docs",),
                "key",
                {"text": "winner"},
                index=False,
                expected_revision=snapshot.revision,
            )
        else:
            store.put(("docs",), "key", {"text": "winner"}, index=False)
        winner = store.get(("docs",), "key")
    finally:
        resume.set()
    result = await asyncio.wait_for(pending, timeout=5)
    current = store.get(("docs",), "key")
    assert current is not None
    if ordinary_pending:
        assert result is None
        assert current.value == {"text": "paused"}
        assert winner is not None and current.revision != winner.revision
        assert store._vectors[("docs",)]["key"] == {"text": [1.0, 0.0]}
    else:
        assert result is False
        assert current == winner
        assert store._vectors[("docs",)]["key"] == {}


async def test_async_contenders_with_yielding_embeddings_have_one_winner() -> None:
    async def embed(texts: list[str]) -> list[list[float]]:
        await asyncio.sleep(0)
        return [[1.0, 0.0] for _ in texts]

    store = InMemoryStore(index={"dims": 2, "embed": embed, "fields": ["text"]})
    results = await asyncio.gather(
        *(
            store.acompare_and_swap(
                ("docs",), "key", {"text": str(i)}, expected_revision=None
            )
            for i in range(8)
        )
    )
    assert sum(results) == 1


@pytest.mark.parametrize("failure", ["error", "cancel", "invalid_count"])
async def test_embedding_failure_leaves_existing_snapshot_and_vectors_untouched(
    failure: str,
) -> None:
    started, resume = asyncio.Event(), asyncio.Event()

    async def embed(texts: list[str]) -> list[list[float]]:
        if texts == ["paused"]:
            started.set()
            await resume.wait()
            if failure == "error":
                raise RuntimeError("embedding failed")
            return []
        return [[1.0, 0.0] for _ in texts]

    store = InMemoryStore(index={"dims": 2, "embed": embed, "fields": ["text"]})
    await store.aput(("docs",), "key", {"text": "initial"})
    snapshot = await store.aget(("docs",), "key")
    assert snapshot is not None
    pending = asyncio.create_task(
        store.acompare_and_swap(
            ("docs",), "key", {"text": "paused"}, expected_revision=snapshot.revision
        )
    )
    try:
        await asyncio.wait_for(started.wait(), timeout=5)
    finally:
        if failure == "cancel":
            pending.cancel()
        resume.set()
    expected_error = {
        "error": RuntimeError,
        "cancel": asyncio.CancelledError,
        "invalid_count": ValueError,
    }[failure]
    with pytest.raises(expected_error):
        await asyncio.wait_for(pending, timeout=5)
    assert await store.aget(("docs",), "key") == snapshot
    assert store._vectors[("docs",)]["key"] == {"text": [1.0, 0.0]}


async def test_async_conditional_delete_and_recreate_invalidates_revision() -> None:
    store = InMemoryStore()
    assert await store.acompare_and_swap(("docs",), "key", {}, expected_revision=None)
    snapshot = await store.aget(("docs",), "key")
    assert snapshot is not None and snapshot.revision is not None
    assert await store.acompare_and_swap(
        ("docs",), "key", None, expected_revision=snapshot.revision
    )
    assert await store.aget(("docs",), "key") is None
    assert await store.acompare_and_swap(("docs",), "key", {}, expected_revision=None)
    assert not await store.acompare_and_swap(
        ("docs",), "key", None, expected_revision=snapshot.revision
    )


def test_conditional_replacement_removes_obsolete_vectors() -> None:
    def embed(texts: list[str]) -> list[list[float]]:
        return [[1.0, 0.0] for _ in texts]

    store = InMemoryStore(
        index={"dims": 2, "embed": embed, "fields": ["left", "right"]}
    )
    value = {"left": "same text", "right": "same text"}
    store.put(("docs",), "key", value)
    snapshot = store.get(("docs",), "key")
    assert snapshot is not None
    assert store.compare_and_swap(
        ("docs",), "key", value, expected_revision=snapshot.revision, index=["left"]
    )
    assert store._vectors[("docs",)]["key"] == {"left": [1.0, 0.0]}
    snapshot = store.get(("docs",), "key")
    assert snapshot is not None
    assert store.compare_and_swap(
        ("docs",), "key", value, expected_revision=snapshot.revision, index=False
    )
    assert store._vectors[("docs",)]["key"] == {}


async def test_retry_recomputes_from_fresh_snapshot_without_lost_updates() -> None:
    store = InMemoryStore()
    await store.aput(("docs",), "key", {"count": 0})

    async def increment() -> int:
        for attempt in range(8):
            snapshot = await store.aget(("docs",), "key")
            assert snapshot is not None and snapshot.revision is not None
            await asyncio.sleep(0)
            if await store.acompare_and_swap(
                ("docs",),
                "key",
                {"count": snapshot.value["count"] + 1},
                expected_revision=snapshot.revision,
            ):
                return attempt
        raise AssertionError("Retry budget exhausted")

    attempts = await asyncio.gather(*(increment() for _ in range(8)))
    snapshot = await store.aget(("docs",), "key")
    assert snapshot is not None and snapshot.value == {"count": 8}
    assert max(attempts) > 0
