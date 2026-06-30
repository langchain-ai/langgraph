"""
Regression tests for deterministic checkpoint ordering in SqliteSaver
and AsyncSqliteSaver when multiple checkpoints share the same timestamp.

Bug: When checkpoints had identical `ts` values, ORDER BY checkpoint_id DESC
     sorted lexically (alphabetical) instead of by insertion order.

Fix: Added `rowid DESC` as a tiebreaker so insertion order is always respected.

Files fixed:
  - libs/checkpoint-sqlite/langgraph/checkpoint/sqlite/__init__.py
  - libs/checkpoint-sqlite/langgraph/checkpoint/sqlite/aio.py
"""

from __future__ import annotations

import asyncio
import uuid

import pytest

from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

FIXED_TS = "2024-01-01T00:00:00+00:00"
THREAD_CONFIG = {"configurable": {"thread_id": "tie-test", "checkpoint_ns": ""}}


def _make_checkpoint(step: int) -> dict:
    """Return a minimal checkpoint with a forced fixed timestamp."""
    return {
        "v": 1,
        "ts": FIXED_TS,  # all identical — this is what triggers the bug
        "id": str(uuid.uuid4()),
        "channel_values": {},
        "channel_versions": {},
        "versions_seen": {},
        "pending_sends": [],
    }


def _make_metadata(step: int) -> dict:
    return {"source": "input", "step": step, "writes": {}, "parents": {}}


# ---------------------------------------------------------------------------
# Sync SqliteSaver tests
# ---------------------------------------------------------------------------


class TestSqliteSaverDeterministicOrdering:
    """Tests for SqliteSaver (sync) ordering correctness under same-timestamp ties."""

    def test_list_returns_insertion_order_on_same_timestamp(self):
        """
        list() must return checkpoints in reverse insertion order (newest first)
        even when all checkpoints share the same timestamp.

        Before the fix: returned in lexical checkpoint_id order (random/wrong).
        After the fix:  returned in rowid DESC order (insertion order, correct).
        """
        with SqliteSaver.from_conn_string(":memory:") as saver:
            inserted_ids = []

            for step in range(5):
                cp = _make_checkpoint(step)
                inserted_ids.append(cp["id"])
                saver.put(THREAD_CONFIG, cp, _make_metadata(step), {})

            results = list(saver.list(THREAD_CONFIG))
            returned_ids = [
                r.config["configurable"]["checkpoint_id"] for r in results
            ]

            # Newest insert should come first (descending insertion order)
            expected = list(reversed(inserted_ids))
            assert returned_ids == expected, (
                f"Expected reverse insertion order:\n  {expected}\n"
                f"Got:\n  {returned_ids}\n"
                "Likely cause: missing rowid DESC tiebreaker in list() query."
            )

    def test_get_tuple_returns_last_inserted_on_same_timestamp(self):
        """
        get_tuple() with no checkpoint_id must return the most recently
        inserted checkpoint, not the lexically largest checkpoint_id.
        """
        with SqliteSaver.from_conn_string(":memory:") as saver:
            last_id = None

            for step in range(5):
                cp = _make_checkpoint(step)
                last_id = cp["id"]
                saver.put(THREAD_CONFIG, cp, _make_metadata(step), {})

            result = saver.get_tuple(THREAD_CONFIG)
            assert result is not None
            returned_id = result.config["configurable"]["checkpoint_id"]

            assert returned_id == last_id, (
                f"Expected last inserted checkpoint_id: {last_id}\n"
                f"Got: {returned_id}\n"
                "Likely cause: missing rowid DESC tiebreaker in get_tuple() query."
            )

    def test_list_with_limit_respects_insertion_order(self):
        """list(limit=N) must return the N most recently inserted checkpoints."""
        with SqliteSaver.from_conn_string(":memory:") as saver:
            inserted_ids = []

            for step in range(6):
                cp = _make_checkpoint(step)
                inserted_ids.append(cp["id"])
                saver.put(THREAD_CONFIG, cp, _make_metadata(step), {})

            results = list(saver.list(THREAD_CONFIG, limit=3))
            returned_ids = [
                r.config["configurable"]["checkpoint_id"] for r in results
            ]

            # Should be the 3 most recently inserted, in reverse insertion order
            expected = list(reversed(inserted_ids))[:3]
            assert returned_ids == expected, (
                f"Expected top 3 by insertion order:\n  {expected}\n"
                f"Got:\n  {returned_ids}"
            )

    def test_ordering_stable_with_two_checkpoints(self):
        """Minimal two-checkpoint tie case — the simplest possible regression check."""
        with SqliteSaver.from_conn_string(":memory:") as saver:
            cp1 = _make_checkpoint(0)
            cp2 = _make_checkpoint(1)

            saver.put(THREAD_CONFIG, cp1, _make_metadata(0), {})
            saver.put(THREAD_CONFIG, cp2, _make_metadata(1), {})

            results = list(saver.list(THREAD_CONFIG))
            returned_ids = [
                r.config["configurable"]["checkpoint_id"] for r in results
            ]

            # cp2 inserted last, must come first
            assert returned_ids == [cp2["id"], cp1["id"]], (
                f"Expected [cp2, cp1], got {returned_ids}"
            )


# ---------------------------------------------------------------------------
# Async AsyncSqliteSaver tests
# ---------------------------------------------------------------------------


class TestAsyncSqliteSaverDeterministicOrdering:
    """Tests for AsyncSqliteSaver (async) ordering correctness under same-timestamp ties."""

    @pytest.fixture
    def event_loop(self):
        loop = asyncio.new_event_loop()
        yield loop
        loop.close()

    @pytest.mark.asyncio
    async def test_alist_returns_insertion_order_on_same_timestamp(self):
        """
        alist() must return checkpoints in reverse insertion order (newest first)
        even when all checkpoints share the same timestamp.
        """
        async with AsyncSqliteSaver.from_conn_string(":memory:") as saver:
            inserted_ids = []

            for step in range(5):
                cp = _make_checkpoint(step)
                inserted_ids.append(cp["id"])
                await saver.aput(THREAD_CONFIG, cp, _make_metadata(step), {})

            results = [r async for r in saver.alist(THREAD_CONFIG)]
            returned_ids = [
                r.config["configurable"]["checkpoint_id"] for r in results
            ]

            expected = list(reversed(inserted_ids))
            assert returned_ids == expected, (
                f"Expected reverse insertion order:\n  {expected}\n"
                f"Got:\n  {returned_ids}\n"
                "Likely cause: missing rowid DESC tiebreaker in alist() query."
            )

    @pytest.mark.asyncio
    async def test_aget_tuple_returns_last_inserted_on_same_timestamp(self):
        """
        aget_tuple() with no checkpoint_id must return the most recently
        inserted checkpoint, not the lexically largest checkpoint_id.
        """
        async with AsyncSqliteSaver.from_conn_string(":memory:") as saver:
            last_id = None

            for step in range(5):
                cp = _make_checkpoint(step)
                last_id = cp["id"]
                await saver.aput(THREAD_CONFIG, cp, _make_metadata(step), {})

            result = await saver.aget_tuple(THREAD_CONFIG)
            assert result is not None
            returned_id = result.config["configurable"]["checkpoint_id"]

            assert returned_id == last_id, (
                f"Expected last inserted checkpoint_id: {last_id}\n"
                f"Got: {returned_id}\n"
                "Likely cause: missing rowid DESC tiebreaker in aget_tuple() query."
            )

    @pytest.mark.asyncio
    async def test_alist_with_limit_respects_insertion_order(self):
        """alist(limit=N) must return the N most recently inserted checkpoints."""
        async with AsyncSqliteSaver.from_conn_string(":memory:") as saver:
            inserted_ids = []

            for step in range(6):
                cp = _make_checkpoint(step)
                inserted_ids.append(cp["id"])
                await saver.aput(THREAD_CONFIG, cp, _make_metadata(step), {})

            results = [r async for r in saver.alist(THREAD_CONFIG, limit=3)]
            returned_ids = [
                r.config["configurable"]["checkpoint_id"] for r in results
            ]

            expected = list(reversed(inserted_ids))[:3]
            assert returned_ids == expected, (
                f"Expected top 3 by insertion order:\n  {expected}\n"
                f"Got:\n  {returned_ids}"
            )

    @pytest.mark.asyncio
    async def test_ordering_stable_with_two_checkpoints(self):
        """Minimal two-checkpoint tie case for async path."""
        async with AsyncSqliteSaver.from_conn_string(":memory:") as saver:
            cp1 = _make_checkpoint(0)
            cp2 = _make_checkpoint(1)

            await saver.aput(THREAD_CONFIG, cp1, _make_metadata(0), {})
            await saver.aput(THREAD_CONFIG, cp2, _make_metadata(1), {})

            results = [r async for r in saver.alist(THREAD_CONFIG)]
            returned_ids = [
                r.config["configurable"]["checkpoint_id"] for r in results
            ]

            assert returned_ids == [cp2["id"], cp1["id"]], (
                f"Expected [cp2, cp1], got {returned_ids}"
            )