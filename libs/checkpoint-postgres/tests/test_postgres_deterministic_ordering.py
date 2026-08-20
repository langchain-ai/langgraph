"""
Regression tests for deterministic checkpoint ordering in PostgresSaver
and AsyncPostgresSaver when multiple checkpoints share the same timestamp.

Bug: When checkpoints had identical `ts` values, ORDER BY checkpoint_id DESC
     sorted lexically (random UUID order) instead of by insertion order.

Fix: Added `ctid DESC` as a tiebreaker so insertion order is always respected.

Files fixed:
  - libs/checkpoint-postgres/langgraph/checkpoint/postgres/__init__.py
  - libs/checkpoint-postgres/langgraph/checkpoint/postgres/aio.py

Setup:
  Requires a running Postgres instance. Set the DB_URI environment variable
  or update the default below.

  export DB_URI="postgres://postgres:postgres@localhost:5432/postgres?sslmode=disable"

  Then run:
  pytest tests/test_postgres_deterministic_ordering.py -v
"""

from __future__ import annotations

import asyncio
import os
import uuid

import pytest

from langgraph.checkpoint.postgres import PostgresSaver
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver


# ---------------------------------------------------------------------------
# Config — override via environment variable if needed
# ---------------------------------------------------------------------------

DB_URI = os.environ.get(
    "DB_URI",
    "postgres://postgres:postgres@localhost:5432/postgres?sslmode=disable",
)

FIXED_TS = "2024-01-01T00:00:00+00:00"


def _thread_config(thread_id: str) -> dict:
    """Each test gets its own thread_id to avoid cross-test pollution."""
    return {"configurable": {"thread_id": thread_id, "checkpoint_ns": ""}}


def _make_checkpoint(step: int) -> dict:
    """Return a minimal checkpoint with a forced fixed timestamp."""
    return {
        "v": 4,
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
# Sync PostgresSaver tests
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def saver():
    """Module-scoped sync saver — setup once, shared across sync tests."""
    with PostgresSaver.from_conn_string(DB_URI) as s:
        s.setup()
        yield s


class TestPostgresSaverDeterministicOrdering:
    """Tests for PostgresSaver (sync) ordering correctness under same-timestamp ties."""

    def test_list_returns_insertion_order_on_same_timestamp(self, saver):
        """
        list() must return checkpoints in reverse insertion order (newest first)
        even when all checkpoints share the same timestamp.

        Before the fix: returned in lexical checkpoint_id (UUID) order — random.
        After the fix:  returned in ctid DESC order — insertion order, correct.
        """
        config = _thread_config(f"sync-list-{uuid.uuid4()}")
        inserted_ids = []

        for step in range(5):
            cp = _make_checkpoint(step)
            inserted_ids.append(cp["id"])
            saver.put(config, cp, _make_metadata(step), {})

        results = list(saver.list(config))
        returned_ids = [r.config["configurable"]["checkpoint_id"] for r in results]

        expected = list(reversed(inserted_ids))
        assert returned_ids == expected, (
            f"Expected reverse insertion order:\n  {expected}\n"
            f"Got:\n  {returned_ids}\n"
            "Likely cause: missing ctid DESC tiebreaker in list() query."
        )

    def test_get_tuple_returns_last_inserted_on_same_timestamp(self, saver):
        """
        get_tuple() with no checkpoint_id must return the most recently
        inserted checkpoint, not the lexically largest UUID checkpoint_id.
        """
        config = _thread_config(f"sync-get-{uuid.uuid4()}")
        last_id = None

        for step in range(5):
            cp = _make_checkpoint(step)
            last_id = cp["id"]
            saver.put(config, cp, _make_metadata(step), {})

        result = saver.get_tuple(config)
        assert result is not None
        returned_id = result.config["configurable"]["checkpoint_id"]

        assert returned_id == last_id, (
            f"Expected last inserted checkpoint_id: {last_id}\n"
            f"Got: {returned_id}\n"
            "Likely cause: missing ctid DESC tiebreaker in get_tuple() query."
        )

    def test_list_with_limit_respects_insertion_order(self, saver):
        """list(limit=N) must return the N most recently inserted checkpoints."""
        config = _thread_config(f"sync-limit-{uuid.uuid4()}")
        inserted_ids = []

        for step in range(6):
            cp = _make_checkpoint(step)
            inserted_ids.append(cp["id"])
            saver.put(config, cp, _make_metadata(step), {})

        results = list(saver.list(config, limit=3))
        returned_ids = [r.config["configurable"]["checkpoint_id"] for r in results]

        expected = list(reversed(inserted_ids))[:3]
        assert returned_ids == expected, (
            f"Expected top 3 by insertion order:\n  {expected}\n"
            f"Got:\n  {returned_ids}"
        )

    def test_ordering_stable_with_two_checkpoints(self, saver):
        """Minimal two-checkpoint tie case — the simplest possible regression check."""
        config = _thread_config(f"sync-two-{uuid.uuid4()}")
        cp1 = _make_checkpoint(0)
        cp2 = _make_checkpoint(1)

        saver.put(config, cp1, _make_metadata(0), {})
        saver.put(config, cp2, _make_metadata(1), {})

        results = list(saver.list(config))
        returned_ids = [r.config["configurable"]["checkpoint_id"] for r in results]

        assert returned_ids == [cp2["id"], cp1["id"]], (
            f"Expected [cp2, cp1], got {returned_ids}"
        )


# ---------------------------------------------------------------------------
# Async AsyncPostgresSaver tests
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def event_loop():
    """Module-scoped event loop for async tests."""
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


@pytest.fixture(scope="module")
async def async_saver():
    """Module-scoped async saver — setup once, shared across async tests."""
    async with AsyncPostgresSaver.from_conn_string(DB_URI) as s:
        await s.setup()
        yield s


class TestAsyncPostgresSaverDeterministicOrdering:
    """Tests for AsyncPostgresSaver (async) ordering correctness under same-timestamp ties."""

    @pytest.mark.asyncio
    async def test_alist_returns_insertion_order_on_same_timestamp(self, async_saver):
        """
        alist() must return checkpoints in reverse insertion order (newest first)
        even when all checkpoints share the same timestamp.
        """
        config = _thread_config(f"async-list-{uuid.uuid4()}")
        inserted_ids = []

        for step in range(5):
            cp = _make_checkpoint(step)
            inserted_ids.append(cp["id"])
            await async_saver.aput(config, cp, _make_metadata(step), {})

        results = [r async for r in async_saver.alist(config)]
        returned_ids = [r.config["configurable"]["checkpoint_id"] for r in results]

        expected = list(reversed(inserted_ids))
        assert returned_ids == expected, (
            f"Expected reverse insertion order:\n  {expected}\n"
            f"Got:\n  {returned_ids}\n"
            "Likely cause: missing ctid DESC tiebreaker in alist() query."
        )

    @pytest.mark.asyncio
    async def test_aget_tuple_returns_last_inserted_on_same_timestamp(self, async_saver):
        """
        aget_tuple() with no checkpoint_id must return the most recently
        inserted checkpoint, not the lexically largest UUID checkpoint_id.
        """
        config = _thread_config(f"async-get-{uuid.uuid4()}")
        last_id = None

        for step in range(5):
            cp = _make_checkpoint(step)
            last_id = cp["id"]
            await async_saver.aput(config, cp, _make_metadata(step), {})

        result = await async_saver.aget_tuple(config)
        assert result is not None
        returned_id = result.config["configurable"]["checkpoint_id"]

        assert returned_id == last_id, (
            f"Expected last inserted checkpoint_id: {last_id}\n"
            f"Got: {returned_id}\n"
            "Likely cause: missing ctid DESC tiebreaker in aget_tuple() query."
        )

    @pytest.mark.asyncio
    async def test_alist_with_limit_respects_insertion_order(self, async_saver):
        """alist(limit=N) must return the N most recently inserted checkpoints."""
        config = _thread_config(f"async-limit-{uuid.uuid4()}")
        inserted_ids = []

        for step in range(6):
            cp = _make_checkpoint(step)
            inserted_ids.append(cp["id"])
            await async_saver.aput(config, cp, _make_metadata(step), {})

        results = [r async for r in async_saver.alist(config, limit=3)]
        returned_ids = [r.config["configurable"]["checkpoint_id"] for r in results]

        expected = list(reversed(inserted_ids))[:3]
        assert returned_ids == expected, (
            f"Expected top 3 by insertion order:\n  {expected}\n"
            f"Got:\n  {returned_ids}"
        )

    @pytest.mark.asyncio
    async def test_ordering_stable_with_two_checkpoints(self, async_saver):
        """Minimal two-checkpoint tie case for async Postgres path."""
        config = _thread_config(f"async-two-{uuid.uuid4()}")
        cp1 = _make_checkpoint(0)
        cp2 = _make_checkpoint(1)

        await async_saver.aput(config, cp1, _make_metadata(0), {})
        await async_saver.aput(config, cp2, _make_metadata(1), {})

        results = [r async for r in async_saver.alist(config)]
        returned_ids = [r.config["configurable"]["checkpoint_id"] for r in results]

        assert returned_ids == [cp2["id"], cp1["id"]], (
            f"Expected [cp2, cp1], got {returned_ids}"
        )