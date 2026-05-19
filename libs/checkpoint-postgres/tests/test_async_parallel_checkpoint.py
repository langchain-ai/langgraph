# type: ignore
"""Regression coverage for concurrent async checkpointing through a pooled saver.

Context: issue #7259 / PR #7269. `AsyncPostgresSaver` serializes every `_cursor()`
operation behind an instance `asyncio.Lock()`; with an `AsyncConnectionPool` that
serialization caps effective concurrency at 1 regardless of `pool.max_size`. These
tests pin the *correctness* invariants that any lock change must preserve:

- many concurrent `aput` through a pooled saver must all persist and be retrievable;
- `from_conn_string` (a single shared `AsyncConnection`) must keep using the instance
  lock — it is not safe to drop serialization for a shared connection.

Both pass on current `main` and are intended to keep passing after #7269 lands, so
they guard the behavior boundary rather than the performance change itself.
"""

import asyncio
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from uuid import uuid4

import pytest
from langgraph.checkpoint.base import empty_checkpoint
from psycopg import AsyncConnection
from psycopg.rows import dict_row
from psycopg_pool import AsyncConnectionPool

from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from tests.conftest import DEFAULT_POSTGRES_URI


class _ExplodingAsyncLock:
    """Stand-in for ``saver.lock``; fails if the saver ever acquires it."""

    async def __aenter__(self):
        raise AssertionError("instance lock was acquired but should have been bypassed")

    async def __aexit__(self, *exc):
        return False


def _cfg(thread_id: str) -> dict:
    return {
        "configurable": {
            "thread_id": thread_id,
            "checkpoint_ns": "",
            "checkpoint_id": "",
        }
    }


@asynccontextmanager
async def _fresh_db() -> AsyncIterator[str]:
    database = f"par_{uuid4().hex[:16]}"
    async with await AsyncConnection.connect(
        DEFAULT_POSTGRES_URI, autocommit=True
    ) as c:
        await c.execute(f"CREATE DATABASE {database}")
    try:
        yield DEFAULT_POSTGRES_URI + database
    finally:
        async with await AsyncConnection.connect(
            DEFAULT_POSTGRES_URI, autocommit=True
        ) as c:
            await c.execute(f"DROP DATABASE {database} WITH (FORCE)")


async def test_parallel_aput_under_pool_persists_all() -> None:
    """100 concurrent ``aput`` through a pooled saver must all persist and be
    individually retrievable. Correctness must not be traded for concurrency."""
    n = 100
    async with _fresh_db() as uri:
        async with AsyncConnectionPool(
            uri,
            min_size=10,
            max_size=10,
            kwargs={"autocommit": True, "row_factory": dict_row},
        ) as pool:
            saver = AsyncPostgresSaver(pool)
            await saver.setup()
            ids = [f"par-{uuid4()}" for _ in range(n)]

            await asyncio.gather(
                *[saver.aput(_cfg(t), empty_checkpoint(), {}, {}) for t in ids]
            )

            got = await asyncio.gather(*[saver.aget_tuple(_cfg(t)) for t in ids])
            assert all(g is not None for g in got)
            assert len({g.config["configurable"]["thread_id"] for g in got}) == n


async def test_from_conn_string_keeps_instance_lock() -> None:
    """``from_conn_string`` yields a single shared ``AsyncConnection``; dropping
    the instance lock there would be unsafe, so it must still be acquired."""
    async with _fresh_db() as uri:
        async with AsyncPostgresSaver.from_conn_string(uri) as saver:
            await saver.setup()
            assert not isinstance(saver.conn, AsyncConnectionPool)
            saver.lock = _ExplodingAsyncLock()
            with pytest.raises(AssertionError, match="should have been bypassed"):
                await saver.aput(_cfg(f"x-{uuid4()}"), empty_checkpoint(), {}, {})
