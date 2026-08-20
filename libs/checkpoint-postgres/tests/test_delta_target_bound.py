"""Stage-1 paging bound for `DeltaChannel` history walks.

The walk reconstructs the value *at the target checkpoint* by following
parent links from the target down to the nearest seed, so rows newer than
the target can never contribute. The stage-1 pager must therefore start at
the target `checkpoint_id`, not at the thread's newest checkpoint —
otherwise every read of an old checkpoint (e.g. `get_state_history` with
`before`) first scans the entire newer prefix of the chain, once per
returned checkpoint.
"""

from __future__ import annotations

from contextlib import asynccontextmanager, contextmanager
from typing import Any
from uuid import uuid4

import pytest
from langgraph.checkpoint.base import Checkpoint, empty_checkpoint
from langgraph.checkpoint.base.id import uuid6
from langgraph.checkpoint.serde.types import _DeltaSnapshot

import langgraph.checkpoint.postgres as postgres_module
import langgraph.checkpoint.postgres.aio as aio_module
from langgraph.checkpoint.postgres import PostgresSaver
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from tests.conftest import DEFAULT_URI

CHANNEL = "items"
PAGE_SIZE = 4
CHAIN_LENGTH = 20
# The seed lives at step 1; target step 3 keeps the expected replay identical
# to the short-chain seed tests: w1 (the seed's own) and w2.
TARGET_STEP = 3
STAGE1_MARKER = "AS ver_0"


class _CountingCursor:
    """Pass-through cursor proxy that counts stage-1 SELECTs."""

    def __init__(self, inner: Any, counts: list[str]) -> None:
        self._inner = inner
        self._counts = counts

    def __getattr__(self, name: str) -> Any:
        return getattr(self._inner, name)

    def execute(self, query: Any, *args: Any, **kwargs: Any) -> Any:
        if STAGE1_MARKER in str(query):
            self._counts.append(str(query))
        return self._inner.execute(query, *args, **kwargs)


async def _build_chain_async(
    saver: AsyncPostgresSaver,
) -> tuple[list[dict], dict | None]:
    """Seed at step 1, then delta-only steps up to `CHAIN_LENGTH`.

    Returns the per-step head configs (index = step).
    """
    thread_id = str(uuid4())
    configs: list[dict] = []
    parent: dict | None = None
    for step in range(CHAIN_LENGTH):
        config: dict = {"configurable": {"thread_id": thread_id, "checkpoint_ns": ""}}
        if parent is not None:
            config["configurable"]["checkpoint_id"] = parent["configurable"][
                "checkpoint_id"
            ]
        cp: Checkpoint = empty_checkpoint()
        cp["id"] = str(uuid6(clock_seq=step))
        new_versions: dict[str, Any] = {}
        if step == 1:
            cp["channel_values"][CHANNEL] = _DeltaSnapshot([10, 20])
            cp["channel_versions"][CHANNEL] = "v1"
            new_versions[CHANNEL] = "v1"
        else:
            cp["channel_versions"][CHANNEL] = f"v{step}"
        parent = await saver.aput(
            config, cp, {"source": "loop", "step": step, "parents": {}}, new_versions
        )
        await saver.aput_writes(parent, [(CHANNEL, f"w{step}")], str(uuid4()))
        configs.append(parent)
    return configs, parent


def _build_chain_sync(saver: PostgresSaver) -> tuple[list[dict], dict | None]:
    thread_id = str(uuid4())
    configs: list[dict] = []
    parent: dict | None = None
    for step in range(CHAIN_LENGTH):
        config: dict = {"configurable": {"thread_id": thread_id, "checkpoint_ns": ""}}
        if parent is not None:
            config["configurable"]["checkpoint_id"] = parent["configurable"][
                "checkpoint_id"
            ]
        cp: Checkpoint = empty_checkpoint()
        cp["id"] = str(uuid6(clock_seq=step))
        new_versions: dict[str, Any] = {}
        if step == 1:
            cp["channel_values"][CHANNEL] = _DeltaSnapshot([10, 20])
            cp["channel_versions"][CHANNEL] = "v1"
            new_versions[CHANNEL] = "v1"
        else:
            cp["channel_versions"][CHANNEL] = f"v{step}"
        parent = saver.put(
            config, cp, {"source": "loop", "step": step, "parents": {}}, new_versions
        )
        saver.put_writes(parent, [(CHANNEL, f"w{step}")], str(uuid4()))
        configs.append(parent)
    return configs, parent


def _assert_result(entry: dict) -> None:
    seed = entry.get("seed")
    assert isinstance(seed, _DeltaSnapshot), f"expected a snapshot, got {seed!r}"
    assert seed.value == [10, 20]
    assert [w[2] for w in entry["writes"]] == ["w1", "w2"]


@pytest.mark.asyncio
async def test_async_walk_pages_only_at_or_below_the_target(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Targeting step 3 of a 20-step chain must not page the newer prefix.

    The seed sits two steps below the target, so with the pager bound to the
    target a single stage-1 page suffices — no matter how many newer
    checkpoints the thread has.
    """
    monkeypatch.setattr(aio_module, "_DELTA_PAGE_SIZE", PAGE_SIZE)
    async with AsyncPostgresSaver.from_conn_string(DEFAULT_URI) as saver:
        await saver.setup()
        configs, _ = await _build_chain_async(saver)
        target = configs[TARGET_STEP]

        stage1_calls: list[str] = []
        inner_cursor = saver._cursor

        @asynccontextmanager
        async def counting_cursor(*args: Any, **kwargs: Any):
            async with inner_cursor(*args, **kwargs) as cur:
                yield _CountingCursor(cur, stage1_calls)

        monkeypatch.setattr(saver, "_cursor", counting_cursor)

        result = await saver.aget_delta_channel_history(
            config=target, channels=[CHANNEL]
        )

        _assert_result(result[CHANNEL])
        assert len(stage1_calls) == 1, (
            f"expected one stage-1 page bounded at the target, got "
            f"{len(stage1_calls)} pages — the walk scanned the newer prefix"
        )


def test_sync_walk_pages_only_at_or_below_the_target(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Sync saver: same paging bound as the async test above."""
    monkeypatch.setattr(postgres_module, "_DELTA_PAGE_SIZE", PAGE_SIZE)
    with PostgresSaver.from_conn_string(DEFAULT_URI) as saver:
        saver.setup()
        configs, _ = _build_chain_sync(saver)
        target = configs[TARGET_STEP]

        stage1_calls: list[str] = []
        inner_cursor = saver._cursor

        @contextmanager
        def counting_cursor(*args: Any, **kwargs: Any):
            with inner_cursor(*args, **kwargs) as cur:
                yield _CountingCursor(cur, stage1_calls)

        monkeypatch.setattr(saver, "_cursor", counting_cursor)

        result = saver.get_delta_channel_history(config=target, channels=[CHANNEL])

        _assert_result(result[CHANNEL])
        assert len(stage1_calls) == 1, (
            f"expected one stage-1 page bounded at the target, got "
            f"{len(stage1_calls)} pages — the walk scanned the newer prefix"
        )
