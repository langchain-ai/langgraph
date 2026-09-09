"""Stage-1 pagination for `DeltaChannel` histories on Postgres.

`get_delta_channel_history` pages the `checkpoints` table newest-first in
chunks of `_DELTA_PAGE_SIZE`, starting from the head of the thread rather than
from the target checkpoint. The target's own row therefore only lands once
paging has reached back to it, which for a long thread can be several pages in.

Until then `parent_of` has no entry for it, and reading the walk cursor out of
that map records `None`, the same value that means "the target is a root". The
cursor is derived once, so a target older than the first page kept that `None`
forever: no seed, no writes, and the channel hydrated empty with no error.
See #8448.

These tests shrink the page size instead of writing 1024+ real checkpoints per
case. The behaviour under test is "the target is not on the first page", and
the page the target lands on is the only thing that decides it.
"""

from __future__ import annotations

from typing import Any
from uuid import uuid4

import pytest
from langgraph.checkpoint.base import (
    Checkpoint,
    DeltaChannelHistory,
    empty_checkpoint,
)
from langgraph.checkpoint.base.id import uuid6
from langgraph.checkpoint.serde.types import _DeltaSnapshot

from langgraph.checkpoint.postgres import PostgresSaver
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from langgraph.checkpoint.postgres.base import _DELTA_PAGE_SIZE
from tests.conftest import DEFAULT_URI

CHANNEL = "items"
STEPS = 8
SEED_STEP = 1
SEED_VALUE = [10, 20]
# Deep enough that the smaller page sizes below have to page past it.
TARGET_STEP = 4

# The real page size holds this whole thread on one page, so it is the control.
# The rest each leave the target off the first page: there are three
# checkpoints newer than `TARGET_STEP`, so any page size at or below 3 does it.
PAGE_SIZES = [_DELTA_PAGE_SIZE, 3, 2, 1]


def _step_args(
    thread_id: str, step: int, parent: dict | None
) -> tuple[dict, Checkpoint, dict[str, Any]]:
    """Return the `(config, checkpoint, new_versions)` triple for one step.

    Step `SEED_STEP` stores a snapshot for `CHANNEL`; the rest only bump the
    channel version, which is what a delta channel does between snapshots.
    """
    config: dict = {"configurable": {"thread_id": thread_id, "checkpoint_ns": ""}}
    if parent is not None:
        config["configurable"]["checkpoint_id"] = parent["configurable"][
            "checkpoint_id"
        ]
    checkpoint: Checkpoint = empty_checkpoint()
    checkpoint["id"] = str(uuid6(clock_seq=step))
    checkpoint["channel_versions"][CHANNEL] = f"v{step}"
    if step == SEED_STEP:
        checkpoint["channel_values"][CHANNEL] = _DeltaSnapshot(list(SEED_VALUE))
        return config, checkpoint, {CHANNEL: f"v{step}"}
    return config, checkpoint, {}


async def _abuild_chain(saver: AsyncPostgresSaver) -> list[dict]:
    """Write `STEPS` linked checkpoints, one write each; return their configs."""
    thread_id = str(uuid4())
    parent: dict | None = None
    configs: list[dict] = []
    for step in range(STEPS):
        config, checkpoint, new_versions = _step_args(thread_id, step, parent)
        parent = await saver.aput(
            config,
            checkpoint,
            {"source": "loop", "step": step, "parents": {}},
            new_versions,
        )
        await saver.aput_writes(parent, [(CHANNEL, f"w{step}")], str(uuid4()))
        configs.append(parent)
    return configs


def _build_chain(saver: PostgresSaver) -> list[dict]:
    """Sync twin of `_abuild_chain`."""
    thread_id = str(uuid4())
    parent: dict | None = None
    configs: list[dict] = []
    for step in range(STEPS):
        config, checkpoint, new_versions = _step_args(thread_id, step, parent)
        parent = saver.put(
            config,
            checkpoint,
            {"source": "loop", "step": step, "parents": {}},
            new_versions,
        )
        saver.put_writes(parent, [(CHANNEL, f"w{step}")], str(uuid4()))
        configs.append(parent)
    return configs


def _assert_history(entry: DeltaChannelHistory, page_size: int) -> None:
    """Check the walk from `TARGET_STEP` back to the snapshot at `SEED_STEP`.

    The chain is steps 3, 2 and 1: the target's own writes are pending for its
    next super-step and excluded, and the walk stops at the snapshot. Writes
    come back oldest first.
    """
    seed = entry.get("seed")
    assert isinstance(seed, _DeltaSnapshot), (
        f"page_size={page_size}: expected a snapshot seed, "
        f"got {entry.get('seed', '<missing>')!r}"
    )
    assert seed.value == SEED_VALUE
    assert [w[2] for w in entry["writes"]] == ["w1", "w2", "w3"], (
        f"page_size={page_size}: got {[w[2] for w in entry['writes']]}"
    )


@pytest.mark.parametrize("page_size", PAGE_SIZES)
async def test_async_target_older_than_the_first_page(
    page_size: int, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("langgraph.checkpoint.postgres.aio._DELTA_PAGE_SIZE", page_size)
    async with AsyncPostgresSaver.from_conn_string(DEFAULT_URI) as saver:
        await saver.setup()
        configs = await _abuild_chain(saver)
        result = await saver.aget_delta_channel_history(
            config=configs[TARGET_STEP], channels=[CHANNEL]
        )
        _assert_history(result[CHANNEL], page_size)


@pytest.mark.parametrize("page_size", PAGE_SIZES)
def test_sync_target_older_than_the_first_page(
    page_size: int, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("langgraph.checkpoint.postgres._DELTA_PAGE_SIZE", page_size)
    with PostgresSaver.from_conn_string(DEFAULT_URI) as saver:
        saver.setup()
        configs = _build_chain(saver)
        result = saver.get_delta_channel_history(
            config=configs[TARGET_STEP], channels=[CHANNEL]
        )
        _assert_history(result[CHANNEL], page_size)


async def test_root_target_has_no_history_and_still_terminates(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A root target is the case where a `None` cursor is the right answer.

    Its walk has nowhere to go, so it never seeds and collects none of the
    thread's writes, and paging has to run out on a short page rather than
    spin. Page size 1 gives one page per checkpoint, so the loop runs the
    length of the thread before stopping.
    """
    monkeypatch.setattr("langgraph.checkpoint.postgres.aio._DELTA_PAGE_SIZE", 1)
    async with AsyncPostgresSaver.from_conn_string(DEFAULT_URI) as saver:
        await saver.setup()
        configs = await _abuild_chain(saver)
        result = await saver.aget_delta_channel_history(
            config=configs[0], channels=[CHANNEL]
        )
        assert result[CHANNEL] == {"writes": []}
