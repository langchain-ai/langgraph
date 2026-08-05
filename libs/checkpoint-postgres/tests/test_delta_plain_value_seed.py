"""Seed detection for `DeltaChannel` histories on Postgres.

`put` only leaves an inline marker in `channel_values` for `_DeltaSnapshot`
values. A plain value — what a thread migrated from a pre-delta channel type
leaves behind — is moved to `checkpoint_blobs` with no marker, so the stage-1
walk has to probe the blobs table to find it. See #8534.
"""

from __future__ import annotations

from typing import Any
from uuid import uuid4

import pytest
from langgraph.checkpoint.base import Checkpoint, empty_checkpoint
from langgraph.checkpoint.base.id import uuid6
from langgraph.checkpoint.serde.types import _DeltaSnapshot

from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from tests.conftest import DEFAULT_URI

CHANNEL = "items"


async def _build_chain(saver: AsyncPostgresSaver, seed_value: Any) -> tuple[str, dict]:
    """Store `seed_value` at step 1, then two steps that store nothing.

    Every step carries a write so the walk has something to collect.
    Returns `(thread_id, head_config)`.
    """
    thread_id = str(uuid4())
    parent: dict | None = None
    for step in range(4):
        config: dict = {"configurable": {"thread_id": thread_id, "checkpoint_ns": ""}}
        if parent is not None:
            config["configurable"]["checkpoint_id"] = parent["configurable"][
                "checkpoint_id"
            ]
        cp: Checkpoint = empty_checkpoint()
        cp["id"] = str(uuid6(clock_seq=step))
        new_versions: dict[str, Any] = {}
        if step == 1:
            cp["channel_values"][CHANNEL] = seed_value
            cp["channel_versions"][CHANNEL] = "v1"
            new_versions[CHANNEL] = "v1"
        else:
            cp["channel_versions"][CHANNEL] = f"v{step}"
        parent = await saver.aput(
            config, cp, {"source": "loop", "step": step, "parents": {}}, new_versions
        )
        await saver.aput_writes(parent, [(CHANNEL, f"w{step}")], str(uuid4()))
    assert parent is not None
    return thread_id, parent


@pytest.mark.asyncio
async def test_plain_value_seed_is_found() -> None:
    """A pre-delta plain value must be located as the seed.

    Before #8534 the walk ran to the root and returned no seed, which happens
    to reconstruct correctly for additive reducers while costing an
    O(thread length) replay on every read.
    """
    async with AsyncPostgresSaver.from_conn_string(DEFAULT_URI) as saver:
        await saver.setup()
        _, head = await _build_chain(saver, [10, 20])

        result = await saver.aget_delta_channel_history(config=head, channels=[CHANNEL])
        entry = result[CHANNEL]

        assert entry.get("seed") == [10, 20], (
            f"expected the plain value as seed, got {entry.get('seed', '<missing>')}"
        )
        # Only the writes between the seed and the head's parent replay: step 1
        # (the seed's own) and step 2. Step 0 is older than the seed, step 3 is
        # pending at the head.
        assert [w[2] for w in entry["writes"]] == ["w1", "w2"]


@pytest.mark.asyncio
async def test_delta_snapshot_seed_is_found() -> None:
    """The `_DeltaSnapshot` path keeps working, so both seed kinds agree."""
    async with AsyncPostgresSaver.from_conn_string(DEFAULT_URI) as saver:
        await saver.setup()
        _, head = await _build_chain(saver, _DeltaSnapshot([10, 20]))

        result = await saver.aget_delta_channel_history(config=head, channels=[CHANNEL])
        entry = result[CHANNEL]

        seed = entry.get("seed")
        assert isinstance(seed, _DeltaSnapshot), f"expected a snapshot, got {seed!r}"
        assert seed.value == [10, 20]
        assert [w[2] for w in entry["writes"]] == ["w1", "w2"]


@pytest.mark.asyncio
async def test_version_bump_without_a_value_does_not_hide_an_older_seed() -> None:
    """A delta-era step bumps `channel_versions` without storing a value, so no
    blob exists for that version. The probe must report no seed there and keep
    walking rather than stopping at a version it cannot resolve.

    Step 0 holds the real value; step 1 bumps the version with nothing stored.
    Walking back from the head has to pass step 1 to reach step 0.
    """
    async with AsyncPostgresSaver.from_conn_string(DEFAULT_URI) as saver:
        await saver.setup()
        thread_id = str(uuid4())
        parent: dict | None = None
        for step in range(4):
            config: dict = {
                "configurable": {"thread_id": thread_id, "checkpoint_ns": ""}
            }
            if parent is not None:
                config["configurable"]["checkpoint_id"] = parent["configurable"][
                    "checkpoint_id"
                ]
            cp: Checkpoint = empty_checkpoint()
            cp["id"] = str(uuid6(clock_seq=step))
            new_versions: dict[str, Any] = {}
            if step == 0:
                cp["channel_values"][CHANNEL] = [10, 20]
                cp["channel_versions"][CHANNEL] = "v0"
                new_versions[CHANNEL] = "v0"
            elif step == 1:
                # Version bumped, value absent -> stored as an "empty" blob.
                cp["channel_versions"][CHANNEL] = "v1"
                new_versions[CHANNEL] = "v1"
            else:
                cp["channel_versions"][CHANNEL] = "v1"
            parent = await saver.aput(
                config,
                cp,
                {"source": "loop", "step": step, "parents": {}},
                new_versions,
            )
            await saver.aput_writes(parent, [(CHANNEL, f"w{step}")], str(uuid4()))
        assert parent is not None

        result = await saver.aget_delta_channel_history(
            config=parent, channels=[CHANNEL]
        )
        entry = result[CHANNEL]

        assert entry.get("seed") == [10, 20], (
            "the walk stopped at the empty blob instead of reaching the real "
            f"value at step 0; got {entry.get('seed', '<missing>')}"
        )
        assert [w[2] for w in entry["writes"]] == ["w0", "w1", "w2"]
