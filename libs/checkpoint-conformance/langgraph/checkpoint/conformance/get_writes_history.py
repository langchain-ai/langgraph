"""Conformance helper for `BaseCheckpointSaver.get_writes_history`.

A reusable battery of correctness checks any third-party saver
implementation can run against its `BaseCheckpointSaver.get_writes_history`
(and `aget_writes_history`) implementation. Use it from a test like:

    @pytest.mark.asyncio
    async def test_my_saver_history():
        await validate_get_writes_history(my_saver_factory)

Where `my_saver_factory` is an async context manager that yields a fresh
`BaseCheckpointSaver`, e.g. `RegisteredCheckpointer.create` or any
equivalent zero-arg factory.

The helper builds checkpoint chains directly via the saver's public
`aput` / `aput_writes` APIs and asserts on the shape of
`aget_writes_history`. It does NOT depend on `langgraph` core, the
`DeltaChannel` runtime, or any sentinel/snapshot wrapper type — the
walk's terminator contract is purely "is there any value present at
``channel_values[ch]``?" and is exercised here with plain values.
"""

from __future__ import annotations

from collections.abc import Callable
from contextlib import AbstractAsyncContextManager
from typing import Any
from uuid import uuid4

from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.base import (
    BaseCheckpointSaver,
    ChannelVersions,
    Checkpoint,
    CheckpointMetadata,
)

from langgraph.checkpoint.conformance.test_utils import (
    generate_checkpoint,
    generate_config,
    generate_metadata,
)

# A factory that, when invoked, yields a fresh saver in an async context.
# Matches the shape of `RegisteredCheckpointer.create` from this package.
SaverFactory = Callable[[], AbstractAsyncContextManager[BaseCheckpointSaver]]


async def _put_chain(
    saver: BaseCheckpointSaver,
    *,
    thread_id: str,
    steps: list[dict[str, Any]],
    namespace: str = "",
) -> list[RunnableConfig]:
    """Put a parent-linked chain of checkpoints.

    Each entry in `steps` describes one super-step:

        {
            "channel_values": {ch: value, ...},   # populates the blob layer
            "channel_versions": {ch: int, ...},   # required when channel_values set
            "writes": [(channel, value), ...],    # written via aput_writes
            "task_id": str,                       # optional, default uuid
        }

    Returns the saved configs for each step (in chain order). Writes are
    attached to the checkpoint they were stored at — the saver returns
    them as `pending_writes` on `get_tuple`.
    """
    stored: list[RunnableConfig] = []
    parent: RunnableConfig | None = None
    for step in steps:
        cp: Checkpoint = generate_checkpoint(
            channel_values=step.get("channel_values", {}),
            channel_versions=step.get("channel_versions", {}),
        )
        config = generate_config(thread_id, checkpoint_ns=namespace)
        if parent is not None:
            config["configurable"]["checkpoint_id"] = parent["configurable"][
                "checkpoint_id"
            ]
        md: CheckpointMetadata = generate_metadata(step=len(stored))
        new_versions: ChannelVersions = step.get(
            "new_versions", step.get("channel_versions", {})
        )
        cur = await saver.aput(config, cp, md, new_versions)
        for ch, value in step.get("writes", []):
            await saver.aput_writes(
                cur, [(ch, value)], step.get("task_id", str(uuid4()))
            )
        stored.append(cur)
        parent = cur
    return stored


def _flatten_values(writes: list[tuple[str, str, Any]]) -> list[Any]:
    """Concatenate write payloads (treating list payloads as elementwise)."""
    out: list[Any] = []
    for _task_id, _channel, value in writes:
        if isinstance(value, list):
            out.extend(value)
        else:
            out.append(value)
    return out


# ---------------------------------------------------------------------------
# Individual scenarios — each takes a factory that yields a fresh saver.
# ---------------------------------------------------------------------------


async def _scenario_empty_channels_returns_empty(factory: SaverFactory) -> None:
    """Empty `channels` short-circuits to `{}`."""
    async with factory() as saver:
        config = generate_config(str(uuid4()))
        # Empty channels must return an empty mapping regardless of whether
        # the target checkpoint exists.
        result = await saver.aget_writes_history(config, [])
        assert result == {}, f"empty channels should return {{}}, got {result!r}"


async def _scenario_root_only_thread(factory: SaverFactory) -> None:
    """A root-only thread (no parent) → empty writes, no seed."""
    async with factory() as saver:
        tid = str(uuid4())
        stored = await _put_chain(
            saver,
            thread_id=tid,
            steps=[
                {
                    "channel_values": {},
                    "channel_versions": {},
                    "writes": [("messages", {"content": "pending"})],
                }
            ],
        )
        result = await saver.aget_writes_history(stored[0], ["messages"])
        assert "messages" in result, f"missing channel entry: {result!r}"
        entry = result["messages"]
        # Pending writes at the target itself are not part of history.
        assert entry["writes"] == [], (
            f"root-only walk should produce no writes, got {entry['writes']!r}"
        )
        assert "seed" not in entry, (
            f"root walk reached without a seed → 'seed' key must be absent, got {entry!r}"
        )


async def _scenario_multi_checkpoint_oldest_to_newest(factory: SaverFactory) -> None:
    """Multi-checkpoint thread with deltas → writes oldest→newest, no seed."""
    async with factory() as saver:
        tid = str(uuid4())
        # Three checkpoints, each carrying a write on the prior step. None
        # of them write a snapshot blob, so the walk should reach root with
        # no seed and the writes should accumulate v0, v1, v2.
        stored = await _put_chain(
            saver,
            thread_id=tid,
            steps=[
                {"writes": [("items", "v0")]},
                {"writes": [("items", "v1")]},
                {"writes": [("items", "v2")]},
                # Trailing checkpoint with no writes — ensures the walk has
                # to step back through three ancestors to collect the writes.
                {"writes": []},
            ],
        )
        leaf_cfg = stored[-1]
        result = await saver.aget_writes_history(leaf_cfg, ["items"])
        entry = result["items"]
        assert "seed" not in entry, (
            f"no snapshot blob → seed must be absent, got {entry!r}"
        )
        values = _flatten_values(entry["writes"])
        assert values == ["v0", "v1", "v2"], (
            f"writes not in oldest→newest order: got {values!r}"
        )


async def _scenario_fork_only_on_path_ancestors(factory: SaverFactory) -> None:
    """Forked thread: walk follows parent_config, not sibling branches.

    Build:

        root --> A1 --> A2          (branch A: writes "a0", "a1", "a2")
            \\--> B1 --> B2         (branch B: writes "b0", "b1", "b2")

    Walking from A2 must see only "a*" writes; from B2, only "b*".
    """
    async with factory() as saver:
        tid_a = f"{uuid4()}-a"
        tid_b = f"{uuid4()}-b"

        # We model the "fork" as two parallel threads sharing structure.
        # The contract of `get_writes_history` is to follow `parent_config`,
        # which is naturally thread-scoped — so two threads suffice to
        # exercise on-path-only behaviour.
        a_chain = await _put_chain(
            saver,
            thread_id=tid_a,
            steps=[
                {"writes": [("items", "a0")]},
                {"writes": [("items", "a1")]},
                {"writes": [("items", "a2")]},
                {"writes": []},
            ],
        )
        b_chain = await _put_chain(
            saver,
            thread_id=tid_b,
            steps=[
                {"writes": [("items", "b0")]},
                {"writes": [("items", "b1")]},
                {"writes": [("items", "b2")]},
                {"writes": []},
            ],
        )

        a_result = await saver.aget_writes_history(a_chain[-1], ["items"])
        b_result = await saver.aget_writes_history(b_chain[-1], ["items"])

        a_values = _flatten_values(a_result["items"]["writes"])
        b_values = _flatten_values(b_result["items"]["writes"])

        assert a_values == ["a0", "a1", "a2"], (
            f"branch A leaked sibling writes: {a_values!r}"
        )
        assert b_values == ["b0", "b1", "b2"], (
            f"branch B leaked sibling writes: {b_values!r}"
        )


async def _scenario_multi_channel_independent_seeds(factory: SaverFactory) -> None:
    """Two channels, seeds at different depths → independent termination.

    Layout (oldest first):

        cp0 (seed value for ch_b)
        cp1 (writes "b1")
        cp2 (seed value for ch_a; writes "b2")
        cp3 (writes "a3", "b3")
        cp4 (leaf, no writes)

    Walking from cp4 with ["ch_a", "ch_b"]:
      * `ch_a` terminates at cp2 (value present), accumulates "a3".
      * `ch_b` keeps walking past cp2 (its blob is for `ch_a` only) and
        terminates at cp0; accumulates "b1", "b2", "b3".

    The walk's terminator contract is "any present value", not a
    sentinel match — so plain values are sufficient to exercise it.
    """
    async with factory() as saver:
        tid = str(uuid4())
        seed_a = ["seed-a"]
        seed_b = ["seed-b"]
        chain = await _put_chain(
            saver,
            thread_id=tid,
            steps=[
                {
                    "channel_values": {"ch_b": seed_b},
                    "channel_versions": {"ch_b": 1},
                    "new_versions": {"ch_b": 1},
                    "writes": [],
                },
                {"writes": [("ch_b", "b1")]},
                {
                    "channel_values": {"ch_a": seed_a},
                    "channel_versions": {"ch_a": 1},
                    "new_versions": {"ch_a": 1},
                    "writes": [("ch_b", "b2")],
                },
                {"writes": [("ch_a", "a3"), ("ch_b", "b3")]},
                {"writes": []},
            ],
        )
        leaf_cfg = chain[-1]
        result = await saver.aget_writes_history(leaf_cfg, ["ch_a", "ch_b"])

        assert "ch_a" in result and "ch_b" in result, (
            f"missing channel(s) in result: {set(result)!r}"
        )

        a_entry = result["ch_a"]
        assert "seed" in a_entry, f"ch_a should be seeded at cp2, got {a_entry!r}"
        assert a_entry["seed"] == seed_a, (
            f"ch_a seed mismatch: expected {seed_a!r}, got {a_entry['seed']!r}"
        )
        a_values = _flatten_values(a_entry["writes"])
        assert a_values == ["a3"], (
            f"ch_a writes should stop at cp2's seed: got {a_values!r}"
        )

        b_entry = result["ch_b"]
        assert "seed" in b_entry, f"ch_b should be seeded at cp0, got {b_entry!r}"
        assert b_entry["seed"] == seed_b, (
            f"ch_b seed mismatch: expected {seed_b!r}, got {b_entry['seed']!r}"
        )
        b_values = _flatten_values(b_entry["writes"])
        assert b_values == ["b1", "b2", "b3"], (
            f"ch_b writes (oldest→newest across cp1..cp3): got {b_values!r}"
        )


async def _scenario_pre_migration_plain_value_terminates(
    factory: SaverFactory,
) -> None:
    """Plain (non-DeltaSnapshot) ancestor value terminates the walk.

    Pre-migration threads stored the full reduced state directly in
    `channel_values[ch]` (no sentinel, no snapshot wrapper). The new
    walker treats *any* present value as a terminator, so pre-migration
    threads continue to work after migration to DeltaChannel.
    """
    async with factory() as saver:
        tid = str(uuid4())
        seed_value = ["legacy", "items"]
        chain = await _put_chain(
            saver,
            thread_id=tid,
            steps=[
                # Pre-migration ancestor: full plain-list value in channel_values.
                {
                    "channel_values": {"items": seed_value},
                    "channel_versions": {"items": 1},
                    "new_versions": {"items": 1},
                    "writes": [],
                },
                # Post-migration deltas accumulate on top.
                {"writes": [("items", ["delta1"])]},
                {"writes": [("items", ["delta2"])]},
                {"writes": []},
            ],
        )
        leaf_cfg = chain[-1]
        result = await saver.aget_writes_history(leaf_cfg, ["items"])
        entry = result["items"]
        assert "seed" in entry, (
            f"plain value at cp0 should terminate the walk, got {entry!r}"
        )
        assert entry["seed"] == seed_value, (
            f"seed should be the plain pre-migration value, got {entry['seed']!r}"
        )
        # Writes from cp1 and cp2 should be accumulated oldest→newest.
        values = _flatten_values(entry["writes"])
        assert values == ["delta1", "delta2"], (
            f"writes should be cp1→cp2 in order: got {values!r}"
        )


_SCENARIOS: list[tuple[str, Callable[[SaverFactory], Any]]] = [
    ("empty_channels_returns_empty", _scenario_empty_channels_returns_empty),
    ("root_only_thread", _scenario_root_only_thread),
    ("multi_checkpoint_oldest_to_newest", _scenario_multi_checkpoint_oldest_to_newest),
    ("fork_only_on_path_ancestors", _scenario_fork_only_on_path_ancestors),
    ("multi_channel_independent_seeds", _scenario_multi_channel_independent_seeds),
    (
        "pre_migration_plain_value_terminates",
        _scenario_pre_migration_plain_value_terminates,
    ),
]


async def validate_get_writes_history(saver_factory: SaverFactory) -> None:
    """Run the full battery of `get_writes_history` correctness scenarios.

    Args:
        saver_factory: A zero-arg callable returning an async context
            manager that yields a fresh `BaseCheckpointSaver`. Each
            scenario calls the factory to enter a new context; scenarios
            may share an instance if the factory chooses (each scenario
            uses disjoint thread IDs so cross-talk is harmless).

    Raises:
        AssertionError: when any scenario fails. The message identifies
            which scenario raised so a failing third-party saver can
            triage quickly.
    """
    for name, scenario in _SCENARIOS:
        try:
            await scenario(saver_factory)
        except AssertionError as e:
            raise AssertionError(f"scenario {name!r} failed: {e}") from e
