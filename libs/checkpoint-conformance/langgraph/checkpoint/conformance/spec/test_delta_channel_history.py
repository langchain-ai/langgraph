"""DELTA_CHANNEL_HISTORY capability tests — aget_delta_channel_history contract."""

from __future__ import annotations

import traceback
from collections.abc import Callable
from uuid import uuid4

from langgraph.checkpoint.base import BaseCheckpointSaver, Checkpoint
from langgraph.checkpoint.base.id import uuid6
from langgraph.checkpoint.serde.types import _DeltaSnapshot

from langgraph.checkpoint.conformance.spec._delta_fixtures import build_delta_chain
from langgraph.checkpoint.conformance.test_utils import generate_metadata


async def test_history_returns_writes_oldest_first(
    saver: BaseCheckpointSaver,
) -> None:
    """Writes are returned oldest-to-newest."""
    tid = str(uuid4())
    # 5 steps: snapshot at 0, writes at 1,2,3,4.
    # Head is step 4. Walk starts at step 3 (parent of head).
    # Collects writes from steps 1,2,3 (between snapshot at 0 and head's parent).
    configs = await build_delta_chain(
        saver, thread_id=tid, channel="ch", snapshots_at_steps=[0], total_steps=5
    )
    head = configs[-1]
    result = await saver.aget_delta_channel_history(config=head, channels=["ch"])
    writes = result["ch"]["writes"]
    values = [w[2] for w in writes]
    assert values == [1, 2, 3], f"Expected [1,2,3], got {values}"


async def test_history_seed_is_nearest_snapshot(
    saver: BaseCheckpointSaver,
) -> None:
    """Seed is the value from the nearest ancestor with channel_values populated."""
    tid = str(uuid4())
    # 6 steps: snapshots at 0 and 3, writes at 1,2,4,5.
    # Head is step 5. Walk from step 4 backward stops at step 3 (snapshot).
    # Collects writes from step 4 only (between step 3 and head's parent step 4).
    configs = await build_delta_chain(
        saver,
        thread_id=tid,
        channel="ch",
        snapshots_at_steps=[0, 3],
        total_steps=6,
    )
    head = configs[-1]
    result = await saver.aget_delta_channel_history(config=head, channels=["ch"])
    assert "seed" in result["ch"], "Expected seed from snapshot at step 3"
    seed = result["ch"]["seed"]
    actual_value = seed.value if isinstance(seed, _DeltaSnapshot) else seed
    assert actual_value == 3, f"Expected seed value 3 (step 3), got {actual_value}"
    writes = result["ch"]["writes"]
    values = [w[2] for w in writes]
    assert values == [4], f"Expected [4], got {values}"


async def test_history_excludes_target_pending_writes(
    saver: BaseCheckpointSaver,
) -> None:
    """Target's own pending_writes are NOT included in the history."""
    tid = str(uuid4())
    configs = await build_delta_chain(
        saver, thread_id=tid, channel="ch", snapshots_at_steps=[0], total_steps=3
    )
    head = configs[-1]
    # Add writes directly to the head checkpoint
    await saver.aput_writes(head, [("ch", "extra")], str(uuid4()))
    result = await saver.aget_delta_channel_history(config=head, channels=["ch"])
    writes = result["ch"]["writes"]
    values = [w[2] for w in writes]
    assert "extra" not in values, f"Target's writes should be excluded, got {values}"


async def test_history_multi_channel(
    saver: BaseCheckpointSaver,
) -> None:
    """Multiple channels have independent walk termination."""
    tid = str(uuid4())
    configs: list = []
    parent_cfg = None

    for step in range(5):
        config = {"configurable": {"thread_id": tid, "checkpoint_ns": ""}}
        if parent_cfg:
            config["configurable"]["checkpoint_id"] = parent_cfg["configurable"][
                "checkpoint_id"
            ]
        cv: dict = {}
        cvs: dict = {}
        if step == 1:
            cv["a"] = _DeltaSnapshot("snap_a")
            cvs["a"] = step + 1
        if step == 3:
            cv["b"] = _DeltaSnapshot("snap_b")
            cvs["b"] = step + 1
        cp = Checkpoint(
            v=1,
            id=str(uuid6(clock_seq=-1)),
            ts="",
            channel_values=cv,
            channel_versions=cvs,
            versions_seen={},
            updated_channels=None,
        )
        parent_cfg = await saver.aput(config, cp, generate_metadata(step=step), cvs)
        configs.append(parent_cfg)
        await saver.aput_writes(parent_cfg, [("a", step), ("b", step)], str(uuid4()))

    head = configs[-1]
    result = await saver.aget_delta_channel_history(config=head, channels=["a", "b"])
    a_writes = [w[2] for w in result["a"]["writes"]]
    b_writes = [w[2] for w in result["b"]["writes"]]
    assert a_writes == [1, 2, 3], f"Expected a writes [1,2,3], got {a_writes}"
    assert b_writes == [3], f"Expected b writes [3], got {b_writes}"


async def test_history_empty_channels_returns_empty(
    saver: BaseCheckpointSaver,
) -> None:
    """Empty channels list returns empty mapping."""
    tid = str(uuid4())
    configs = await build_delta_chain(
        saver, thread_id=tid, channel="ch", snapshots_at_steps=[0], total_steps=3
    )
    result = await saver.aget_delta_channel_history(config=configs[-1], channels=[])
    assert result == {}


async def test_history_walk_to_root_no_seed(
    saver: BaseCheckpointSaver,
) -> None:
    """Walk reaches root without finding seed — no 'seed' key in result."""
    tid = str(uuid4())
    configs = await build_delta_chain(
        saver,
        thread_id=tid,
        channel="ch",
        snapshots_at_steps=[],
        total_steps=4,
    )
    head = configs[-1]
    result = await saver.aget_delta_channel_history(config=head, channels=["ch"])
    assert "seed" not in result["ch"], f"Expected no seed, got {result['ch']}"


async def test_history_migration_plain_value_as_seed(
    saver: BaseCheckpointSaver,
) -> None:
    """Pre-delta plain value in channel_values acts as seed (migration case).

    When a thread was originally using a regular channel (BinaryOperatorAggregate)
    and later switches to DeltaChannel, the old checkpoint has a plain value in
    channel_values[ch] (not a _DeltaSnapshot). The walk should treat it as the
    seed and terminate there.
    """
    tid = str(uuid4())
    configs: list = []
    parent_cfg = None

    for step in range(4):
        config = {"configurable": {"thread_id": tid, "checkpoint_ns": ""}}
        if parent_cfg:
            config["configurable"]["checkpoint_id"] = parent_cfg["configurable"][
                "checkpoint_id"
            ]
        cv: dict = {}
        cvs: dict = {}
        # Step 1: plain value (migration case — old checkpoint before delta)
        if step == 1:
            cv["ch"] = [10, 20, 30]
            cvs["ch"] = step + 1
        cp = Checkpoint(
            v=1,
            id=str(uuid6(clock_seq=-1)),
            ts="",
            channel_values=cv,
            channel_versions=cvs,
            versions_seen={},
            updated_channels=None,
        )
        parent_cfg = await saver.aput(config, cp, generate_metadata(step=step), cvs)
        configs.append(parent_cfg)
        if step != 1:
            await saver.aput_writes(parent_cfg, [("ch", step)], str(uuid4()))

    head = configs[-1]
    result = await saver.aget_delta_channel_history(config=head, channels=["ch"])
    # Seed should be the plain value from step 1
    assert "seed" in result["ch"], "Expected seed from migration plain value at step 1"
    seed = result["ch"]["seed"]
    assert seed == [10, 20, 30], f"Expected plain value [10,20,30], got {seed}"
    # Writes should be from step 2 only (between seed at step 1 and head's parent step 2)
    writes = result["ch"]["writes"]
    values = [w[2] for w in writes]
    assert values == [2], f"Expected [2], got {values}"


async def test_history_seed_ancestor_own_writes_are_replayed(
    saver: BaseCheckpointSaver,
) -> None:
    """Writes stored AT the seed ancestor must be included in `writes`.

    A stored value is the state ENTERING its checkpoint; the writes stored
    under that same checkpoint are what produced its child and are therefore
    NOT subsumed by it. Only writes at ancestors OLDER than the seed are
    subsumed, and the walk terminates before reaching them.

    This holds for plain-value seeds (migration from a pre-delta channel type)
    exactly as it does for `_DeltaSnapshot` seeds. Skipping the seed
    ancestor's own writes silently drops the first post-migration write.
    """
    tid = str(uuid4())
    configs: list = []
    parent_cfg = None

    # Each step's write is labelled by the role it plays, so the assertion
    # below reads directly rather than by step index.
    writes_by_step = {
        0: "older-than-seed",  # subsumed by the value stored at step 1
        1: "at-seed",  # the seed's own write, produced step 2
        2: "after-seed",  # delta-era write on the path to the head
        3: "pending-at-head",  # pending for the next step, never replayed
    }
    # Steps 0 and 1 store a plain value; 1 is the nearest, so it is the seed.
    values_by_step = {0: [10], 1: [10, 20]}

    for step in range(4):
        config = {"configurable": {"thread_id": tid, "checkpoint_ns": ""}}
        if parent_cfg:
            config["configurable"]["checkpoint_id"] = parent_cfg["configurable"][
                "checkpoint_id"
            ]
        cv: dict = {}
        cvs: dict = {}
        if step in values_by_step:
            cv["ch"] = values_by_step[step]
            cvs["ch"] = step + 1
        cp = Checkpoint(
            v=1,
            id=str(uuid6(clock_seq=-1)),
            ts="",
            channel_values=cv,
            channel_versions=cvs,
            versions_seen={},
            updated_channels=None,
        )
        parent_cfg = await saver.aput(config, cp, generate_metadata(step=step), cvs)
        configs.append(parent_cfg)
        await saver.aput_writes(
            parent_cfg, [("ch", writes_by_step[step])], str(uuid4())
        )

    head = configs[-1]
    result = await saver.aget_delta_channel_history(config=head, channels=["ch"])

    assert "seed" in result["ch"], "Expected seed from plain value at step 1"
    assert result["ch"]["seed"] == [10, 20], (
        f"Expected nearest plain value [10, 20], got {result['ch']['seed']}"
    )
    values = [w[2] for w in result["ch"]["writes"]]
    assert values == ["at-seed", "after-seed"], (
        f'Expected ["at-seed", "after-seed"], got {values}'
    )


ALL_DELTA_CHANNEL_HISTORY_TESTS = [
    test_history_returns_writes_oldest_first,
    test_history_seed_is_nearest_snapshot,
    test_history_excludes_target_pending_writes,
    test_history_multi_channel,
    test_history_empty_channels_returns_empty,
    test_history_walk_to_root_no_seed,
    test_history_migration_plain_value_as_seed,
    test_history_seed_ancestor_own_writes_are_replayed,
]


async def run_delta_channel_history_tests(
    saver: BaseCheckpointSaver,
    on_test_result: Callable[[str, str, bool, str | None], None] | None = None,
) -> tuple[int, int, list[str]]:
    """Run all delta_channel_history tests. Returns (passed, failed, failure_names)."""
    passed = 0
    failed = 0
    failures: list[str] = []
    for test_fn in ALL_DELTA_CHANNEL_HISTORY_TESTS:
        try:
            await test_fn(saver)
            passed += 1
            if on_test_result:
                on_test_result("delta_channel_history", test_fn.__name__, True, None)
        except Exception:
            failed += 1
            msg = f"{test_fn.__name__}: {traceback.format_exc()}"
            failures.append(msg)
            if on_test_result:
                on_test_result(
                    "delta_channel_history",
                    test_fn.__name__,
                    False,
                    traceback.format_exc(),
                )
    return passed, failed, failures
