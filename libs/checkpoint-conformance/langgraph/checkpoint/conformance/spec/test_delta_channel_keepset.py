"""DELTA_CHANNEL_KEEPSET capability tests — aget_delta_channel_keepset contract."""

from __future__ import annotations

import traceback
from collections.abc import Callable
from uuid import uuid4

from langgraph.checkpoint.base import BaseCheckpointSaver

from langgraph.checkpoint.conformance.spec._delta_fixtures import build_delta_chain


async def test_keepset_empty_channels_returns_target_only(
    saver: BaseCheckpointSaver,
) -> None:
    """Empty channels → keep-set is just {target_id}."""
    tid = str(uuid4())
    configs = await build_delta_chain(
        saver, thread_id=tid, channel="ch", snapshots_at_steps=[0], total_steps=4
    )
    head = configs[-1]
    keep = await saver.aget_delta_channel_keepset(config=head, channels=[])
    head_id = head["configurable"]["checkpoint_id"]
    assert keep == {head_id}, f"Expected only target, got {keep}"


async def test_keepset_snapshot_at_target(
    saver: BaseCheckpointSaver,
) -> None:
    """When target itself has a snapshot, keep-set is just {target_id}."""
    tid = str(uuid4())
    configs = await build_delta_chain(
        saver,
        thread_id=tid,
        channel="ch",
        snapshots_at_steps=[0, 3],
        total_steps=4,
    )
    head = configs[3]
    keep = await saver.aget_delta_channel_keepset(config=head, channels=["ch"])
    head_id = head["configurable"]["checkpoint_id"]
    assert keep == {head_id}, f"Snapshot at target should yield only target, got {keep}"


async def test_keepset_snapshot_n_back(
    saver: BaseCheckpointSaver,
) -> None:
    """Snapshot N steps back → target + intermediates + snapshot ancestor."""
    tid = str(uuid4())
    configs = await build_delta_chain(
        saver, thread_id=tid, channel="ch", snapshots_at_steps=[0], total_steps=5
    )
    head = configs[-1]
    keep = await saver.aget_delta_channel_keepset(config=head, channels=["ch"])
    expected_ids = {c["configurable"]["checkpoint_id"] for c in configs}
    assert keep == expected_ids, f"Expected all ancestors, got {keep}"


async def test_keepset_multi_channel_union(
    saver: BaseCheckpointSaver,
) -> None:
    """Multi-channel keep-set is the union (max chain per channel)."""
    tid = str(uuid4())
    from langgraph.checkpoint.base import Checkpoint
    from langgraph.checkpoint.base.id import uuid6
    from langgraph.checkpoint.serde.types import _DeltaSnapshot

    from langgraph.checkpoint.conformance.test_utils import generate_metadata

    configs: list = []
    parent_cfg = None
    for step in range(6):
        config = {"configurable": {"thread_id": tid, "checkpoint_ns": ""}}
        if parent_cfg:
            config["configurable"]["checkpoint_id"] = parent_cfg["configurable"][
                "checkpoint_id"
            ]
        cv: dict = {}
        cvs: dict = {}
        # Channel "a" has snapshot at step 3 (recent)
        if step == 3:
            cv["a"] = _DeltaSnapshot("snap_a")
            cvs["a"] = step + 1
        # Channel "b" has snapshot at step 1 (further back)
        if step == 1:
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

    head = configs[-1]
    keep = await saver.aget_delta_channel_keepset(config=head, channels=["a", "b"])
    # Union: b needs back to step 1, so steps 1..5 (all except step 0) plus head
    expected_ids = {c["configurable"]["checkpoint_id"] for c in configs[1:]}
    expected_ids.add(head["configurable"]["checkpoint_id"])
    assert keep == expected_ids, f"Expected union, got {keep} vs {expected_ids}"


async def test_keepset_walk_to_root(
    saver: BaseCheckpointSaver,
) -> None:
    """No snapshot anywhere → entire chain to root is in keep-set."""
    tid = str(uuid4())
    configs = await build_delta_chain(
        saver, thread_id=tid, channel="ch", snapshots_at_steps=[], total_steps=4
    )
    head = configs[-1]
    keep = await saver.aget_delta_channel_keepset(config=head, channels=["ch"])
    all_ids = {c["configurable"]["checkpoint_id"] for c in configs}
    assert keep == all_ids, f"Expected full chain, got {keep}"


async def test_keepset_deterministic(
    saver: BaseCheckpointSaver,
) -> None:
    """Same inputs return identical sets."""
    tid = str(uuid4())
    configs = await build_delta_chain(
        saver, thread_id=tid, channel="ch", snapshots_at_steps=[0], total_steps=5
    )
    head = configs[-1]
    keep1 = await saver.aget_delta_channel_keepset(config=head, channels=["ch"])
    keep2 = await saver.aget_delta_channel_keepset(config=head, channels=["ch"])
    assert keep1 == keep2


ALL_DELTA_CHANNEL_KEEPSET_TESTS = [
    test_keepset_empty_channels_returns_target_only,
    test_keepset_snapshot_at_target,
    test_keepset_snapshot_n_back,
    test_keepset_multi_channel_union,
    test_keepset_walk_to_root,
    test_keepset_deterministic,
]


async def run_delta_channel_keepset_tests(
    saver: BaseCheckpointSaver,
    on_test_result: Callable[[str, str, bool, str | None], None] | None = None,
) -> tuple[int, int, list[str]]:
    """Run all delta_channel_keepset tests. Returns (passed, failed, failure_names)."""
    passed = 0
    failed = 0
    failures: list[str] = []
    for test_fn in ALL_DELTA_CHANNEL_KEEPSET_TESTS:
        try:
            await test_fn(saver)
            passed += 1
            if on_test_result:
                on_test_result("delta_channel_keepset", test_fn.__name__, True, None)
        except Exception:
            failed += 1
            msg = f"{test_fn.__name__}: {traceback.format_exc()}"
            failures.append(msg)
            if on_test_result:
                on_test_result(
                    "delta_channel_keepset",
                    test_fn.__name__,
                    False,
                    traceback.format_exc(),
                )
    return passed, failed, failures
