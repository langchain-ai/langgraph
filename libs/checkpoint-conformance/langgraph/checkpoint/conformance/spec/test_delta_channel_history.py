"""DELTA_CHANNEL_HISTORY capability tests — aget_delta_channel_history contract."""

from __future__ import annotations

import traceback
from collections.abc import Callable
from uuid import uuid4

from langgraph.checkpoint.base import BaseCheckpointSaver

from langgraph.checkpoint.conformance.spec._delta_fixtures import build_delta_chain


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
    from langgraph.checkpoint.serde.types import _DeltaSnapshot

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
    from langgraph.checkpoint.base import Checkpoint
    from langgraph.checkpoint.base.id import uuid6
    from langgraph.checkpoint.serde.types import _DeltaSnapshot

    from langgraph.checkpoint.conformance.test_utils import generate_metadata

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
    from langgraph.checkpoint.base import Checkpoint
    from langgraph.checkpoint.base.id import uuid6

    from langgraph.checkpoint.conformance.test_utils import generate_metadata

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


# Task ids used by the ordering tests below. `build_delta_chain` tags its own
# writes with a `uuid4`, whose hex digits are all <= "f", so "aaaa..." sorts
# before every fixture task id and "zzzz..." sorts after every one of them.
# That makes the expected order fully determined rather than dependent on which
# uuid4 the fixture happened to draw.
TASK_ID_SORTS_FIRST = "aaaaaaaa-0000-0000-0000-000000000000"
TASK_ID_SORTS_LAST = "zzzzzzzz-0000-0000-0000-000000000000"


async def test_history_orders_parallel_writes_by_task_path(
    saver: BaseCheckpointSaver,
) -> None:
    """Writes from several tasks in one super-step replay in task_path order.

    Live execution sorts a super-step's tasks by `task_path_str(path[:3])`
    before applying their values, so replay has to recover that order rather
    than `task_id` order — `task_id` is a hash of the path, so the two
    disagree, and reducers are only required to be batching-invariant, not
    order-invariant.

    The two task_ids are assigned so they sort in the *opposite* order from
    their task_paths. A saver ordering by `(task_id, idx)` therefore returns
    these writes reversed, rather than passing by happening to agree.
    """
    configs = await build_delta_chain(
        saver,
        thread_id=str(uuid4()),
        channel="ch",
        snapshots_at_steps=[0],
        total_steps=3,
    )
    # The chain is: step 0 snapshot (seed), step 1 write, step 2 write.
    # `aget_delta_channel_history` walks from the head's parent back to the
    # seed, so it collects step 1's writes only — step 0 terminates the walk
    # and step 2 is the head, whose own writes are pending for the next
    # super-step and excluded. So step 1 is where these writes have to go.
    step_1, head = configs[1], configs[2]
    await saver.aput_writes(
        step_1, [("ch", "second")], TASK_ID_SORTS_FIRST, "~pull, 02"
    )
    await saver.aput_writes(step_1, [("ch", "first")], TASK_ID_SORTS_LAST, "~pull, 01")

    result = await saver.aget_delta_channel_history(config=head, channels=["ch"])
    values = [w[2] for w in result["ch"]["writes"]]
    # 1 is the fixture's own write at step 1. It carries no task_path, so it
    # sorts ahead of both writes added above.
    assert values == [1, "first", "second"], (
        f"Expected task_path order [1, 'first', 'second'], got {values}. "
        "Ordering by (task_id, idx) alone yields [1, 'second', 'first']."
    )


async def test_history_orders_pathless_writes_first(
    saver: BaseCheckpointSaver,
) -> None:
    """Writes stored without a task_path sort ahead of path-carrying ones.

    A task-less write (graph input) persists `task_path=""`, as does any row
    written before a saver recorded the column. `""` precedes every
    `task_path_str` output because that function prefixes tuples with `~`, so
    those writes replay first — where live execution applies graph input.
    """
    configs = await build_delta_chain(
        saver,
        thread_id=str(uuid4()),
        channel="ch",
        snapshots_at_steps=[0],
        total_steps=3,
    )
    # Same chain shape as above: step 1 is the only step the walk collects.
    step_1, head = configs[1], configs[2]
    # Committed in the opposite order to the one they must replay in, so the
    # assertion cannot pass on insertion order alone.
    await saver.aput_writes(
        step_1, [("ch", "from_node")], TASK_ID_SORTS_FIRST, "~pull, a"
    )
    await saver.aput_writes(step_1, [("ch", "from_input")], TASK_ID_SORTS_LAST)

    result = await saver.aget_delta_channel_history(config=head, channels=["ch"])
    values = [w[2] for w in result["ch"]["writes"]]
    # Both 1 (the fixture's write) and "from_input" are pathless, so they sort
    # by task_id among themselves and both precede the path-carrying write.
    assert values == [1, "from_input", "from_node"], (
        f"Expected pathless writes first, got {values}"
    )


ALL_DELTA_CHANNEL_HISTORY_TESTS = [
    test_history_returns_writes_oldest_first,
    test_history_seed_is_nearest_snapshot,
    test_history_excludes_target_pending_writes,
    test_history_multi_channel,
    test_history_empty_channels_returns_empty,
    test_history_walk_to_root_no_seed,
    test_history_migration_plain_value_as_seed,
    test_history_orders_parallel_writes_by_task_path,
    test_history_orders_pathless_writes_first,
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
