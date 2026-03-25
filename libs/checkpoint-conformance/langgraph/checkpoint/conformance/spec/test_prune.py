"""PRUNE capability tests — aprune(strategy)."""

from __future__ import annotations

import traceback
from collections.abc import Callable
from uuid import uuid4

from langgraph.checkpoint.base import BaseCheckpointSaver

from langgraph.checkpoint.conformance.test_utils import (
    generate_checkpoint,
    generate_config,
    generate_metadata,
)


async def _setup_thread(saver: BaseCheckpointSaver, tid: str, n: int = 3) -> list[dict]:
    """Create n checkpoints on tid. Returns stored configs."""
    stored = []
    parent_cfg = None
    for i in range(n):
        config = generate_config(tid)
        if parent_cfg:
            config["configurable"]["checkpoint_id"] = parent_cfg["configurable"][
                "checkpoint_id"
            ]
        cp = generate_checkpoint()
        parent_cfg = await saver.aput(config, cp, generate_metadata(step=i), {})
        stored.append(parent_cfg)
    return stored


async def test_prune_keep_latest_single_thread(
    saver: BaseCheckpointSaver,
) -> None:
    """Only latest checkpoint survives."""
    tid = str(uuid4())
    configs = await _setup_thread(saver, tid, n=4)

    await saver.aprune([tid], strategy="keep_latest")
    results = []
    async for tup in saver.alist(generate_config(tid)):
        results.append(tup)

    assert len(results) == 1
    assert (
        results[0].config["configurable"]["checkpoint_id"]
        == configs[-1]["configurable"]["checkpoint_id"]
    )


async def test_prune_keep_latest_multiple_threads(
    saver: BaseCheckpointSaver,
) -> None:
    """Each thread keeps its latest."""
    tid1, tid2 = str(uuid4()), str(uuid4())
    c1 = await _setup_thread(saver, tid1, n=3)
    c2 = await _setup_thread(saver, tid2, n=2)

    await saver.aprune([tid1, tid2], strategy="keep_latest")
    for tid, expected_last in [(tid1, c1[-1]), (tid2, c2[-1])]:
        results = []
        async for tup in saver.alist(generate_config(tid)):
            results.append(tup)
        assert len(results) == 1
        assert (
            results[0].config["configurable"]["checkpoint_id"]
            == expected_last["configurable"]["checkpoint_id"]
        )


async def test_prune_keep_latest_across_namespaces(
    saver: BaseCheckpointSaver,
) -> None:
    """Latest per namespace kept."""
    tid = str(uuid4())

    # Root namespace: 3 checkpoints
    parent = None
    for i in range(3):
        cfg = generate_config(tid, checkpoint_ns="")
        if parent:
            cfg["configurable"]["checkpoint_id"] = parent["configurable"][
                "checkpoint_id"
            ]
        cp = generate_checkpoint()
        parent = await saver.aput(cfg, cp, generate_metadata(step=i), {})
    root_latest = parent

    # Child namespace: 2 checkpoints
    parent = None
    for i in range(2):
        cfg = generate_config(tid, checkpoint_ns="child:1")
        if parent:
            cfg["configurable"]["checkpoint_id"] = parent["configurable"][
                "checkpoint_id"
            ]
        cp = generate_checkpoint()
        parent = await saver.aput(cfg, cp, generate_metadata(step=i), {})
    child_latest = parent

    assert root_latest is not None
    assert child_latest is not None
    await saver.aprune([tid], strategy="keep_latest")
    for ns, expected in [("", root_latest), ("child:1", child_latest)]:
        results = []
        async for tup in saver.alist(generate_config(tid, checkpoint_ns=ns)):
            results.append(tup)
        assert len(results) == 1
        assert (
            results[0].config["configurable"]["checkpoint_id"]
            == expected["configurable"]["checkpoint_id"]
        )


async def test_prune_keep_latest_preserves_writes(
    saver: BaseCheckpointSaver,
) -> None:
    """Latest checkpoint's writes kept."""
    tid = str(uuid4())
    configs = await _setup_thread(saver, tid, n=3)

    # Add writes to the latest
    await saver.aput_writes(configs[-1], [("ch", "val")], str(uuid4()))

    await saver.aprune([tid], strategy="keep_latest")
    tup = await saver.aget_tuple(generate_config(tid))
    assert tup is not None
    assert tup.pending_writes is not None
    assert len(tup.pending_writes) == 1, (
        f"Expected 1 write, got {len(tup.pending_writes)}"
    )
    assert tup.pending_writes[0][1] == "ch", (
        f"channel mismatch: {tup.pending_writes[0][1]!r}"
    )
    assert tup.pending_writes[0][2] == "val", (
        f"value mismatch: {tup.pending_writes[0][2]!r}"
    )


async def test_prune_delete_all(saver: BaseCheckpointSaver) -> None:
    """delete_all strategy removes everything."""
    tid = str(uuid4())
    await _setup_thread(saver, tid, n=3)

    await saver.aprune([tid], strategy="delete")
    results = []
    async for tup in saver.alist(generate_config(tid)):
        results.append(tup)
    assert len(results) == 0


async def test_prune_preserves_other_threads(
    saver: BaseCheckpointSaver,
) -> None:
    """Unlisted threads untouched."""
    tid1, tid2 = str(uuid4()), str(uuid4())
    await _setup_thread(saver, tid1, n=3)
    await _setup_thread(saver, tid2, n=2)

    # Snapshot tid2 before prune
    pre_ids = []
    async for tup in saver.alist(generate_config(tid2)):
        pre_ids.append(tup.checkpoint["id"])

    await saver.aprune([tid1], strategy="keep_latest")
    # tid2 should be fully intact — same checkpoint IDs
    post_ids = []
    async for tup in saver.alist(generate_config(tid2)):
        post_ids.append(tup.checkpoint["id"])
    assert post_ids == pre_ids, f"tid2 changed: {pre_ids} -> {post_ids}"


async def test_prune_empty_list_noop(saver: BaseCheckpointSaver) -> None:
    """Empty thread_ids no error."""
    await saver.aprune([], strategy="keep_latest")


async def test_prune_nonexistent_noop(saver: BaseCheckpointSaver) -> None:
    """Missing threads no error."""
    await saver.aprune([str(uuid4())], strategy="keep_latest")


ALL_PRUNE_TESTS = [
    test_prune_keep_latest_single_thread,
    test_prune_keep_latest_multiple_threads,
    test_prune_keep_latest_across_namespaces,
    test_prune_keep_latest_preserves_writes,
    test_prune_delete_all,
    test_prune_preserves_other_threads,
    test_prune_empty_list_noop,
    test_prune_nonexistent_noop,
]


async def run_prune_tests(
    saver: BaseCheckpointSaver,
    on_test_result: Callable[[str, str, bool, str | None], None] | None = None,
) -> tuple[int, int, list[str]]:
    """Run all prune tests. Returns (passed, failed, failure_names)."""
    passed = 0
    failed = 0
    failures: list[str] = []
    for test_fn in ALL_PRUNE_TESTS:
        try:
            await test_fn(saver)
            passed += 1
            if on_test_result:
                on_test_result("prune", test_fn.__name__, True, None)
        except Exception as e:
            failed += 1
            msg = f"{test_fn.__name__}: {e}"
            failures.append(msg)
            if on_test_result:
                on_test_result("prune", test_fn.__name__, False, traceback.format_exc())
    return passed, failed, failures
