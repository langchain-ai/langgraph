"""DELETE_THREAD capability tests — adelete_thread."""

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


async def test_delete_thread_removes_checkpoints(
    saver: BaseCheckpointSaver,
) -> None:
    """All checkpoints gone after delete."""
    tid = str(uuid4())
    parent_cfg = None
    for i in range(3):
        config = generate_config(tid)
        if parent_cfg:
            config["configurable"]["checkpoint_id"] = parent_cfg["configurable"][
                "checkpoint_id"
            ]
        cp = generate_checkpoint()
        parent_cfg = await saver.aput(config, cp, generate_metadata(step=i), {})

    # Pre-delete: verify data exists
    assert await saver.aget_tuple(generate_config(tid)) is not None, (
        "Pre-delete: checkpoint should exist"
    )

    await saver.adelete_thread(tid)

    tup = await saver.aget_tuple(generate_config(tid))
    assert tup is None

    results = []
    async for t in saver.alist(generate_config(tid)):
        results.append(t)
    assert len(results) == 0


async def test_delete_thread_removes_writes(saver: BaseCheckpointSaver) -> None:
    """Pending writes gone after delete."""
    tid = str(uuid4())
    config = generate_config(tid)
    cp = generate_checkpoint()
    stored = await saver.aput(config, cp, generate_metadata(), {})
    await saver.aput_writes(stored, [("ch", "val")], str(uuid4()))

    # Pre-delete: verify writes exist
    pre_tup = await saver.aget_tuple(generate_config(tid))
    assert pre_tup is not None, "Pre-delete: checkpoint should exist"
    assert pre_tup.pending_writes is not None and len(pre_tup.pending_writes) == 1, (
        f"Pre-delete: expected 1 write, got {len(pre_tup.pending_writes) if pre_tup.pending_writes else 0}"
    )

    await saver.adelete_thread(tid)

    tup = await saver.aget_tuple(generate_config(tid))
    assert tup is None


async def test_delete_thread_removes_all_namespaces(
    saver: BaseCheckpointSaver,
) -> None:
    """Root + child namespaces both removed."""
    tid = str(uuid4())

    for ns in ["", "child:1"]:
        cfg = generate_config(tid, checkpoint_ns=ns)
        cp = generate_checkpoint()
        await saver.aput(cfg, cp, generate_metadata(), {})

    # Pre-delete: verify each namespace has data
    for ns in ["", "child:1"]:
        pre = await saver.aget_tuple(generate_config(tid, checkpoint_ns=ns))
        assert pre is not None, f"Pre-delete: namespace '{ns}' should have data"

    await saver.adelete_thread(tid)

    for ns in ["", "child:1"]:
        tup = await saver.aget_tuple(generate_config(tid, checkpoint_ns=ns))
        assert tup is None


async def test_delete_thread_preserves_other_threads(
    saver: BaseCheckpointSaver,
) -> None:
    """Other threads untouched."""
    tid1, tid2 = str(uuid4()), str(uuid4())

    for tid in (tid1, tid2):
        cfg = generate_config(tid)
        cp = generate_checkpoint()
        await saver.aput(cfg, cp, generate_metadata(), {})

    await saver.adelete_thread(tid1)

    assert await saver.aget_tuple(generate_config(tid1)) is None
    assert await saver.aget_tuple(generate_config(tid2)) is not None


async def test_delete_thread_nonexistent_noop(
    saver: BaseCheckpointSaver,
) -> None:
    """No error for missing thread."""
    # Should not raise
    await saver.adelete_thread(str(uuid4()))


ALL_DELETE_THREAD_TESTS = [
    test_delete_thread_removes_checkpoints,
    test_delete_thread_removes_writes,
    test_delete_thread_removes_all_namespaces,
    test_delete_thread_preserves_other_threads,
    test_delete_thread_nonexistent_noop,
]


async def run_delete_thread_tests(
    saver: BaseCheckpointSaver,
    on_test_result: Callable[[str, str, bool, str | None], None] | None = None,
) -> tuple[int, int, list[str]]:
    """Run all delete_thread tests. Returns (passed, failed, failure_names)."""
    passed = 0
    failed = 0
    failures: list[str] = []
    for test_fn in ALL_DELETE_THREAD_TESTS:
        try:
            await test_fn(saver)
            passed += 1
            if on_test_result:
                on_test_result("delete_thread", test_fn.__name__, True, None)
        except Exception as e:
            failed += 1
            msg = f"{test_fn.__name__}: {e}"
            failures.append(msg)
            if on_test_result:
                on_test_result(
                    "delete_thread", test_fn.__name__, False, traceback.format_exc()
                )
    return passed, failed, failures
