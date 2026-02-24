"""DELETE_FOR_RUNS capability tests — adelete_for_runs."""

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


async def _put_with_run_id(
    saver: BaseCheckpointSaver,
    tid: str,
    run_id: str,
    *,
    checkpoint_ns: str = "",
    parent_config: dict | None = None,
) -> dict:
    """Put a checkpoint with a run_id in metadata, return stored config."""
    config = generate_config(tid, checkpoint_ns=checkpoint_ns)
    if parent_config:
        config["configurable"]["checkpoint_id"] = parent_config["configurable"][
            "checkpoint_id"
        ]
    cp = generate_checkpoint()
    md = generate_metadata(run_id=run_id)
    return await saver.aput(config, cp, md, {})


async def test_delete_for_runs_single(saver: BaseCheckpointSaver) -> None:
    """One run_id removed."""
    tid = str(uuid4())
    run1, run2 = str(uuid4()), str(uuid4())

    stored1 = await _put_with_run_id(saver, tid, run1)
    await _put_with_run_id(saver, tid, run2, parent_config=stored1)

    # Pre-delete: verify both runs exist
    pre_results = []
    async for tup in saver.alist(generate_config(tid)):
        pre_results.append(tup)
    pre_run_ids = {t.metadata.get("run_id") for t in pre_results}
    assert run1 in pre_run_ids, "Pre-delete: run1 should exist"
    assert run2 in pre_run_ids, "Pre-delete: run2 should exist"

    await saver.adelete_for_runs([run1])
    # run1's checkpoint should be gone; run2 should remain
    results = []
    async for tup in saver.alist(generate_config(tid)):
        results.append(tup)

    run_ids = {t.metadata.get("run_id") for t in results}
    assert run1 not in run_ids
    assert run2 in run_ids


async def test_delete_for_runs_multiple(saver: BaseCheckpointSaver) -> None:
    """List of run_ids removed."""
    tid = str(uuid4())
    run1, run2, run3 = str(uuid4()), str(uuid4()), str(uuid4())

    s1 = await _put_with_run_id(saver, tid, run1)
    s2 = await _put_with_run_id(saver, tid, run2, parent_config=s1)
    await _put_with_run_id(saver, tid, run3, parent_config=s2)

    # Pre-delete: verify all 3 runs exist
    pre_results = []
    async for tup in saver.alist(generate_config(tid)):
        pre_results.append(tup)
    pre_run_ids = {t.metadata.get("run_id") for t in pre_results}
    assert run1 in pre_run_ids, "Pre-delete: run1 should exist"
    assert run2 in pre_run_ids, "Pre-delete: run2 should exist"
    assert run3 in pre_run_ids, "Pre-delete: run3 should exist"

    await saver.adelete_for_runs([run1, run2])
    results = []
    async for tup in saver.alist(generate_config(tid)):
        results.append(tup)

    run_ids = {t.metadata.get("run_id") for t in results}
    assert run1 not in run_ids
    assert run2 not in run_ids
    assert run3 in run_ids


async def test_delete_for_runs_preserves_other_runs(
    saver: BaseCheckpointSaver,
) -> None:
    """Unrelated runs untouched."""
    tid = str(uuid4())
    run_keep = str(uuid4())
    run_delete = str(uuid4())

    await _put_with_run_id(saver, tid, run_keep)
    await _put_with_run_id(saver, tid, run_delete)

    # Pre-delete: verify both runs exist
    pre_results = []
    async for tup in saver.alist(generate_config(tid)):
        pre_results.append(tup)
    pre_run_ids = {t.metadata.get("run_id") for t in pre_results}
    assert run_keep in pre_run_ids, "Pre-delete: run_keep should exist"
    assert run_delete in pre_run_ids, "Pre-delete: run_delete should exist"

    await saver.adelete_for_runs([run_delete])
    results = []
    async for tup in saver.alist(generate_config(tid)):
        results.append(tup)

    run_ids = {t.metadata.get("run_id") for t in results}
    assert run_keep in run_ids


async def test_delete_for_runs_removes_writes(
    saver: BaseCheckpointSaver,
) -> None:
    """Associated writes cleaned up."""
    tid = str(uuid4())
    run1 = str(uuid4())

    stored = await _put_with_run_id(saver, tid, run1)
    await saver.aput_writes(stored, [("ch", "val")], str(uuid4()))

    # Pre-delete: verify writes exist
    pre_tup = await saver.aget_tuple(stored)
    assert pre_tup is not None, "Pre-delete: checkpoint should exist"
    assert pre_tup.pending_writes is not None and len(pre_tup.pending_writes) == 1, (
        f"Pre-delete: expected 1 write, got {len(pre_tup.pending_writes) if pre_tup.pending_writes else 0}"
    )

    await saver.adelete_for_runs([run1])
    # The checkpoint (and its writes) should be gone
    tup = await saver.aget_tuple(stored)
    assert tup is None


async def test_delete_for_runs_empty_list_noop(
    saver: BaseCheckpointSaver,
) -> None:
    """Empty list no error."""
    await saver.adelete_for_runs([])


async def test_delete_for_runs_nonexistent_noop(
    saver: BaseCheckpointSaver,
) -> None:
    """Missing run_ids no error."""
    await saver.adelete_for_runs([str(uuid4())])


async def test_delete_for_runs_across_namespaces(
    saver: BaseCheckpointSaver,
) -> None:
    """All namespaces cleaned."""
    tid = str(uuid4())
    run1 = str(uuid4())

    await _put_with_run_id(saver, tid, run1, checkpoint_ns="")
    await _put_with_run_id(saver, tid, run1, checkpoint_ns="child:1")

    # Pre-delete: verify run1 present in both namespaces
    for ns in ["", "child:1"]:
        pre_results = []
        async for tup in saver.alist(generate_config(tid, checkpoint_ns=ns)):
            pre_results.append(tup)
        pre_run_ids = {t.metadata.get("run_id") for t in pre_results}
        assert run1 in pre_run_ids, f"Pre-delete: run1 should exist in ns='{ns}'"

    await saver.adelete_for_runs([run1])
    for ns in ["", "child:1"]:
        results = []
        async for tup in saver.alist(generate_config(tid, checkpoint_ns=ns)):
            results.append(tup)
        run_ids = {t.metadata.get("run_id") for t in results}
        assert run1 not in run_ids


ALL_DELETE_FOR_RUNS_TESTS = [
    test_delete_for_runs_single,
    test_delete_for_runs_multiple,
    test_delete_for_runs_preserves_other_runs,
    test_delete_for_runs_removes_writes,
    test_delete_for_runs_empty_list_noop,
    test_delete_for_runs_nonexistent_noop,
    test_delete_for_runs_across_namespaces,
]


async def run_delete_for_runs_tests(
    saver: BaseCheckpointSaver,
    on_test_result: Callable[[str, str, bool, str | None], None] | None = None,
) -> tuple[int, int, list[str]]:
    """Run all delete_for_runs tests. Returns (passed, failed, failure_names)."""
    passed = 0
    failed = 0
    failures: list[str] = []
    for test_fn in ALL_DELETE_FOR_RUNS_TESTS:
        try:
            await test_fn(saver)
            passed += 1
            if on_test_result:
                on_test_result("delete_for_runs", test_fn.__name__, True, None)
        except Exception as e:
            failed += 1
            msg = f"{test_fn.__name__}: {e}"
            failures.append(msg)
            if on_test_result:
                on_test_result(
                    "delete_for_runs", test_fn.__name__, False, traceback.format_exc()
                )
    return passed, failed, failures
