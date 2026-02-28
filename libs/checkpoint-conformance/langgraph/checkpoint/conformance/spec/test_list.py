"""LIST capability tests — alist with various filters."""

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


async def _setup_list_data(saver: BaseCheckpointSaver) -> dict:
    """Populate saver with test data for list tests. Returns lookup info."""
    tid = str(uuid4())
    ids = []
    parent_cfg = None
    for i in range(4):
        config = generate_config(tid)
        if parent_cfg:
            config["configurable"]["checkpoint_id"] = parent_cfg["configurable"][
                "checkpoint_id"
            ]
        cp = generate_checkpoint()
        source = "input" if i % 2 == 0 else "loop"
        md = generate_metadata(source=source, step=i)
        parent_cfg = await saver.aput(config, cp, md, {})
        ids.append(cp["id"])

    return {
        "thread_id": tid,
        "checkpoint_ids": ids,
        "latest_config": parent_cfg,
    }


async def test_list_all(saver: BaseCheckpointSaver) -> None:
    """No filters returns all checkpoints for the thread."""
    data = await _setup_list_data(saver)
    tid = data["thread_id"]

    results = []
    async for tup in saver.alist(generate_config(tid)):
        results.append(tup)

    assert len(results) == 4


async def test_list_by_thread(saver: BaseCheckpointSaver) -> None:
    """Filter by thread_id — other threads not returned."""
    data = await _setup_list_data(saver)

    # List for a non-existent thread
    results = []
    async for tup in saver.alist(generate_config(str(uuid4()))):
        results.append(tup)
    assert len(results) == 0

    # List for actual thread
    results = []
    async for tup in saver.alist(generate_config(data["thread_id"])):
        results.append(tup)
    assert len(results) == 4


async def test_list_by_namespace(saver: BaseCheckpointSaver) -> None:
    """Filter by checkpoint_ns."""
    tid = str(uuid4())

    # Root namespace
    cfg1 = generate_config(tid, checkpoint_ns="")
    cp1 = generate_checkpoint()
    await saver.aput(cfg1, cp1, generate_metadata(), {})

    # Child namespace
    cfg2 = generate_config(tid, checkpoint_ns="child:1")
    cp2 = generate_checkpoint()
    await saver.aput(cfg2, cp2, generate_metadata(), {})

    root_results = []
    async for tup in saver.alist(generate_config(tid, checkpoint_ns="")):
        root_results.append(tup)
    assert len(root_results) == 1

    child_results = []
    async for tup in saver.alist(generate_config(tid, checkpoint_ns="child:1")):
        child_results.append(tup)
    assert len(child_results) == 1


async def test_list_ordering(saver: BaseCheckpointSaver) -> None:
    """Newest first (descending checkpoint_id)."""
    data = await _setup_list_data(saver)
    ids = data["checkpoint_ids"]

    results = []
    async for tup in saver.alist(generate_config(data["thread_id"])):
        results.append(tup.checkpoint["id"])

    # Should be in reverse order (newest first)
    assert results == list(reversed(ids))


async def test_list_metadata_filter_single_key(
    saver: BaseCheckpointSaver,
) -> None:
    """filter={'source': 'input'} returns only input checkpoints."""
    data = await _setup_list_data(saver)

    results = []
    async for tup in saver.alist(
        generate_config(data["thread_id"]),
        filter={"source": "input"},
    ):
        results.append(tup)

    assert len(results) == 2, (
        f"Expected 2 'input' checkpoints (steps 0,2), got {len(results)}"
    )
    for tup in results:
        assert tup.metadata["source"] == "input"


async def test_list_metadata_filter_step(saver: BaseCheckpointSaver) -> None:
    """filter={'step': 1} returns matching checkpoints."""
    data = await _setup_list_data(saver)

    results = []
    async for tup in saver.alist(
        generate_config(data["thread_id"]),
        filter={"step": 1},
    ):
        results.append(tup)

    assert len(results) == 1
    assert results[0].metadata["step"] == 1


async def test_list_before(saver: BaseCheckpointSaver) -> None:
    """Pagination cursor — only checkpoints before the given one."""
    data = await _setup_list_data(saver)
    ids = data["checkpoint_ids"]

    # Use the 3rd checkpoint as the 'before' cursor (index 2)
    before_cfg = generate_config(data["thread_id"], checkpoint_id=ids[2])
    results = []
    async for tup in saver.alist(
        generate_config(data["thread_id"]),
        before=before_cfg,
    ):
        results.append(tup)

    # Should only include checkpoints before ids[2]
    result_ids = [t.checkpoint["id"] for t in results]
    assert ids[2] not in result_ids
    assert ids[3] not in result_ids
    assert set(result_ids) == {ids[0], ids[1]}, (
        f"Expected {{ids[0], ids[1]}}, got {set(result_ids)}"
    )


async def test_list_limit(saver: BaseCheckpointSaver) -> None:
    """limit=1, limit=N."""
    data = await _setup_list_data(saver)

    results = []
    async for tup in saver.alist(generate_config(data["thread_id"]), limit=1):
        results.append(tup)
    assert len(results) == 1

    results = []
    async for tup in saver.alist(generate_config(data["thread_id"]), limit=2):
        results.append(tup)
    assert len(results) == 2


async def test_list_limit_plus_before(saver: BaseCheckpointSaver) -> None:
    """Pagination with limit."""
    data = await _setup_list_data(saver)
    ids = data["checkpoint_ids"]

    before_cfg = generate_config(data["thread_id"], checkpoint_id=ids[3])
    results = []
    async for tup in saver.alist(
        generate_config(data["thread_id"]),
        before=before_cfg,
        limit=1,
    ):
        results.append(tup)

    assert len(results) == 1
    assert results[0].checkpoint["id"] == ids[2]


async def test_list_combined_thread_and_filter(
    saver: BaseCheckpointSaver,
) -> None:
    """thread_id + metadata filter combined."""
    data = await _setup_list_data(saver)

    results = []
    async for tup in saver.alist(
        generate_config(data["thread_id"]),
        filter={"source": "loop"},
    ):
        results.append(tup)

    assert len(results) == 2, (
        f"Expected 2 'loop' checkpoints (steps 1,3), got {len(results)}"
    )
    for tup in results:
        assert tup.metadata["source"] == "loop"


async def test_list_empty_result(saver: BaseCheckpointSaver) -> None:
    """No matches returns empty."""
    results = []
    async for tup in saver.alist(
        generate_config(str(uuid4())),
        filter={"source": "nonexistent"},
    ):
        results.append(tup)
    assert len(results) == 0


async def test_list_includes_pending_writes(saver: BaseCheckpointSaver) -> None:
    """pending_writes in listed tuples."""
    tid = str(uuid4())
    config = generate_config(tid)
    cp = generate_checkpoint()
    stored = await saver.aput(config, cp, generate_metadata(), {})

    await saver.aput_writes(stored, [("ch", "val")], str(uuid4()))

    results = []
    async for tup in saver.alist(generate_config(tid)):
        results.append(tup)

    assert len(results) == 1
    assert results[0].pending_writes is not None
    assert len(results[0].pending_writes) == 1, (
        f"Expected 1 write, got {len(results[0].pending_writes)}"
    )
    assert results[0].pending_writes[0][1] == "ch", (
        f"channel mismatch: {results[0].pending_writes[0][1]!r}"
    )
    assert results[0].pending_writes[0][2] == "val", (
        f"value mismatch: {results[0].pending_writes[0][2]!r}"
    )


async def test_list_multiple_namespaces(saver: BaseCheckpointSaver) -> None:
    """Root namespace checkpoint listed correctly."""
    tid = str(uuid4())

    for ns in ["", "child:1", "child:2"]:
        cfg = generate_config(tid, checkpoint_ns=ns)
        cp = generate_checkpoint()
        await saver.aput(cfg, cp, generate_metadata(), {})

    # List with root namespace filter — should return exactly the root checkpoint
    results = []
    async for tup in saver.alist(generate_config(tid, checkpoint_ns="")):
        results.append(tup)
    assert len(results) == 1, f"Expected 1 root checkpoint, got {len(results)}"


async def test_list_metadata_filter_multiple_keys(
    saver: BaseCheckpointSaver,
) -> None:
    """filter with multiple keys — all must match."""
    tid = str(uuid4())

    # Create checkpoints with different metadata combos
    for source, step in [("input", 1), ("loop", 1), ("input", 2)]:
        cfg = generate_config(tid)
        cp = generate_checkpoint()
        await saver.aput(cfg, cp, generate_metadata(source=source, step=step), {})

    results = []
    async for tup in saver.alist(
        generate_config(tid),
        filter={"source": "input", "step": 2},
    ):
        results.append(tup)

    assert len(results) == 1, (
        f"Expected 1 match for source=input+step=2, got {len(results)}"
    )
    assert results[0].metadata["source"] == "input"
    assert results[0].metadata["step"] == 2


async def test_list_metadata_filter_no_match(
    saver: BaseCheckpointSaver,
) -> None:
    """Multi-key filter that matches nothing returns empty."""
    data = await _setup_list_data(saver)

    results = []
    async for tup in saver.alist(
        generate_config(data["thread_id"]),
        filter={"source": "update", "step": 99},
    ):
        results.append(tup)

    assert len(results) == 0


async def test_list_metadata_custom_keys(
    saver: BaseCheckpointSaver,
) -> None:
    """Custom (non-standard) metadata keys are filterable."""
    tid = str(uuid4())

    cfg = generate_config(tid)
    cp = generate_checkpoint()
    await saver.aput(cfg, cp, generate_metadata(score=42, run_id="run-abc"), {})

    cfg2 = generate_config(tid)
    cp2 = generate_checkpoint()
    await saver.aput(cfg2, cp2, generate_metadata(score=99, run_id="run-xyz"), {})

    # Filter by custom key
    results = []
    async for tup in saver.alist(
        generate_config(tid),
        filter={"score": 42},
    ):
        results.append(tup)

    assert len(results) == 1
    assert results[0].metadata["score"] == 42
    assert results[0].metadata["run_id"] == "run-abc"


async def test_list_global_search(
    saver: BaseCheckpointSaver,
) -> None:
    """alist(None, filter=...) searches across all threads."""
    tid1, tid2 = str(uuid4()), str(uuid4())

    # Use a unique marker so we don't collide with other tests' data
    marker = str(uuid4())

    cfg1 = generate_config(tid1)
    cp1 = generate_checkpoint()
    await saver.aput(cfg1, cp1, generate_metadata(source="input", marker=marker), {})

    cfg2 = generate_config(tid2)
    cp2 = generate_checkpoint()
    await saver.aput(cfg2, cp2, generate_metadata(source="loop", marker=marker), {})

    # Search across all threads with filter
    results = []
    async for tup in saver.alist(None, filter={"source": "input", "marker": marker}):
        results.append(tup)
    assert len(results) == 1
    assert results[0].config["configurable"]["thread_id"] == tid1

    # Search with marker only — should find both
    results = []
    async for tup in saver.alist(None, filter={"marker": marker}):
        results.append(tup)
    assert len(results) == 2


ALL_LIST_TESTS = [
    test_list_all,
    test_list_by_thread,
    test_list_by_namespace,
    test_list_ordering,
    test_list_metadata_filter_single_key,
    test_list_metadata_filter_step,
    test_list_metadata_filter_multiple_keys,
    test_list_metadata_filter_no_match,
    test_list_metadata_custom_keys,
    test_list_global_search,
    test_list_before,
    test_list_limit,
    test_list_limit_plus_before,
    test_list_combined_thread_and_filter,
    test_list_empty_result,
    test_list_includes_pending_writes,
    test_list_multiple_namespaces,
]


async def run_list_tests(
    saver: BaseCheckpointSaver,
    on_test_result: Callable[[str, str, bool, str | None], None] | None = None,
) -> tuple[int, int, list[str]]:
    """Run all list tests. Returns (passed, failed, failure_names)."""
    passed = 0
    failed = 0
    failures: list[str] = []
    for test_fn in ALL_LIST_TESTS:
        try:
            await test_fn(saver)
            passed += 1
            if on_test_result:
                on_test_result("list", test_fn.__name__, True, None)
        except Exception as e:
            failed += 1
            msg = f"{test_fn.__name__}: {e}"
            failures.append(msg)
            if on_test_result:
                on_test_result("list", test_fn.__name__, False, traceback.format_exc())
    return passed, failed, failures
