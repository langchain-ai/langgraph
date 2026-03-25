"""COPY_THREAD capability tests — acopy_thread."""

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


async def _setup_source_thread(
    saver: BaseCheckpointSaver,
    tid: str,
    *,
    n: int = 3,
    namespaces: list[str] | None = None,
) -> list[dict]:
    """Create n checkpoints on tid (optionally across namespaces). Returns stored configs."""
    nss = namespaces or [""]
    stored = []
    for ns in nss:
        parent_cfg = None
        for i in range(n):
            config = generate_config(tid, checkpoint_ns=ns)
            if parent_cfg:
                config["configurable"]["checkpoint_id"] = parent_cfg["configurable"][
                    "checkpoint_id"
                ]
            cp = generate_checkpoint(channel_values={"step": i})
            cp["channel_versions"] = {"step": 1}
            parent_cfg = await saver.aput(
                config, cp, generate_metadata(step=i), {"step": 1}
            )
            stored.append(parent_cfg)
    return stored


async def test_copy_thread_basic(saver: BaseCheckpointSaver) -> None:
    """Checkpoints appear on target thread."""
    src = str(uuid4())
    dst = str(uuid4())
    await _setup_source_thread(saver, src)

    await saver.acopy_thread(src, dst)
    results = []
    async for tup in saver.alist(generate_config(dst)):
        results.append(tup)
    assert len(results) == 3, f"Expected 3 copied checkpoints, got {len(results)}"


async def test_copy_thread_all_checkpoints(saver: BaseCheckpointSaver) -> None:
    """All checkpoints copied, not just latest."""
    src = str(uuid4())
    dst = str(uuid4())
    await _setup_source_thread(saver, src, n=3)

    await saver.acopy_thread(src, dst)
    src_results = []
    async for tup in saver.alist(generate_config(src)):
        src_results.append(tup)

    dst_results = []
    async for tup in saver.alist(generate_config(dst)):
        dst_results.append(tup)

    assert len(dst_results) == len(src_results)
    # Verify content matches
    for s, d in zip(
        sorted(src_results, key=lambda t: t.checkpoint["id"]),
        sorted(dst_results, key=lambda t: t.checkpoint["id"]),
        strict=True,
    ):
        assert s.checkpoint["channel_values"] == d.checkpoint["channel_values"], (
            f"channel_values mismatch for checkpoint {s.checkpoint['id']}"
        )


async def test_copy_thread_preserves_metadata(
    saver: BaseCheckpointSaver,
) -> None:
    """Metadata intact on copied checkpoints."""
    src = str(uuid4())
    dst = str(uuid4())
    await _setup_source_thread(saver, src, n=2)

    await saver.acopy_thread(src, dst)
    src_tuples = []
    async for tup in saver.alist(generate_config(src)):
        src_tuples.append(tup)

    dst_tuples = []
    async for tup in saver.alist(generate_config(dst)):
        dst_tuples.append(tup)

    for s, d in zip(
        sorted(src_tuples, key=lambda t: t.metadata.get("step", 0)),
        sorted(dst_tuples, key=lambda t: t.metadata.get("step", 0)),
        strict=True,
    ):
        for key in s.metadata:
            assert s.metadata.get(key) == d.metadata.get(key), (
                f"metadata[{key!r}] mismatch: {s.metadata.get(key)!r} != {d.metadata.get(key)!r}"
            )


async def test_copy_thread_preserves_namespaces(
    saver: BaseCheckpointSaver,
) -> None:
    """Root + child namespaces copied."""
    src = str(uuid4())
    dst = str(uuid4())
    await _setup_source_thread(saver, src, n=1, namespaces=["", "child:1"])

    await saver.acopy_thread(src, dst)
    for ns in ["", "child:1"]:
        results = []
        async for tup in saver.alist(generate_config(dst, checkpoint_ns=ns)):
            results.append(tup)
        assert len(results) == 1, (
            f"Expected 1 checkpoint in namespace '{ns}', got {len(results)}"
        )


async def test_copy_thread_preserves_writes(saver: BaseCheckpointSaver) -> None:
    """Pending writes copied."""
    src = str(uuid4())
    dst = str(uuid4())
    configs = await _setup_source_thread(saver, src, n=1)

    # Add a write to the source
    await saver.aput_writes(configs[-1], [("ch", "write_val")], str(uuid4()))

    await saver.acopy_thread(src, dst)
    tup = await saver.aget_tuple(generate_config(dst))
    assert tup is not None
    assert tup.pending_writes is not None
    assert len(tup.pending_writes) == 1, (
        f"Expected 1 write, got {len(tup.pending_writes)}"
    )
    assert tup.pending_writes[0][1] == "ch", (
        f"channel mismatch: {tup.pending_writes[0][1]!r}"
    )
    assert tup.pending_writes[0][2] == "write_val", (
        f"value mismatch: {tup.pending_writes[0][2]!r}"
    )


async def test_copy_thread_preserves_ordering(
    saver: BaseCheckpointSaver,
) -> None:
    """Checkpoint order maintained."""
    src = str(uuid4())
    dst = str(uuid4())
    await _setup_source_thread(saver, src, n=4)

    await saver.acopy_thread(src, dst)
    src_ids = []
    async for tup in saver.alist(generate_config(src)):
        src_ids.append(tup.checkpoint["id"])

    dst_ids = []
    async for tup in saver.alist(generate_config(dst)):
        dst_ids.append(tup.checkpoint["id"])

    # Order should match (both newest-first)
    assert src_ids == dst_ids


async def test_copy_thread_source_unchanged(saver: BaseCheckpointSaver) -> None:
    """Source thread still intact after copy."""
    src = str(uuid4())
    dst = str(uuid4())
    await _setup_source_thread(saver, src, n=2)

    # Snapshot source before copy
    src_before = []
    async for tup in saver.alist(generate_config(src)):
        src_before.append(tup.checkpoint["id"])

    await saver.acopy_thread(src, dst)
    # Source should be unchanged
    src_after = []
    async for tup in saver.alist(generate_config(src)):
        src_after.append(tup.checkpoint["id"])

    assert src_before == src_after


async def test_copy_thread_nonexistent_source(
    saver: BaseCheckpointSaver,
) -> None:
    """Graceful handling of non-existent source thread."""
    src = str(uuid4())
    dst = str(uuid4())

    # Should not raise (or raise a known error)
    try:
        await saver.acopy_thread(src, dst)
    except Exception:
        pass  # Some implementations may raise; that's acceptable

    # Destination should be empty
    results = []
    async for tup in saver.alist(generate_config(dst)):
        results.append(tup)
    assert len(results) == 0


ALL_COPY_THREAD_TESTS = [
    test_copy_thread_basic,
    test_copy_thread_all_checkpoints,
    test_copy_thread_preserves_metadata,
    test_copy_thread_preserves_namespaces,
    test_copy_thread_preserves_writes,
    test_copy_thread_preserves_ordering,
    test_copy_thread_source_unchanged,
    test_copy_thread_nonexistent_source,
]


async def run_copy_thread_tests(
    saver: BaseCheckpointSaver,
    on_test_result: Callable[[str, str, bool, str | None], None] | None = None,
) -> tuple[int, int, list[str]]:
    """Run all copy_thread tests. Returns (passed, failed, failure_names)."""
    passed = 0
    failed = 0
    failures: list[str] = []
    for test_fn in ALL_COPY_THREAD_TESTS:
        try:
            await test_fn(saver)
            passed += 1
            if on_test_result:
                on_test_result("copy_thread", test_fn.__name__, True, None)
        except Exception as e:
            failed += 1
            msg = f"{test_fn.__name__}: {e}"
            failures.append(msg)
            if on_test_result:
                on_test_result(
                    "copy_thread", test_fn.__name__, False, traceback.format_exc()
                )
    return passed, failed, failures
