"""PUT_WRITES capability tests — aput_writes + pending_writes retrieval."""

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


async def test_put_writes_basic(saver: BaseCheckpointSaver) -> None:
    """Write stored, visible in aget_tuple pending_writes."""
    tid = str(uuid4())
    config = generate_config(tid)
    cp = generate_checkpoint()
    stored = await saver.aput(config, cp, generate_metadata(), {})

    task_id = str(uuid4())
    await saver.aput_writes(stored, [("channel1", "value1")], task_id)

    tup = await saver.aget_tuple(stored)
    assert tup is not None
    assert tup.pending_writes is not None
    # Verify exact write tuple: (task_id, channel, value)
    matching = [w for w in tup.pending_writes if w[0] == task_id and w[1] == "channel1"]
    assert len(matching) == 1, f"Expected 1 write, got {len(matching)}"
    assert matching[0][2] == "value1", f"Value mismatch: {matching[0][2]!r}"


async def test_put_writes_multiple_writes_same_task(
    saver: BaseCheckpointSaver,
) -> None:
    """Multiple (channel, value) pairs in a single call."""
    tid = str(uuid4())
    config = generate_config(tid)
    cp = generate_checkpoint()
    stored = await saver.aput(config, cp, generate_metadata(), {})

    task_id = str(uuid4())
    writes = [("ch1", "v1"), ("ch2", "v2"), ("ch3", "v3")]
    await saver.aput_writes(stored, writes, task_id)

    tup = await saver.aget_tuple(stored)
    assert tup is not None
    assert tup.pending_writes is not None
    assert len(tup.pending_writes) == 3, (
        f"Expected 3 writes, got {len(tup.pending_writes)}"
    )
    channels = {w[1] for w in tup.pending_writes}
    assert channels == {"ch1", "ch2", "ch3"}, f"Expected exact channels, got {channels}"
    # Verify values per channel
    for expected_ch, expected_val in writes:
        match = [
            w for w in tup.pending_writes if w[0] == task_id and w[1] == expected_ch
        ]
        assert len(match) == 1, f"Expected 1 write for {expected_ch}, got {len(match)}"
        assert match[0][2] == expected_val, (
            f"Value mismatch for {expected_ch}: {match[0][2]!r}"
        )


async def test_put_writes_multiple_tasks(saver: BaseCheckpointSaver) -> None:
    """Different task_ids produce separate writes."""
    tid = str(uuid4())
    config = generate_config(tid)
    cp = generate_checkpoint()
    stored = await saver.aput(config, cp, generate_metadata(), {})

    t1, t2 = str(uuid4()), str(uuid4())
    await saver.aput_writes(stored, [("ch", "from_t1")], t1)
    await saver.aput_writes(stored, [("ch", "from_t2")], t2)

    tup = await saver.aget_tuple(stored)
    assert tup is not None
    assert tup.pending_writes is not None
    assert len(tup.pending_writes) == 2, (
        f"Expected 2 writes, got {len(tup.pending_writes)}"
    )
    # Verify values per task
    t1_writes = [w for w in tup.pending_writes if w[0] == t1 and w[1] == "ch"]
    t2_writes = [w for w in tup.pending_writes if w[0] == t2 and w[1] == "ch"]
    assert len(t1_writes) == 1, f"Expected 1 write from t1, got {len(t1_writes)}"
    assert len(t2_writes) == 1, f"Expected 1 write from t2, got {len(t2_writes)}"
    assert t1_writes[0][2] == "from_t1", f"t1 value: {t1_writes[0][2]!r}"
    assert t2_writes[0][2] == "from_t2", f"t2 value: {t2_writes[0][2]!r}"


async def test_put_writes_preserves_task_id(saver: BaseCheckpointSaver) -> None:
    """task_id in pending_writes matches what was passed to aput_writes."""
    tid = str(uuid4())
    config = generate_config(tid)
    cp = generate_checkpoint()
    stored = await saver.aput(config, cp, generate_metadata(), {})

    task_id = str(uuid4())
    await saver.aput_writes(stored, [("ch", "val")], task_id)

    tup = await saver.aget_tuple(stored)
    assert tup is not None
    assert tup.pending_writes is not None
    assert any(w[0] == task_id for w in tup.pending_writes)


async def test_put_writes_preserves_channel_and_value(
    saver: BaseCheckpointSaver,
) -> None:
    """Channel name + value round-trip."""
    tid = str(uuid4())
    config = generate_config(tid)
    cp = generate_checkpoint()
    stored = await saver.aput(config, cp, generate_metadata(), {})

    task_id = str(uuid4())
    await saver.aput_writes(stored, [("my_channel", {"data": 123})], task_id)

    tup = await saver.aget_tuple(stored)
    assert tup is not None
    assert tup.pending_writes is not None
    match = [w for w in tup.pending_writes if w[0] == task_id and w[1] == "my_channel"]
    assert len(match) == 1, f"Expected 1 write, got {len(match)}"
    assert match[0][2] == {"data": 123}, f"Value mismatch: {match[0][2]!r}"


async def test_put_writes_task_path(saver: BaseCheckpointSaver) -> None:
    """task_path parameter accepted without error."""
    tid = str(uuid4())
    config = generate_config(tid)
    cp = generate_checkpoint()
    stored = await saver.aput(config, cp, generate_metadata(), {})

    task_id = str(uuid4())
    # Should not raise
    await saver.aput_writes(stored, [("ch", "v")], task_id, task_path="a:b:c")

    tup = await saver.aget_tuple(stored)
    assert tup is not None
    assert tup.pending_writes is not None
    assert len(tup.pending_writes) == 1


async def test_put_writes_idempotent(saver: BaseCheckpointSaver) -> None:
    """Duplicate (task_id, idx) doesn't duplicate writes."""
    tid = str(uuid4())
    config = generate_config(tid)
    cp = generate_checkpoint()
    stored = await saver.aput(config, cp, generate_metadata(), {})

    task_id = str(uuid4())
    await saver.aput_writes(stored, [("ch", "val")], task_id)
    await saver.aput_writes(stored, [("ch", "val")], task_id)

    tup = await saver.aget_tuple(stored)
    assert tup is not None
    assert tup.pending_writes is not None
    assert len(tup.pending_writes) == 1, (
        f"Expected exactly 1 write total, got {len(tup.pending_writes)}"
    )
    # Should not have duplicated
    matching = [w for w in tup.pending_writes if w[0] == task_id and w[1] == "ch"]
    assert len(matching) == 1


async def test_put_writes_special_channels(saver: BaseCheckpointSaver) -> None:
    """ERROR and INTERRUPT channels handled correctly."""
    from langgraph.checkpoint.serde.types import ERROR, INTERRUPT

    tid = str(uuid4())
    config = generate_config(tid)
    cp = generate_checkpoint()
    stored = await saver.aput(config, cp, generate_metadata(), {})

    task_id = str(uuid4())
    await saver.aput_writes(
        stored,
        [(ERROR, "something went wrong"), (INTERRUPT, {"reason": "human_input"})],
        task_id,
    )

    tup = await saver.aget_tuple(stored)
    assert tup is not None
    assert tup.pending_writes is not None
    channels = {w[1] for w in tup.pending_writes}
    assert ERROR in channels
    assert INTERRUPT in channels
    # Verify values
    err_writes = [w for w in tup.pending_writes if w[0] == task_id and w[1] == ERROR]
    assert len(err_writes) == 1, f"Expected 1 ERROR write, got {len(err_writes)}"
    assert err_writes[0][2] == "something went wrong", (
        f"ERROR value: {err_writes[0][2]!r}"
    )
    int_writes = [
        w for w in tup.pending_writes if w[0] == task_id and w[1] == INTERRUPT
    ]
    assert len(int_writes) == 1, f"Expected 1 INTERRUPT write, got {len(int_writes)}"
    assert int_writes[0][2] == {"reason": "human_input"}, (
        f"INTERRUPT value: {int_writes[0][2]!r}"
    )


async def test_put_writes_across_namespaces(saver: BaseCheckpointSaver) -> None:
    """Writes isolated by checkpoint_ns."""
    tid = str(uuid4())

    # Root namespace checkpoint + write
    cfg_root = generate_config(tid, checkpoint_ns="")
    cp_root = generate_checkpoint()
    stored_root = await saver.aput(cfg_root, cp_root, generate_metadata(), {})
    root_task = str(uuid4())
    await saver.aput_writes(stored_root, [("ch", "root_val")], root_task)

    # Child namespace checkpoint + write
    cfg_child = generate_config(tid, checkpoint_ns="child:1")
    cp_child = generate_checkpoint()
    stored_child = await saver.aput(cfg_child, cp_child, generate_metadata(), {})
    child_task = str(uuid4())
    await saver.aput_writes(stored_child, [("ch", "child_val")], child_task)

    # Verify isolation — root should have exactly 1 write with root_val
    tup_root = await saver.aget_tuple(stored_root)
    assert tup_root is not None
    assert tup_root.pending_writes is not None
    root_ch = [w for w in tup_root.pending_writes if w[1] == "ch"]
    assert len(root_ch) == 1, f"Expected 1 root write, got {len(root_ch)}"
    assert root_ch[0][2] == "root_val", f"Root value: {root_ch[0][2]!r}"

    # Child should have exactly 1 write with child_val
    tup_child = await saver.aget_tuple(stored_child)
    assert tup_child is not None
    assert tup_child.pending_writes is not None
    child_ch = [w for w in tup_child.pending_writes if w[1] == "ch"]
    assert len(child_ch) == 1, f"Expected 1 child write, got {len(child_ch)}"
    assert child_ch[0][2] == "child_val", f"Child value: {child_ch[0][2]!r}"


async def test_put_writes_cleared_on_next_checkpoint(
    saver: BaseCheckpointSaver,
) -> None:
    """New checkpoint starts with fresh pending_writes."""
    tid = str(uuid4())
    config = generate_config(tid)
    cp1 = generate_checkpoint()
    stored1 = await saver.aput(config, cp1, generate_metadata(step=0), {})

    await saver.aput_writes(stored1, [("ch", "old_write")], str(uuid4()))

    # New checkpoint
    config2 = generate_config(tid)
    config2["configurable"]["checkpoint_id"] = stored1["configurable"]["checkpoint_id"]
    cp2 = generate_checkpoint()
    stored2 = await saver.aput(config2, cp2, generate_metadata(step=1), {})

    tup = await saver.aget_tuple(stored2)
    assert tup is not None
    # New checkpoint should have no pending writes
    writes = tup.pending_writes or []
    assert len(writes) == 0


ALL_PUT_WRITES_TESTS = [
    test_put_writes_basic,
    test_put_writes_multiple_writes_same_task,
    test_put_writes_multiple_tasks,
    test_put_writes_preserves_task_id,
    test_put_writes_preserves_channel_and_value,
    test_put_writes_task_path,
    test_put_writes_idempotent,
    test_put_writes_special_channels,
    test_put_writes_across_namespaces,
    test_put_writes_cleared_on_next_checkpoint,
]


async def run_put_writes_tests(
    saver: BaseCheckpointSaver,
    on_test_result: Callable[[str, str, bool, str | None], None] | None = None,
) -> tuple[int, int, list[str]]:
    """Run all put_writes tests. Returns (passed, failed, failure_names)."""
    passed = 0
    failed = 0
    failures: list[str] = []
    for test_fn in ALL_PUT_WRITES_TESTS:
        try:
            await test_fn(saver)
            passed += 1
            if on_test_result:
                on_test_result("put_writes", test_fn.__name__, True, None)
        except Exception as e:
            failed += 1
            msg = f"{test_fn.__name__}: {e}"
            failures.append(msg)
            if on_test_result:
                on_test_result(
                    "put_writes", test_fn.__name__, False, traceback.format_exc()
                )
    return passed, failed, failures
