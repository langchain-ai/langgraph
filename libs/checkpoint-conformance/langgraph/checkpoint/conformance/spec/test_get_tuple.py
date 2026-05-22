"""GET_TUPLE capability tests — aget_tuple retrieval."""

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


async def test_get_tuple_nonexistent_returns_none(
    saver: BaseCheckpointSaver,
) -> None:
    """Missing thread returns None."""
    config = generate_config(str(uuid4()))
    tup = await saver.aget_tuple(config)
    assert tup is None


async def test_get_tuple_latest_when_no_checkpoint_id(
    saver: BaseCheckpointSaver,
) -> None:
    """Returns newest checkpoint when no checkpoint_id in config."""
    tid = str(uuid4())
    ids = []
    parent_cfg = None
    for i in range(3):
        config = generate_config(tid)
        if parent_cfg:
            config["configurable"]["checkpoint_id"] = parent_cfg["configurable"][
                "checkpoint_id"
            ]
        cp = generate_checkpoint()
        parent_cfg = await saver.aput(config, cp, generate_metadata(step=i), {})
        ids.append(cp["id"])

    # Get without checkpoint_id — should return the latest
    tup = await saver.aget_tuple(generate_config(tid))
    assert tup is not None
    assert tup.checkpoint["id"] == ids[-1]
    assert tup.metadata["step"] == 2, (
        f"Expected latest step=2, got {tup.metadata['step']}"
    )


async def test_get_tuple_specific_checkpoint_id(
    saver: BaseCheckpointSaver,
) -> None:
    """Returns exact match when checkpoint_id specified."""
    tid = str(uuid4())

    config1 = generate_config(tid)
    cp1 = generate_checkpoint()
    stored1 = await saver.aput(config1, cp1, generate_metadata(step=0), {})

    config2 = generate_config(tid)
    config2["configurable"]["checkpoint_id"] = stored1["configurable"]["checkpoint_id"]
    cp2 = generate_checkpoint()
    await saver.aput(config2, cp2, generate_metadata(step=1), {})

    # Fetch the first one specifically
    tup = await saver.aget_tuple(stored1)
    assert tup is not None
    assert tup.checkpoint["id"] == cp1["id"]


async def test_get_tuple_config_structure(saver: BaseCheckpointSaver) -> None:
    """tuple.config has thread_id, checkpoint_ns, checkpoint_id."""
    tid = str(uuid4())
    config = generate_config(tid)
    cp = generate_checkpoint()
    stored = await saver.aput(config, cp, generate_metadata(), {})

    tup = await saver.aget_tuple(stored)
    assert tup is not None
    conf = tup.config["configurable"]
    assert conf["thread_id"] == tid
    assert conf.get("checkpoint_ns", "") == "", (
        f"Expected checkpoint_ns='', got {conf.get('checkpoint_ns')!r}"
    )
    assert conf["checkpoint_id"] == cp["id"]


async def test_get_tuple_checkpoint_fields(saver: BaseCheckpointSaver) -> None:
    """All Checkpoint fields present."""
    tid = str(uuid4())
    config = generate_config(tid)
    cp = generate_checkpoint(channel_values={"k": "v"})
    cp["channel_versions"] = {"k": 1}
    stored = await saver.aput(config, cp, generate_metadata(), {"k": 1})

    tup = await saver.aget_tuple(stored)
    assert tup is not None
    c = tup.checkpoint
    assert c["id"] == cp["id"], f"id mismatch: {c['id']!r} != {cp['id']!r}"
    assert c["v"] == 1, f"Expected v=1, got {c['v']!r}"
    assert "ts" in c and c["ts"], "ts should be non-empty"
    assert c["channel_values"] == {"k": "v"}, f"channel_values: {c['channel_values']!r}"
    assert "channel_versions" in c
    assert "versions_seen" in c


async def test_get_tuple_metadata(saver: BaseCheckpointSaver) -> None:
    """metadata populated correctly."""
    tid = str(uuid4())
    config = generate_config(tid)
    cp = generate_checkpoint()
    md = generate_metadata(source="input", step=-1)
    stored = await saver.aput(config, cp, md, {})

    tup = await saver.aget_tuple(stored)
    assert tup is not None
    assert tup.metadata["source"] == "input"
    assert tup.metadata["step"] == -1


async def test_get_tuple_parent_config(saver: BaseCheckpointSaver) -> None:
    """parent_config when parent exists, None otherwise."""
    tid = str(uuid4())

    # First checkpoint — no parent
    config1 = generate_config(tid)
    cp1 = generate_checkpoint()
    stored1 = await saver.aput(config1, cp1, generate_metadata(step=0), {})

    tup1 = await saver.aget_tuple(stored1)
    assert tup1 is not None
    assert tup1.parent_config is None

    # Second checkpoint — has parent
    config2 = generate_config(tid)
    config2["configurable"]["checkpoint_id"] = stored1["configurable"]["checkpoint_id"]
    cp2 = generate_checkpoint()
    stored2 = await saver.aput(config2, cp2, generate_metadata(step=1), {})

    tup2 = await saver.aget_tuple(stored2)
    assert tup2 is not None
    assert tup2.parent_config is not None
    assert (
        tup2.parent_config["configurable"]["checkpoint_id"]
        == stored1["configurable"]["checkpoint_id"]
    )


async def test_get_tuple_pending_writes(saver: BaseCheckpointSaver) -> None:
    """pending_writes from put_writes visible."""
    tid = str(uuid4())
    config = generate_config(tid)
    cp = generate_checkpoint()
    stored = await saver.aput(config, cp, generate_metadata(), {})

    task_id = str(uuid4())
    await saver.aput_writes(stored, [("ch", "val")], task_id)

    tup = await saver.aget_tuple(stored)
    assert tup is not None
    assert tup.pending_writes is not None
    assert len(tup.pending_writes) == 1, (
        f"Expected 1 write, got {len(tup.pending_writes)}"
    )
    assert tup.pending_writes[0][0] == task_id, (
        f"task_id mismatch: {tup.pending_writes[0][0]!r}"
    )
    assert tup.pending_writes[0][1] == "ch", (
        f"channel mismatch: {tup.pending_writes[0][1]!r}"
    )
    assert tup.pending_writes[0][2] == "val", (
        f"value mismatch: {tup.pending_writes[0][2]!r}"
    )


async def test_get_tuple_respects_namespace(saver: BaseCheckpointSaver) -> None:
    """checkpoint_ns filtering."""
    tid = str(uuid4())

    cfg_root = generate_config(tid, checkpoint_ns="")
    cp_root = generate_checkpoint()
    stored_root = await saver.aput(cfg_root, cp_root, generate_metadata(), {})

    cfg_child = generate_config(tid, checkpoint_ns="child:1")
    cp_child = generate_checkpoint()
    stored_child = await saver.aput(cfg_child, cp_child, generate_metadata(), {})

    tup_root = await saver.aget_tuple(stored_root)
    assert tup_root is not None
    assert tup_root.checkpoint["id"] == cp_root["id"]

    tup_child = await saver.aget_tuple(stored_child)
    assert tup_child is not None
    assert tup_child.checkpoint["id"] == cp_child["id"]


async def test_get_tuple_nonexistent_checkpoint_id(
    saver: BaseCheckpointSaver,
) -> None:
    """Specific but missing checkpoint_id returns None."""
    tid = str(uuid4())
    nonexistent_id = str(uuid4())
    # Put one checkpoint so the thread exists
    config = generate_config(tid)
    cp = generate_checkpoint()
    await saver.aput(config, cp, generate_metadata(), {})

    # Ask for a non-existent checkpoint_id
    bad_cfg = generate_config(tid, checkpoint_id=nonexistent_id)
    tup = await saver.aget_tuple(bad_cfg)
    assert tup is None


ALL_GET_TUPLE_TESTS = [
    test_get_tuple_nonexistent_returns_none,
    test_get_tuple_latest_when_no_checkpoint_id,
    test_get_tuple_specific_checkpoint_id,
    test_get_tuple_config_structure,
    test_get_tuple_checkpoint_fields,
    test_get_tuple_metadata,
    test_get_tuple_parent_config,
    test_get_tuple_pending_writes,
    test_get_tuple_respects_namespace,
    test_get_tuple_nonexistent_checkpoint_id,
]


async def run_get_tuple_tests(
    saver: BaseCheckpointSaver,
    on_test_result: Callable[[str, str, bool, str | None], None] | None = None,
) -> tuple[int, int, list[str]]:
    """Run all get_tuple tests. Returns (passed, failed, failure_names)."""
    passed = 0
    failed = 0
    failures: list[str] = []
    for test_fn in ALL_GET_TUPLE_TESTS:
        try:
            await test_fn(saver)
            passed += 1
            if on_test_result:
                on_test_result("get_tuple", test_fn.__name__, True, None)
        except Exception as e:
            failed += 1
            msg = f"{test_fn.__name__}: {e}"
            failures.append(msg)
            if on_test_result:
                on_test_result(
                    "get_tuple", test_fn.__name__, False, traceback.format_exc()
                )
    return passed, failed, failures
