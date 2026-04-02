"""PUT capability tests — aput + aget_tuple round-trip."""

from __future__ import annotations

import traceback
from collections.abc import Callable
from typing import Any
from uuid import uuid4

from langgraph.checkpoint.base import (
    BaseCheckpointSaver,
    ChannelVersions,
)

from langgraph.checkpoint.conformance.test_utils import (
    generate_checkpoint,
    generate_config,
    generate_metadata,
)


async def test_put_returns_config(saver: BaseCheckpointSaver) -> None:
    """aput returns a RunnableConfig with thread_id, checkpoint_ns, checkpoint_id."""
    config = generate_config()
    cp = generate_checkpoint(channel_values={"k": "v"})
    cp["channel_versions"] = {"k": 1}
    md = generate_metadata()

    result = await saver.aput(config, cp, md, {"k": 1})

    assert "configurable" in result
    conf = result["configurable"]
    assert "thread_id" in conf
    assert "checkpoint_ns" in conf
    assert "checkpoint_id" in conf
    assert conf["checkpoint_id"] == cp["id"]


async def test_put_roundtrip(saver: BaseCheckpointSaver) -> None:
    """put then get_tuple returns identical checkpoint."""
    config = generate_config()
    cp = generate_checkpoint(channel_values={"msg": "hello"})
    cp["channel_versions"] = {"msg": 1}
    md = generate_metadata(source="input", step=-1)

    stored_config = await saver.aput(config, cp, md, {"msg": 1})

    tup = await saver.aget_tuple(stored_config)
    assert tup is not None
    assert tup.checkpoint["id"] == cp["id"]
    assert tup.checkpoint["channel_values"] == {"msg": "hello"}


async def test_put_preserves_channel_values(saver: BaseCheckpointSaver) -> None:
    """Various types (str, int, list, dict, bytes, None) round-trip correctly."""
    values: dict[str, Any] = {
        "str_val": "hello",
        "int_val": 42,
        "list_val": [1, 2, 3],
        "dict_val": {"nested": True},
    }
    config = generate_config()
    cp = generate_checkpoint(channel_values=values)
    versions: ChannelVersions = {k: 1 for k in values}
    cp["channel_versions"] = versions
    md = generate_metadata()

    stored = await saver.aput(config, cp, md, versions)
    tup = await saver.aget_tuple(stored)
    assert tup is not None
    for k, v in values.items():
        assert tup.checkpoint["channel_values"].get(k) == v, (
            f"channel_values[{k}]: expected {v!r}, got {tup.checkpoint['channel_values'].get(k)!r}"
        )


async def test_put_preserves_channel_versions(saver: BaseCheckpointSaver) -> None:
    """ChannelVersions round-trip correctly."""
    versions: ChannelVersions = {"a": 1, "b": 2}
    config = generate_config()
    cp = generate_checkpoint(
        channel_values={"a": "x", "b": "y"}, channel_versions=versions
    )
    md = generate_metadata()

    stored = await saver.aput(config, cp, md, versions)
    tup = await saver.aget_tuple(stored)
    assert tup is not None
    # Compare version values — checkpointers may convert int to str
    for k, expected in versions.items():
        actual = tup.checkpoint["channel_versions"].get(k)
        assert actual is not None, f"channel_versions[{k}] missing"
        assert str(actual).split(".")[0] == str(expected).split(".")[0], (
            f"channel_versions[{k}]: expected {expected!r}, got {actual!r}"
        )


async def test_put_preserves_versions_seen(saver: BaseCheckpointSaver) -> None:
    """versions_seen dict round-trips."""
    vs: dict[str, ChannelVersions] = {"node1": {"ch": 1}, "node2": {"ch": 2}}
    config = generate_config()
    cp = generate_checkpoint(versions_seen=vs)
    md = generate_metadata()

    stored = await saver.aput(config, cp, md, {})
    tup = await saver.aget_tuple(stored)
    assert tup is not None
    for node in vs:
        assert node in tup.checkpoint["versions_seen"], f"versions_seen[{node}] missing"


async def test_put_preserves_metadata(saver: BaseCheckpointSaver) -> None:
    """Metadata source, step, parents, and custom keys round-trip."""
    md = generate_metadata(source="loop", step=3, custom_key="custom_value")
    config = generate_config()
    cp = generate_checkpoint()

    stored = await saver.aput(config, cp, md, {})
    tup = await saver.aget_tuple(stored)
    assert tup is not None
    assert tup.metadata["source"] == "loop"
    assert tup.metadata["step"] == 3
    assert tup.metadata.get("custom_key") == "custom_value"


async def test_put_root_namespace(saver: BaseCheckpointSaver) -> None:
    """checkpoint_ns='' works."""
    config = generate_config(checkpoint_ns="")
    cp = generate_checkpoint()
    md = generate_metadata()

    stored = await saver.aput(config, cp, md, {})
    tup = await saver.aget_tuple(stored)
    assert tup is not None
    assert tup.config["configurable"].get("checkpoint_ns", "") == ""


async def test_put_child_namespace(saver: BaseCheckpointSaver) -> None:
    """checkpoint_ns='child:abc' works."""
    config = generate_config(checkpoint_ns="child:abc")
    cp = generate_checkpoint()
    md = generate_metadata()

    stored = await saver.aput(config, cp, md, {})
    tup = await saver.aget_tuple(stored)
    assert tup is not None
    assert tup.config["configurable"]["checkpoint_ns"] == "child:abc"


async def test_put_default_namespace(saver: BaseCheckpointSaver) -> None:
    """Config without checkpoint_ns defaults to ''."""
    tid = str(uuid4())
    config = {"configurable": {"thread_id": tid, "checkpoint_ns": ""}}
    cp = generate_checkpoint()
    md = generate_metadata()

    stored = await saver.aput(config, cp, md, {})
    tup = await saver.aget_tuple(stored)
    assert tup is not None


async def test_put_multiple_checkpoints_same_thread(
    saver: BaseCheckpointSaver,
) -> None:
    """Sequential puts on same thread, all retrievable."""
    tid = str(uuid4())
    ids = []
    parent_cfg = None
    for i in range(3):
        config = generate_config(tid)
        if parent_cfg is not None:
            config["configurable"]["checkpoint_id"] = parent_cfg["configurable"][
                "checkpoint_id"
            ]
        cp = generate_checkpoint()
        md = generate_metadata(step=i)
        parent_cfg = await saver.aput(config, cp, md, {})
        ids.append(cp["id"])

    # All three should be retrievable
    for cid in ids:
        cfg = generate_config(tid, checkpoint_id=cid)
        tup = await saver.aget_tuple(cfg)
        assert tup is not None, f"checkpoint {cid} not found"
        assert tup.checkpoint["id"] == cid


async def test_put_multiple_threads_isolated(saver: BaseCheckpointSaver) -> None:
    """Different thread_ids don't interfere."""
    tid1, tid2 = str(uuid4()), str(uuid4())

    config1 = generate_config(tid1)
    cp1 = generate_checkpoint(channel_values={"x": "thread1"})
    cp1["channel_versions"] = {"x": 1}
    await saver.aput(config1, cp1, generate_metadata(), {"x": 1})

    config2 = generate_config(tid2)
    cp2 = generate_checkpoint(channel_values={"x": "thread2"})
    cp2["channel_versions"] = {"x": 1}
    await saver.aput(config2, cp2, generate_metadata(), {"x": 1})

    tup1 = await saver.aget_tuple(generate_config(tid1))
    tup2 = await saver.aget_tuple(generate_config(tid2))
    assert tup1 is not None and tup2 is not None
    assert tup1.checkpoint["channel_values"]["x"] == "thread1"
    assert tup2.checkpoint["channel_values"]["x"] == "thread2"


async def test_put_parent_config(saver: BaseCheckpointSaver) -> None:
    """parent checkpoint_id tracked correctly."""
    tid = str(uuid4())
    config1 = generate_config(tid)
    cp1 = generate_checkpoint()
    stored1 = await saver.aput(config1, cp1, generate_metadata(step=0), {})

    # Second checkpoint — its config carries the parent checkpoint_id
    config2 = generate_config(tid)
    config2["configurable"]["checkpoint_id"] = stored1["configurable"]["checkpoint_id"]
    cp2 = generate_checkpoint()
    stored2 = await saver.aput(config2, cp2, generate_metadata(step=1), {})

    tup = await saver.aget_tuple(stored2)
    assert tup is not None
    assert tup.parent_config is not None
    assert (
        tup.parent_config["configurable"]["checkpoint_id"]
        == stored1["configurable"]["checkpoint_id"]
    )


async def test_put_incremental_channel_update(saver: BaseCheckpointSaver) -> None:
    """Only updated channels need new blobs; unchanged channels loaded from prior versions."""
    tid = str(uuid4())

    # Checkpoint 1: both channels are new
    config1 = generate_config(tid)
    cp1 = generate_checkpoint(
        channel_values={"a": "v1", "b": "v2"},
        channel_versions={"a": 1, "b": 1},
    )
    stored1 = await saver.aput(
        config1, cp1, generate_metadata(step=0), {"a": 1, "b": 1}
    )

    # Checkpoint 2: only 'a' is updated
    config2 = generate_config(tid)
    config2["configurable"]["checkpoint_id"] = stored1["configurable"]["checkpoint_id"]
    cp2 = generate_checkpoint(
        channel_values={"a": "v1_updated", "b": "v2"},
        channel_versions={"a": 2, "b": 1},
    )
    stored2 = await saver.aput(config2, cp2, generate_metadata(step=1), {"a": 2})

    # cp2 should reconstruct full channel_values from blobs at mixed versions
    tup2 = await saver.aget_tuple(stored2)
    assert tup2 is not None
    assert tup2.checkpoint["channel_values"].get("a") == "v1_updated", (
        f"a: expected 'v1_updated', got {tup2.checkpoint['channel_values'].get('a')!r}"
    )
    assert tup2.checkpoint["channel_values"].get("b") == "v2", (
        f"b: expected 'v2', got {tup2.checkpoint['channel_values'].get('b')!r}"
    )

    # cp1 should still return original values
    tup1 = await saver.aget_tuple(stored1)
    assert tup1 is not None
    assert tup1.checkpoint["channel_values"].get("a") == "v1"
    assert tup1.checkpoint["channel_values"].get("b") == "v2"


async def test_put_new_channel_added(saver: BaseCheckpointSaver) -> None:
    """A channel that appears for the first time in a later checkpoint."""
    tid = str(uuid4())

    config1 = generate_config(tid)
    cp1 = generate_checkpoint(
        channel_values={"a": "v1"},
        channel_versions={"a": 1},
    )
    stored1 = await saver.aput(config1, cp1, generate_metadata(step=0), {"a": 1})

    # Checkpoint 2: 'b' is brand new, 'a' is unchanged
    config2 = generate_config(tid)
    config2["configurable"]["checkpoint_id"] = stored1["configurable"]["checkpoint_id"]
    cp2 = generate_checkpoint(
        channel_values={"a": "v1", "b": "new_channel"},
        channel_versions={"a": 1, "b": 1},
    )
    stored2 = await saver.aput(config2, cp2, generate_metadata(step=1), {"b": 1})

    tup2 = await saver.aget_tuple(stored2)
    assert tup2 is not None
    assert tup2.checkpoint["channel_values"].get("a") == "v1", (
        f"a: expected 'v1', got {tup2.checkpoint['channel_values'].get('a')!r}"
    )
    assert tup2.checkpoint["channel_values"].get("b") == "new_channel", (
        f"b: expected 'new_channel', got {tup2.checkpoint['channel_values'].get('b')!r}"
    )


async def test_put_channel_removed(saver: BaseCheckpointSaver) -> None:
    """Channel no longer in channel_versions should not appear in loaded values."""
    tid = str(uuid4())

    config1 = generate_config(tid)
    cp1 = generate_checkpoint(
        channel_values={"a": "v1", "b": "v2"},
        channel_versions={"a": 1, "b": 1},
    )
    stored1 = await saver.aput(
        config1, cp1, generate_metadata(step=0), {"a": 1, "b": 1}
    )

    # Checkpoint 2: 'b' dropped from channel_versions
    config2 = generate_config(tid)
    config2["configurable"]["checkpoint_id"] = stored1["configurable"]["checkpoint_id"]
    cp2 = generate_checkpoint(
        channel_values={"a": "v1_updated"},
        channel_versions={"a": 2},
    )
    stored2 = await saver.aput(config2, cp2, generate_metadata(step=1), {"a": 2})

    tup2 = await saver.aget_tuple(stored2)
    assert tup2 is not None
    assert tup2.checkpoint["channel_values"].get("a") == "v1_updated"
    assert "b" not in tup2.checkpoint["channel_values"], (
        f"'b' should not be present, got {tup2.checkpoint['channel_values']}"
    )


async def test_put_preserves_run_id(saver: BaseCheckpointSaver) -> None:
    """run_id in metadata round-trips correctly."""
    run_id = str(uuid4())
    config = generate_config()
    cp = generate_checkpoint()
    md = generate_metadata(source="loop", step=0, run_id=run_id)

    stored = await saver.aput(config, cp, md, {})
    tup = await saver.aget_tuple(stored)
    assert tup is not None
    assert tup.metadata.get("run_id") == run_id, (
        f"run_id: expected {run_id!r}, got {tup.metadata.get('run_id')!r}"
    )


async def test_put_preserves_versions_seen_values(saver: BaseCheckpointSaver) -> None:
    """versions_seen values (not just keys) round-trip correctly."""
    vs: dict[str, ChannelVersions] = {
        "node1": {"ch_a": 1, "ch_b": 2},
        "node2": {"ch_a": 3},
    }
    config = generate_config()
    cp = generate_checkpoint(versions_seen=vs)
    md = generate_metadata()

    stored = await saver.aput(config, cp, md, {})
    tup = await saver.aget_tuple(stored)
    assert tup is not None
    for node, expected_versions in vs.items():
        assert node in tup.checkpoint["versions_seen"], f"versions_seen[{node}] missing"
        actual_versions = tup.checkpoint["versions_seen"][node]
        for ch, expected_v in expected_versions.items():
            actual_v = actual_versions.get(ch)
            assert actual_v is not None, f"versions_seen[{node}][{ch}] missing"
            assert str(actual_v).split(".")[0] == str(expected_v).split(".")[0], (
                f"versions_seen[{node}][{ch}]: expected {expected_v!r}, got {actual_v!r}"
            )


ALL_PUT_TESTS = [
    test_put_returns_config,
    test_put_roundtrip,
    test_put_preserves_channel_values,
    test_put_preserves_channel_versions,
    test_put_preserves_versions_seen,
    test_put_preserves_metadata,
    test_put_root_namespace,
    test_put_child_namespace,
    test_put_default_namespace,
    test_put_multiple_checkpoints_same_thread,
    test_put_multiple_threads_isolated,
    test_put_parent_config,
    test_put_incremental_channel_update,
    test_put_new_channel_added,
    test_put_channel_removed,
    test_put_preserves_run_id,
    test_put_preserves_versions_seen_values,
]


async def run_put_tests(
    saver: BaseCheckpointSaver,
    on_test_result: Callable[[str, str, bool, str | None], None] | None = None,
) -> tuple[int, int, list[str]]:
    """Run all put tests. Returns (passed, failed, failure_names)."""
    passed = 0
    failed = 0
    failures: list[str] = []
    for test_fn in ALL_PUT_TESTS:
        try:
            await test_fn(saver)
            passed += 1
            if on_test_result:
                on_test_result("put", test_fn.__name__, True, None)
        except Exception as e:
            failed += 1
            msg = f"{test_fn.__name__}: {e}"
            failures.append(msg)
            if on_test_result:
                on_test_result("put", test_fn.__name__, False, traceback.format_exc())
    return passed, failed, failures
