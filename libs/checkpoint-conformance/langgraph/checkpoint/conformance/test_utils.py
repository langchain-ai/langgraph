"""Test utilities: checkpoint generators, assertion helpers, bulk operations."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any
from uuid import uuid4

from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.base import (
    ChannelVersions,
    Checkpoint,
    CheckpointMetadata,
    CheckpointTuple,
)
from langgraph.checkpoint.base.id import uuid6


def generate_checkpoint(
    *,
    checkpoint_id: str | None = None,
    channel_values: dict[str, Any] | None = None,
    channel_versions: ChannelVersions | None = None,
    versions_seen: dict[str, ChannelVersions] | None = None,
) -> Checkpoint:
    """Create a well-formed Checkpoint with sensible defaults."""
    return Checkpoint(
        v=1,
        id=checkpoint_id or str(uuid6(clock_seq=-1)),
        ts=datetime.now(timezone.utc).isoformat(),
        channel_values=channel_values if channel_values is not None else {},
        channel_versions=channel_versions if channel_versions is not None else {},
        versions_seen=versions_seen if versions_seen is not None else {},
        pending_sends=[],  # ty: ignore[invalid-key]
        updated_channels=None,
    )


def generate_config(
    thread_id: str | None = None,
    *,
    checkpoint_ns: str = "",
    checkpoint_id: str | None = None,
) -> RunnableConfig:
    """Create a RunnableConfig targeting a specific thread / namespace / checkpoint."""
    configurable: dict[str, Any] = {
        "thread_id": thread_id or str(uuid4()),
        "checkpoint_ns": checkpoint_ns,
    }
    if checkpoint_id is not None:
        configurable["checkpoint_id"] = checkpoint_id
    return {"configurable": configurable}


def generate_metadata(
    source: str = "loop",
    step: int = 0,
    **extra: Any,
) -> CheckpointMetadata:
    """Create CheckpointMetadata with defaults."""
    md: dict[str, Any] = {"source": source, "step": step, "parents": {}}
    md.update(extra)
    return md


async def put_test_checkpoint(
    saver: Any,
    *,
    thread_id: str | None = None,
    checkpoint_ns: str = "",
    parent_config: RunnableConfig | None = None,
    channel_values: dict[str, Any] | None = None,
    channel_versions: ChannelVersions | None = None,
    metadata: CheckpointMetadata | None = None,
    new_versions: ChannelVersions | None = None,
) -> RunnableConfig:
    """Put a single test checkpoint and return the stored config.

    Handles wiring up parent_config, channel_values -> new_versions, etc.
    """
    tid = thread_id or str(uuid4())
    cp = generate_checkpoint(
        channel_values=channel_values,
        channel_versions=channel_versions,
    )

    # When channel_values are provided, ensure channel_versions + new_versions
    # are consistent so the checkpointer stores the blobs correctly.
    vals = channel_values or {}
    cv = channel_versions
    if cv is None and vals:
        cv = {k: 1 for k in vals}
        cp["channel_versions"] = cv
    nv = new_versions
    if nv is None:
        nv = cv or {}

    md = metadata or generate_metadata()

    config = generate_config(tid, checkpoint_ns=checkpoint_ns)
    if parent_config is not None:
        config["configurable"]["checkpoint_id"] = parent_config["configurable"][
            "checkpoint_id"
        ]

    return await saver.aput(config, cp, md, nv)


async def put_test_checkpoints(
    saver: Any,
    *,
    n_threads: int = 1,
    n_checkpoints: int = 1,
    namespaces: list[str] | None = None,
    channel_values: dict[str, Any] | None = None,
) -> list[RunnableConfig]:
    """Convenience: put multiple checkpoints across threads/namespaces.

    Returns the stored configs in insertion order.
    """
    nss = namespaces or [""]
    stored: list[RunnableConfig] = []
    for t in range(n_threads):
        tid = f"thread-{t}"
        for ns in nss:
            parent: RunnableConfig | None = None
            for _c in range(n_checkpoints):
                cfg = await put_test_checkpoint(
                    saver,
                    thread_id=tid,
                    checkpoint_ns=ns,
                    parent_config=parent,
                    channel_values=channel_values,
                )
                parent = cfg
                stored.append(cfg)
    return stored


def assert_checkpoint_equal(
    actual: Checkpoint,
    expected: Checkpoint,
    *,
    check_channel_values: bool = True,
) -> None:
    """Assert two checkpoints are semantically equal."""
    assert actual["v"] == expected["v"], f"v mismatch: {actual['v']} != {expected['v']}"
    assert actual["id"] == expected["id"], (
        f"id mismatch: {actual['id']} != {expected['id']}"
    )
    assert actual["channel_versions"] == expected["channel_versions"], (
        "channel_versions mismatch"
    )
    assert actual["versions_seen"] == expected["versions_seen"], (
        "versions_seen mismatch"
    )
    if check_channel_values:
        assert actual["channel_values"] == expected["channel_values"], (
            "channel_values mismatch"
        )


def assert_tuple_equal(
    actual: CheckpointTuple,
    expected: CheckpointTuple,
    *,
    check_writes: bool = True,
    check_channel_values: bool = True,
) -> None:
    """Assert two CheckpointTuples are semantically equal."""
    # Config
    a_conf = actual.config["configurable"]
    e_conf = expected.config["configurable"]
    assert a_conf["thread_id"] == e_conf["thread_id"], (
        f"thread_id mismatch: {a_conf['thread_id']} != {e_conf['thread_id']}"
    )
    assert a_conf.get("checkpoint_ns", "") == e_conf.get("checkpoint_ns", ""), (
        "checkpoint_ns mismatch"
    )
    assert a_conf["checkpoint_id"] == e_conf["checkpoint_id"], "checkpoint_id mismatch"

    # Checkpoint
    assert_checkpoint_equal(
        actual.checkpoint,
        expected.checkpoint,
        check_channel_values=check_channel_values,
    )

    # Metadata
    for k, v in expected.metadata.items():
        assert actual.metadata.get(k) == v, (
            f"metadata[{k}] mismatch: {actual.metadata.get(k)} != {v}"
        )

    # Parent config
    if expected.parent_config is not None:
        assert actual.parent_config is not None, "expected parent_config, got None"
        assert (
            actual.parent_config["configurable"]["checkpoint_id"]
            == expected.parent_config["configurable"]["checkpoint_id"]
        ), "parent checkpoint_id mismatch"
    else:
        assert actual.parent_config is None, (
            f"expected no parent_config, got {actual.parent_config}"
        )

    # Pending writes
    if check_writes and expected.pending_writes is not None:
        assert actual.pending_writes is not None
        assert len(actual.pending_writes) == len(expected.pending_writes), (
            f"pending_writes length mismatch: {len(actual.pending_writes)} != {len(expected.pending_writes)}"
        )
