"""Shared fixtures for delta-channel conformance tests.

Builds a parent chain with `_DeltaSnapshot` blobs at known positions via
direct `aput` / `aput_writes` calls. No langgraph or Pregel dependency.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any
from uuid import uuid4

from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.base import BaseCheckpointSaver, Checkpoint
from langgraph.checkpoint.base.id import uuid6

from langgraph.checkpoint.conformance.test_utils import generate_metadata


async def build_delta_chain(
    saver: BaseCheckpointSaver,
    *,
    thread_id: str | None = None,
    checkpoint_ns: str = "",
    channel: str = "messages",
    snapshots_at_steps: Sequence[int] = (0,),
    total_steps: int = 6,
    write_value_fn: Any | None = None,
) -> list[RunnableConfig]:
    """Build a parent chain with `_DeltaSnapshot` at known positions.

    Args:
        saver: Checkpointer instance.
        thread_id: Defaults to a random UUID.
        checkpoint_ns: Namespace (default root).
        channel: Channel name used for snapshots and writes.
        snapshots_at_steps: Steps at which a `_DeltaSnapshot` blob is stored
            in `channel_values[channel]`. Step 0 is the oldest checkpoint.
        total_steps: Number of checkpoints in the chain.
        write_value_fn: Callable(step) -> write value. Defaults to step index.

    Returns:
        List of stored configs (oldest first), one per step.
    """
    if write_value_fn is None:

        def write_value_fn(step: int) -> Any:
            return step

    from langgraph.checkpoint.serde.types import _DeltaSnapshot

    thread_id = thread_id or str(uuid4())
    snapshot_set = set(snapshots_at_steps)
    stored: list[RunnableConfig] = []
    parent_cfg: RunnableConfig | None = None

    for step in range(total_steps):
        config: RunnableConfig = {
            "configurable": {
                "thread_id": thread_id,
                "checkpoint_ns": checkpoint_ns,
            }
        }
        if parent_cfg:
            config["configurable"]["checkpoint_id"] = parent_cfg["configurable"][
                "checkpoint_id"
            ]

        channel_values: dict[str, Any] = {}
        channel_versions: dict[str, int] = {}
        if step in snapshot_set:
            channel_values[channel] = _DeltaSnapshot(
                write_value_fn(step),
            )
            channel_versions[channel] = step + 1

        cp = Checkpoint(
            v=1,
            id=str(uuid6(clock_seq=-1)),
            ts="",
            channel_values=channel_values,
            channel_versions=channel_versions,
            versions_seen={},
            updated_channels=None,
        )
        new_versions = dict(channel_versions)
        parent_cfg = await saver.aput(
            config, cp, generate_metadata(step=step), new_versions
        )
        stored.append(parent_cfg)

        # Write a pending write for non-snapshot steps so the walk has
        # something to collect.
        if step not in snapshot_set:
            await saver.aput_writes(
                parent_cfg, [(channel, write_value_fn(step))], str(uuid4())
            )

    return stored
