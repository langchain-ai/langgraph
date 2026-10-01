"""Durable queue of state updates, delivered to a running graph at superstep
boundaries. See `Pregel.queue_state`.

Each queued item is one checkpoint in a reserved namespace derived from the
namespace of the graph it is addressed to: `__queue__` for the root graph,
`<checkpoint_ns>|__queue__` for a subgraph. The update itself is stored as the
checkpoint's channel values, so it goes through the saver's typed serde; the
metadata holds only JSON-native fields, so every saver can filter on them.

Only three existing saver operations are used: `put` to accept an item,
`list` with a metadata filter to read the pending items, and `put` again to
acknowledge one.
"""

from __future__ import annotations

import asyncio
import concurrent.futures
from collections.abc import Sequence
from datetime import datetime, timezone
from typing import Any, NamedTuple, cast

from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.base import (
    BaseCheckpointSaver,
    Checkpoint,
    CheckpointMetadata,
    CheckpointTuple,
)

from langgraph._internal._config import patch_configurable
from langgraph._internal._constants import (
    CONF,
    CONFIG_KEY_CHECKPOINT_ID,
    CONFIG_KEY_CHECKPOINT_NS,
    NS_SEP,
    QUEUE_NS,
    QUEUED,
)
from langgraph.pregel._checkpoint import empty_checkpoint
from langgraph.types import Command

QUEUE_META_STEER = "steer"
QUEUE_META_CONSUMED = "consumed"
QUEUE_META_ACCEPTED_AT = "accepted_at"
QUEUE_META_ERROR = "error"
# key in a regular checkpoint's metadata listing the ids of the queued items
# applied since the previous checkpoint, so a resumed loop can tell an item
# that was applied but not yet acknowledged from one that is still pending.
# Double-underscored so that `get_checkpoint_metadata` never copies a key of
# the same name from a caller's config metadata into a checkpoint.
CHECKPOINT_META_QUEUE_CONSUMED = "__queue_consumed"


class QueueItemMetadata(CheckpointMetadata, total=False):
    """Metadata of a queued-update checkpoint.

    Internal: these keys only mean something in a queue namespace, so they are
    typed here rather than on `CheckpointMetadata`.
    """

    steer: str | None
    consumed: bool
    accepted_at: str
    error: str


class QueueItem(NamedTuple):
    """A pending queued update as read from the saver."""

    id: str
    values: Any
    steer: str | None
    accepted_at: str
    config: RunnableConfig
    checkpoint: Checkpoint
    metadata: CheckpointMetadata


def queue_checkpoint_ns(checkpoint_ns: str) -> str:
    """Queue namespace for the graph running under `checkpoint_ns`."""
    return f"{checkpoint_ns}{NS_SEP}{QUEUE_NS}" if checkpoint_ns else QUEUE_NS


def is_queue_checkpoint_ns(checkpoint_ns: str) -> bool:
    return checkpoint_ns == QUEUE_NS or checkpoint_ns.endswith(f"{NS_SEP}{QUEUE_NS}")


def queue_config(config: RunnableConfig) -> RunnableConfig:
    """Config addressing the queue of the graph addressed by `config`."""
    ns = config[CONF].get(CONFIG_KEY_CHECKPOINT_NS) or ""
    return patch_configurable(
        config,
        {
            CONFIG_KEY_CHECKPOINT_NS: queue_checkpoint_ns(ns),
            CONFIG_KEY_CHECKPOINT_ID: None,
        },
    )


def queue_item_writes(values: Any) -> list[tuple[str, Any]]:
    """Map a queued update to channel writes, as `Command(update=)` would."""
    return list(Command(update=values)._update_as_tuples())


def queue_item_versions(checkpoint: Checkpoint) -> dict[str, str]:
    """`new_versions` to pass to `put` for an item checkpoint."""
    return {QUEUED: checkpoint["id"]}


def create_queue_item(
    values: Any, steer: str | None
) -> tuple[Checkpoint, QueueItemMetadata]:
    checkpoint = empty_checkpoint()
    checkpoint["channel_values"] = {QUEUED: values}
    # savers key channel blobs by (channel, version), not by checkpoint id, so
    # the version must be unique per item: its own id is
    checkpoint["channel_versions"] = {QUEUED: checkpoint["id"]}
    metadata: QueueItemMetadata = {
        "source": "queue",
        "step": -1,
        "parents": {},
        "steer": steer,
        "consumed": False,
        "accepted_at": datetime.now(timezone.utc).isoformat(),
    }
    return checkpoint, metadata


def _item_from_tuple(saved: CheckpointTuple, config: RunnableConfig) -> QueueItem:
    metadata = saved.metadata or {}
    return QueueItem(
        id=saved.checkpoint["id"],
        values=saved.checkpoint["channel_values"].get(QUEUED),
        steer=metadata.get(QUEUE_META_STEER),
        accepted_at=metadata.get(QUEUE_META_ACCEPTED_AT, ""),
        config=config,
        checkpoint=saved.checkpoint,
        metadata=metadata,
    )


def list_pending(saver: BaseCheckpointSaver, config: RunnableConfig) -> list[QueueItem]:
    """Pending items of the queue addressed by `config`, in accept order."""
    qconfig = queue_config(config)
    items = [
        _item_from_tuple(saved, qconfig)
        for saved in saver.list(qconfig, filter={QUEUE_META_CONSUMED: False})
    ]
    # savers list newest first; item ids are time-ordered
    items.sort(key=lambda item: item.id)
    return items


async def alist_pending(
    saver: BaseCheckpointSaver, config: RunnableConfig
) -> list[QueueItem]:
    qconfig = queue_config(config)
    items = [
        _item_from_tuple(saved, qconfig)
        async for saved in saver.alist(qconfig, filter={QUEUE_META_CONSUMED: False})
    ]
    items.sort(key=lambda item: item.id)
    return items


def _ack_metadata(item: QueueItem, error: str | None) -> QueueItemMetadata:
    metadata = cast(QueueItemMetadata, {**item.metadata, "consumed": True})
    if error is not None:
        metadata["error"] = error
    return metadata


def ack(
    saver: BaseCheckpointSaver, item: QueueItem, *, error: str | None = None
) -> None:
    """Mark an item consumed. Idempotent: the saver upserts on the item's id."""
    saver.put(item.config, item.checkpoint, _ack_metadata(item, error), {})


async def aack(
    saver: BaseCheckpointSaver, item: QueueItem, *, error: str | None = None
) -> None:
    await saver.aput(item.config, item.checkpoint, _ack_metadata(item, error), {})


def ack_after(
    prev: concurrent.futures.Future | None,
    saver: BaseCheckpointSaver,
    items: Sequence[QueueItem],
) -> None:
    """Acknowledge `items` once `prev`, the checkpoint write that records
    them as consumed, has completed."""
    if prev is not None:
        prev.result()
    for item in items:
        ack(saver, item)


async def aack_after(
    prev: asyncio.Future | None,
    saver: BaseCheckpointSaver,
    items: Sequence[QueueItem],
) -> None:
    if prev is not None:
        await prev
    for item in items:
        await aack(saver, item)
