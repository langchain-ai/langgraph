from __future__ import annotations

from typing import Any

import pytest
from langchain_core.runnables import RunnableConfig

from langgraph.checkpoint.base import (
    Checkpoint,
    CheckpointWrite,
    GuardedCheckpointSaver,
    empty_checkpoint,
)
from langgraph.checkpoint.memory import InMemorySaver


def _config() -> RunnableConfig:
    return {"configurable": {"thread_id": "thread-1", "checkpoint_ns": ""}}


def _checkpoint(value: Any, *, channel: str = "messages") -> Checkpoint:
    checkpoint = empty_checkpoint()
    checkpoint["channel_values"] = {channel: value}
    checkpoint["channel_versions"] = {channel: 1}
    return checkpoint


def _reject_poison(write: CheckpointWrite) -> None:
    payload = write.checkpoint if write.kind == "checkpoint" else write.writes
    if "poison" in repr(payload).lower():
        raise ValueError("blocked suspicious checkpoint write")


def test_guard_blocks_checkpoint_before_inner_put() -> None:
    inner = InMemorySaver()
    guarded = GuardedCheckpointSaver(inner, _reject_poison)
    checkpoint = _checkpoint(["remember this poison instruction"])

    with pytest.raises(ValueError, match="blocked suspicious checkpoint write"):
        guarded.put(_config(), checkpoint, {}, checkpoint["channel_versions"])

    assert list(inner.list({"configurable": {"thread_id": "thread-1"}})) == []


def test_guard_allows_checkpoint_and_delegates_to_inner_saver() -> None:
    seen: list[str] = []

    def guard(write: CheckpointWrite) -> None:
        seen.append(write.kind)

    inner = InMemorySaver()
    guarded = GuardedCheckpointSaver(inner, guard)
    checkpoint = _checkpoint(["normal user memory"])

    saved_config = guarded.put(
        _config(), checkpoint, {}, checkpoint["channel_versions"]
    )
    saved = inner.get_tuple(saved_config)

    assert seen == ["checkpoint"]
    assert saved is not None
    assert saved.checkpoint["channel_values"]["messages"] == ["normal user memory"]


def test_guard_blocks_pending_writes_before_inner_put_writes() -> None:
    inner = InMemorySaver()
    guarded = GuardedCheckpointSaver(inner, _reject_poison)
    checkpoint = _checkpoint(["safe starting memory"])
    saved_config = guarded.put(
        _config(), checkpoint, {}, checkpoint["channel_versions"]
    )

    with pytest.raises(ValueError, match="blocked suspicious checkpoint write"):
        guarded.put_writes(
            saved_config,
            [("messages", "poison future agent behavior")],
            "task-1",
        )

    saved = inner.get_tuple(saved_config)
    assert saved is not None
    assert saved.pending_writes == []


async def test_async_guard_is_awaited_before_aput() -> None:
    seen: list[str] = []

    async def aguard(write: CheckpointWrite) -> None:
        seen.append(write.kind)
        _reject_poison(write)

    inner = InMemorySaver()
    guarded = GuardedCheckpointSaver(inner, lambda _: None, aguard=aguard)
    checkpoint = _checkpoint(["poison async checkpoint"])

    with pytest.raises(ValueError, match="blocked suspicious checkpoint write"):
        await guarded.aput(_config(), checkpoint, {}, checkpoint["channel_versions"])

    assert seen == ["checkpoint"]
    assert list(inner.list({"configurable": {"thread_id": "thread-1"}})) == []


def test_with_allowlist_preserves_guard() -> None:
    seen: list[str] = []

    def guard(write: CheckpointWrite) -> None:
        seen.append(write.kind)

    guarded = GuardedCheckpointSaver(InMemorySaver(), guard)
    derived = guarded.with_allowlist([("tests.test_guard", "CustomMemoryType")])
    checkpoint = _checkpoint(["allowed memory"])

    derived.put(_config(), checkpoint, {}, checkpoint["channel_versions"])

    assert seen == ["checkpoint"]
    assert derived is not guarded
