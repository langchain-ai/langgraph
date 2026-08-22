"""Regression tests for issue #8550: SQLite delta history skips parent
checkpoints with non-monotonic IDs.

When a parent checkpoint has an ID that is lexicographically *larger*
than its child's (e.g. parent='z-older', child='a-younger'), the
old `DELTA_STAGE1_SQL` range scan (`checkpoint_id <= ?`) would
silently drop the parent. The recursive CTE must walk the parent
chain via `parent_checkpoint_id` regardless of ID ordering.

These tests use only `langgraph-checkpoint` base primitives — no
dependency on `langgraph` core (channels, graph) — so they run in
sqlite's standalone CI environment.
"""

from __future__ import annotations

from typing import Any

import pytest
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.base import empty_checkpoint

from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

pytestmark = pytest.mark.anyio


def _mk_ckpt(ckpt_id: str, values: dict[str, Any]) -> dict[str, Any]:
    """Build a minimal checkpoint with a given ID and channel_values."""
    cp = empty_checkpoint()
    cp["id"] = ckpt_id
    cp["channel_values"] = values
    return cp


# ---------------------------------------------------------------------------
# Sync: SqliteSaver
# ---------------------------------------------------------------------------


def test_list_includes_non_monotonic_parent_sync() -> None:
    """list() must return every checkpoint on the thread regardless of ID order."""
    with SqliteSaver.from_conn_string(":memory:") as saver:
        config: RunnableConfig = {"configurable": {"thread_id": "t", "checkpoint_ns": ""}}

        root = saver.put(
            config,
            _mk_ckpt("z-older", {"ch": "seed"}),
            {},
            {},
        )
        saver.put_writes(root, [("ch", "write-root")], "task")

        child = saver.put(
            root,
            _mk_ckpt("a-younger", {"ch": "seed-wchild"}),
            {},
            {},
        )
        writes_cfg = child.copy()
        writes_cfg["configurable"]["task_id"] = "task"
        saver.put_writes(writes_cfg, [("ch", "write-child")], "task")

        history = list(saver.list(config))
        ids = {t.config["configurable"]["checkpoint_id"] for t in history}
        assert "z-older" in ids, (
            "parent checkpoint 'z-older' was skipped by list()"
        )
        assert "a-younger" in ids


def test_delta_history_includes_non_monotonic_parent_sync() -> None:
    """get_delta_channel_history must include writes from the
    parent whose ID is lexicographically larger than its child's."""
    with SqliteSaver.from_conn_string(":memory:") as saver:
        config: RunnableConfig = {"configurable": {"thread_id": "t", "checkpoint_ns": ""}}

        root = saver.put(
            config,
            _mk_ckpt("z-older", {"ch": "seed"}),
            {},
            {},
        )
        saver.put_writes(root, [("ch", "write-root")], "task")

        child = saver.put(
            root,
            _mk_ckpt("a-younger", {"ch": "seed-wchild"}),
            {},
            {},
        )
        writes_cfg = child.copy()
        writes_cfg["configurable"]["task_id"] = "task"
        saver.put_writes(writes_cfg, [("ch", "write-child")], "task")

        result = saver.get_delta_channel_history(config=child, channels=["ch"])
        entry = result["ch"]
        write_values: list[Any] = []
        for _task_id, channel, value in entry["writes"]:
            assert channel == "ch"
            write_values.extend(value if isinstance(value, list) else [value])

        assert "write-root" in write_values, (
            "parent write dropped: the walk skipped the 'z-older' ancestor"
        )
        assert "write-child" in write_values


def test_delta_history_multi_hop_non_monotonic_sync() -> None:
    """Three-hop chain with alternating large/small IDs."""
    with SqliteSaver.from_conn_string(":memory:") as saver:
        config: RunnableConfig = {"configurable": {"thread_id": "t-hop", "checkpoint_ns": ""}}

        root = saver.put(
            config,
            _mk_ckpt("z-grandparent", {"ch": "gp"}),
            {},
            {},
        )
        saver.put_writes(root, [("ch", "write-gp")], "task")

        mid = saver.put(
            root,
            _mk_ckpt("a-parent", {"ch": "p"}),
            {},
            {},
        )
        saver.put_writes(mid, [("ch", "write-p")], "task")

        leaf = saver.put(
            mid,
            _mk_ckpt("z-child", {"ch": "c"}),
            {},
            {},
        )
        writes_cfg = leaf.copy()
        writes_cfg["configurable"]["task_id"] = "task"
        saver.put_writes(writes_cfg, [("ch", "write-c")], "task")

        history = list(saver.list(config))
        ids = {t.config["configurable"]["checkpoint_id"] for t in history}
        assert ids == {"z-grandparent", "a-parent", "z-child"}

        result = saver.get_delta_channel_history(config=leaf, channels=["ch"])
        entry = result["ch"]
        write_values: list[Any] = []
        for _task_id, _ch, value in entry["writes"]:
            write_values.extend(value if isinstance(value, list) else [value])

        assert write_values == ["write-gp", "write-p", "write-c"]


# ---------------------------------------------------------------------------
# Async: AsyncSqliteSaver
# ---------------------------------------------------------------------------


async def test_list_includes_non_monotonic_parent_async() -> None:
    async with AsyncSqliteSaver.from_conn_string(":memory:") as saver:
        config: RunnableConfig = {"configurable": {"thread_id": "t-async", "checkpoint_ns": ""}}

        root = await saver.aput(
            config,
            _mk_ckpt("z-older", {"ch": "seed"}),
            {},
            {},
        )
        await saver.aput_writes(root, [("ch", "write-root")], "task")

        child = await saver.aput(
            root,
            _mk_ckpt("a-younger", {"ch": "seed-wchild"}),
            {},
            {},
        )
        writes_cfg = child.copy()
        writes_cfg["configurable"]["task_id"] = "task"
        await saver.aput_writes(writes_cfg, [("ch", "write-child")], "task")

        history = [tup async for tup in saver.alist(config)]
        ids = {t.config["configurable"]["checkpoint_id"] for t in history}
        assert "z-older" in ids
        assert "a-younger" in ids


async def test_delta_history_includes_non_monotonic_parent_async() -> None:
    async with AsyncSqliteSaver.from_conn_string(":memory:") as saver:
        config: RunnableConfig = {"configurable": {"thread_id": "t-async", "checkpoint_ns": ""}}

        root = await saver.aput(
            config,
            _mk_ckpt("z-older", {"ch": "seed"}),
            {},
            {},
        )
        await saver.aput_writes(root, [("ch", "write-root")], "task")

        child = await saver.aput(
            root,
            _mk_ckpt("a-younger", {"ch": "seed-wchild"}),
            {},
            {},
        )
        writes_cfg = child.copy()
        writes_cfg["configurable"]["task_id"] = "task"
        await saver.aput_writes(writes_cfg, [("ch", "write-child")], "task")

        result = await saver.aget_delta_channel_history(config=child, channels=["ch"])
        entry = result["ch"]
        write_values: list[Any] = []
        for _task_id, _ch, value in entry["writes"]:
            write_values.extend(value if isinstance(value, list) else [value])

        assert "write-root" in write_values
        assert "write-child" in write_values