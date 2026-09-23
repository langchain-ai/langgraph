from __future__ import annotations

from typing import Any

import pytest
from langgraph.checkpoint.base import (
    BaseCheckpointSaver,
    Checkpoint,
    DeltaChannelHistory,
    empty_checkpoint,
)

from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

CHANNEL = "ch"
CONFIG: dict[str, Any] = {"configurable": {"thread_id": "t", "checkpoint_ns": ""}}
EXPECTED: DeltaChannelHistory = {
    "writes": [("task", CHANNEL, "write-root")],
    "seed": "seed",
}


def _checkpoint(checkpoint_id: str, values: dict[str, Any]) -> Checkpoint:
    value = empty_checkpoint()
    value["id"] = checkpoint_id
    value["channel_values"] = values
    return value


PARENT_ID_ORDERS = [
    pytest.param("z-older", "a-newer", id="parent_id_sorts_above_child"),
    pytest.param("a-older", "z-newer", id="parent_id_sorts_below_child"),
]


@pytest.mark.parametrize(("root_id", "child_id"), PARENT_ID_ORDERS)
def test_sync_walk_reaches_parent_whatever_the_id_order(
    root_id: str, child_id: str
) -> None:
    with SqliteSaver.from_conn_string(":memory:") as saver:
        root = saver.put(CONFIG, _checkpoint(root_id, {CHANNEL: "seed"}), {}, {})
        saver.put_writes(root, [(CHANNEL, "write-root")], "task")
        child = saver.put(root, _checkpoint(child_id, {}), {}, {})

        got = saver.get_delta_channel_history(config=child, channels=[CHANNEL])
        reference = BaseCheckpointSaver.get_delta_channel_history(
            saver, config=child, channels=[CHANNEL]
        )
        assert got[CHANNEL] == EXPECTED
        assert got[CHANNEL] == reference[CHANNEL], "fast path disagrees with base"


@pytest.mark.parametrize(("root_id", "child_id"), PARENT_ID_ORDERS)
async def test_async_walk_reaches_parent_whatever_the_id_order(
    root_id: str, child_id: str
) -> None:
    async with AsyncSqliteSaver.from_conn_string(":memory:") as saver:
        root = await saver.aput(CONFIG, _checkpoint(root_id, {CHANNEL: "seed"}), {}, {})
        await saver.aput_writes(root, [(CHANNEL, "write-root")], "task")
        child = await saver.aput(root, _checkpoint(child_id, {}), {}, {})

        got = await saver.aget_delta_channel_history(config=child, channels=[CHANNEL])
        assert got[CHANNEL] == EXPECTED


def test_walk_reaches_root_of_long_chain_with_descending_ids() -> None:
    steps = 40
    with SqliteSaver.from_conn_string(":memory:") as saver:
        parent = saver.put(
            CONFIG, _checkpoint(f"id-{steps:03d}", {CHANNEL: "seed"}), {}, {}
        )
        saver.put_writes(parent, [(CHANNEL, "write-root")], "task")
        for step in range(steps - 1, 0, -1):
            parent = saver.put(parent, _checkpoint(f"id-{step:03d}", {}), {}, {})

        got = saver.get_delta_channel_history(config=parent, channels=[CHANNEL])
        assert got[CHANNEL] == EXPECTED


def test_walk_terminates_when_put_makes_the_parent_chain_cycle() -> None:
    with SqliteSaver.from_conn_string(":memory:") as saver:
        a = saver.put(CONFIG, _checkpoint("cid-a", {}), {}, {})
        b = saver.put(a, _checkpoint("cid-b", {}), {}, {})
        repoint_a_under_b = _checkpoint("cid-a", {})
        saver.put(b, repoint_a_under_b, {}, {})

        got = saver.get_delta_channel_history(config=b, channels=[CHANNEL])
        assert got[CHANNEL] == {"writes": []}
