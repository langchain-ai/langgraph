"""The stage-1 ancestor walk follows `parent_checkpoint_id`, not id order.

Ancestry is defined by the `parent_checkpoint_id` column. An earlier stage-1
query filtered `checkpoint_id <= target` and streamed `ORDER BY checkpoint_id
DESC`, so it also required every child's id to sort above its parent's. Real
ids are `uuid6` and happen to satisfy that, but the contract never promised it,
and a parent sorting above its child was dropped from the stream: its seed and
its writes vanished with no error. See #8550.

Following parent pointers in a recursive CTE removes both assumptions, at the
cost of needing a cycle guard, which the bounded id scan got for free.
"""

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


# `z` sorts above `a`, so the parent's id sorts above its child's. Real uuid6
# ids never do this; nothing in the contract stops a caller supplying ids that
# do, and clock skew between two processes writing one thread produces it.
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
        assert got[CHANNEL] == EXPECTED
        # The unoptimised implementation is the contract; the fast path must
        # agree with it on the same rows.
        assert (
            got[CHANNEL]
            == BaseCheckpointSaver.get_delta_channel_history(
                saver, config=child, channels=[CHANNEL]
            )[CHANNEL]
        )


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


def test_long_chain_walks_the_whole_thread() -> None:
    """A migrated thread can hold its only stored value at the root.

    The walk then legitimately runs the length of the thread, so nothing in
    the cycle check may cut it short. Ids descend as the chain grows here, so
    id order fights the walk at every step.
    """
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


def test_cyclic_parent_chain_terminates() -> None:
    """A parent chain that loops must not feed the walk forever.

    Reachable through `put` alone, no corruption needed: it writes with
    `INSERT OR REPLACE`, so re-putting an existing checkpoint id under a
    descendant's config repoints that checkpoint at its own descendant. The
    old id-range scan read a finite row set and could not loop; a recursive
    parent-pointer query can, so the walk stops on a repeated id.

    Note this one fails by hanging, not by asserting, since a regression
    means the row stream never ends. The package has no timeout plugin, so
    the CI job timeout is the backstop.
    """
    with SqliteSaver.from_conn_string(":memory:") as saver:
        a = saver.put(CONFIG, _checkpoint("cid-a", {}), {}, {})
        b = saver.put(a, _checkpoint("cid-b", {}), {}, {})
        # Re-put "cid-a" with "cid-b" as its parent: a -> b -> a.
        saver.put(b, _checkpoint("cid-a", {}), {}, {})

        got = saver.get_delta_channel_history(config=b, channels=[CHANNEL])
        # Nothing on the cycle stores a value, so no seed. The point of the
        # test is that the call returns at all.
        assert "seed" not in got[CHANNEL]
