"""Smoke tests for `BaseCheckpointSaver.get_delta_channel_history` on sqlite.

`SqliteSaver` (and `AsyncSqliteSaver`) deliberately don't override the
default implementation in `BaseCheckpointSaver` — these tests pin the
default impl to behave correctly end-to-end against a real persistent
saver and a real `DeltaChannel`-backed graph.

Scenarios covered:

* Empty `channels` argument returns an empty mapping (no I/O).
* On a non-trivial multi-checkpoint thread, per-channel writes come back
  oldest→newest.
* When the walk reaches the root without ever finding a stored value,
  `seed` is omitted from the entry (consumer treats absence as "start
  empty").
* When a `_DeltaSnapshot` blob is present at an ancestor, it is returned
  as the `seed`.
* The async saver returns the same shape via `aget_delta_channel_history`.
"""

from __future__ import annotations

import operator
from typing import Annotated, Any

import pytest
from langchain_core.runnables import RunnableConfig

# `langgraph` is not a dep of `langgraph-checkpoint-sqlite`. When tests run
# in the sqlite lib's standalone CI environment without it installed, skip
# the whole module rather than failing at import.
pytest.importorskip("langgraph.channels.delta", reason="langgraph core not installed")
pytest.importorskip("langgraph.graph", reason="langgraph core not installed")

from langgraph.channels.delta import DeltaChannel  # type: ignore[import-untyped]  # noqa: E402,I001
from langgraph.checkpoint.serde.types import _DeltaSnapshot  # noqa: E402
from langgraph.graph import END, START, StateGraph  # type: ignore[import-untyped]  # noqa: E402
from typing_extensions import TypedDict  # noqa: E402

from langgraph.checkpoint.sqlite import SqliteSaver  # noqa: E402
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver  # noqa: E402

pytestmark = pytest.mark.anyio


# ---------------------------------------------------------------------------
# Graph helpers
# ---------------------------------------------------------------------------


def _noop(_state: Any) -> dict[str, Any]:
    return {}


class _DeltaState(TypedDict):
    items: Annotated[list, DeltaChannel(operator.add)]


def _delta_graph(checkpointer: Any) -> Any:
    return (
        StateGraph(_DeltaState)
        .add_node("noop", _noop)
        .add_edge(START, "noop")
        .add_edge("noop", END)
        .compile(checkpointer=checkpointer)
    )


def _drive(graph: Any, config: RunnableConfig, n: int) -> None:
    for i in range(n):
        graph.invoke({"items": [f"v{i}"]}, config)


async def _adrive(graph: Any, config: RunnableConfig, n: int) -> None:
    for i in range(n):
        await graph.ainvoke({"items": [f"v{i}"]}, config)


def _pick_non_root(saver: Any, config: RunnableConfig) -> RunnableConfig:
    """Return a config pointing at a checkpoint that has at least one ancestor.

    `get_delta_channel_history` walks the parent chain — calling it on the root
    checkpoint produces `writes=[]` and no `seed`, which is uninteresting
    for the multi-step assertions below.
    """
    history = list(saver.list(config))
    assert history, "expected non-empty history"
    # `list` yields newest→oldest; the second entry has the first entry
    # as its parent, so its parent_config is non-None.
    for tup in history:
        if tup.parent_config is not None:
            return tup.config
    raise AssertionError("no checkpoint with a parent in history")


async def _apick_non_root(saver: Any, config: RunnableConfig) -> RunnableConfig:
    history = [tup async for tup in saver.alist(config)]
    assert history, "expected non-empty history"
    for tup in history:
        if tup.parent_config is not None:
            return tup.config
    raise AssertionError("no checkpoint with a parent in history")


# ---------------------------------------------------------------------------
# Sync: SqliteSaver
# ---------------------------------------------------------------------------


def test_empty_channels_returns_empty_mapping_sync() -> None:
    """Empty `channels` short-circuits to `{}` without touching storage."""
    with SqliteSaver.from_conn_string(":memory:") as saver:
        config: RunnableConfig = {"configurable": {"thread_id": "empty"}}
        assert saver.get_delta_channel_history(config=config, channels=[]) == {}


def test_writes_history_oldest_to_newest_sync() -> None:
    """Per-channel writes accumulated across the walk come back oldest→newest."""
    with SqliteSaver.from_conn_string(":memory:") as saver:
        config: RunnableConfig = {"configurable": {"thread_id": "history-sync"}}
        graph = _delta_graph(saver)
        _drive(graph, config, 3)

        target_cfg = _pick_non_root(saver, config)
        result = saver.get_delta_channel_history(config=target_cfg, channels=["items"])

        assert "items" in result
        entry = result["items"]
        assert isinstance(entry["writes"], list)

        # If any writes were collected, their values should be in oldest→newest
        # order — i.e. tagged 'v0', 'v1', ... matching invoke order.
        write_values: list[Any] = []
        for _task_id, channel, value in entry["writes"]:
            assert channel == "items"
            write_values.extend(value if isinstance(value, list) else [value])

        # `_drive` invokes with payloads ['v0'], ['v1'], ['v2']. Whatever
        # subset shows up in the chain must be a contiguous prefix in order.
        for idx, val in enumerate(write_values):
            assert val == f"v{idx}", (
                f"writes not in oldest→newest order: {write_values}"
            )


def test_seed_present_when_snapshot_in_ancestor_sync() -> None:
    """Inserting a `_DeltaSnapshot` blob at an ancestor → walk returns it as `seed`."""
    with SqliteSaver.from_conn_string(":memory:") as saver:
        config: RunnableConfig = {"configurable": {"thread_id": "seed-sync"}}
        graph = _delta_graph(saver)
        _drive(graph, config, 2)

        # Find the oldest non-root checkpoint, then walk to its parent and
        # rewrite that parent's `channel_values["items"]` to a real
        # `_DeltaSnapshot`. After this surgery, calling `get_delta_channel_history`
        # at the leaf must return the snapshot value as `seed`.
        history = list(saver.list(config))
        assert len(history) >= 2
        leaf_tup = history[0]
        # Walk to an ancestor with a parent_config (any non-root will do).
        ancestor_tup = next(
            (tup for tup in history if tup.parent_config is not None), None
        )
        assert ancestor_tup is not None
        parent_cfg = ancestor_tup.parent_config
        assert parent_cfg is not None
        parent_tup = saver.get_tuple(parent_cfg)
        assert parent_tup is not None

        snapshot_value = ["seeded", "items"]
        parent_tup.checkpoint["channel_values"]["items"] = _DeltaSnapshot(
            snapshot_value
        )
        # Make sure the channel has a version so the optimized blob lookup
        # in any future override has something to hit.
        parent_tup.checkpoint["channel_versions"].setdefault("items", 1)
        saver.put(
            parent_tup.parent_config or {"configurable": parent_cfg["configurable"]},
            parent_tup.checkpoint,
            parent_tup.metadata,
            {},
        )

        result = saver.get_delta_channel_history(
            config=leaf_tup.config, channels=["items"]
        )
        entry = result["items"]
        assert "seed" in entry, f"expected seed to be present, got {entry}"
        seed = entry["seed"]
        assert isinstance(seed, _DeltaSnapshot), (
            f"expected _DeltaSnapshot, got {seed!r}"
        )
        assert seed.value == snapshot_value


def test_seed_omitted_when_walk_reaches_root_sync() -> None:
    """`get_delta_channel_history` on the root checkpoint → no `seed` key, no writes."""
    with SqliteSaver.from_conn_string(":memory:") as saver:
        config: RunnableConfig = {"configurable": {"thread_id": "root-sync"}}
        graph = _delta_graph(saver)
        _drive(graph, config, 1)

        history = list(saver.list(config))
        # Root is the oldest checkpoint (no parent_config).
        root_tup = history[-1]
        assert root_tup.parent_config is None

        result = saver.get_delta_channel_history(
            config=root_tup.config, channels=["items"]
        )
        entry = result["items"]
        assert "seed" not in entry, f"root-walk should have no seed, got {entry}"
        assert entry["writes"] == []


# ---------------------------------------------------------------------------
# Async: AsyncSqliteSaver
# ---------------------------------------------------------------------------


async def test_empty_channels_returns_empty_mapping_async() -> None:
    """Async equivalent of the empty-channels short-circuit."""
    async with AsyncSqliteSaver.from_conn_string(":memory:") as saver:
        config: RunnableConfig = {"configurable": {"thread_id": "empty-async"}}
        assert await saver.aget_delta_channel_history(config=config, channels=[]) == {}


async def test_writes_history_oldest_to_newest_async() -> None:
    """Async equivalent of the oldest→newest ordering check."""
    async with AsyncSqliteSaver.from_conn_string(":memory:") as saver:
        config: RunnableConfig = {"configurable": {"thread_id": "history-async"}}
        graph = _delta_graph(saver)
        await _adrive(graph, config, 3)

        target_cfg = await _apick_non_root(saver, config)
        result = await saver.aget_delta_channel_history(
            config=target_cfg, channels=["items"]
        )

        assert "items" in result
        entry = result["items"]
        assert isinstance(entry["writes"], list)

        write_values: list[Any] = []
        for _task_id, channel, value in entry["writes"]:
            assert channel == "items"
            write_values.extend(value if isinstance(value, list) else [value])

        for idx, val in enumerate(write_values):
            assert val == f"v{idx}", (
                f"writes not in oldest→newest order: {write_values}"
            )


async def test_seed_omitted_when_walk_reaches_root_async() -> None:
    """Async equivalent of the root-walk seed-absence check."""
    async with AsyncSqliteSaver.from_conn_string(":memory:") as saver:
        config: RunnableConfig = {"configurable": {"thread_id": "root-async"}}
        graph = _delta_graph(saver)
        await _adrive(graph, config, 1)

        history = [tup async for tup in saver.alist(config)]
        root_tup = history[-1]
        assert root_tup.parent_config is None

        result = await saver.aget_delta_channel_history(
            config=root_tup.config, channels=["items"]
        )
        entry = result["items"]
        assert "seed" not in entry, f"root-walk should have no seed, got {entry}"
        assert entry["writes"] == []
