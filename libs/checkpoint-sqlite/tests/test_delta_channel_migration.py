"""Sqlite-specific migration smoke tests: BinaryOperatorAggregate -> DeltaChannel.

Mirrors `libs/langgraph/tests/test_delta_channel_migration.py` (which
covers `InMemorySaver` + a third-party fallback to the base default
impl). This file exercises the same migration scenario through the
sqlite-specific `SqliteSaver.get_delta_channel_history` override —
specifically that the streaming ancestor walk finds a pre-migration
plain `channel_values[ch]` entry and surfaces it as the `seed`, with
post-migration writes folding on top through the reducer.

Pre-migration checkpoints under `BinaryOperatorAggregate` carry the
full accumulated value at every settled super-step boundary. The
override has to identify those as "real" seeds (not `_DeltaSnapshot`
sentinels) — the saver layer is intentionally delta-agnostic and just
returns whatever is stored in `channel_values[ch]`.
"""

from __future__ import annotations

import operator
from typing import Annotated, Any

import pytest
from langchain_core.runnables import RunnableConfig

# `langgraph` core isn't a dep of `langgraph-checkpoint-sqlite`. Skip the
# whole module rather than importerror-ing in the standalone CI shape.
pytest.importorskip("langgraph.channels.delta", reason="langgraph core not installed")
pytest.importorskip("langgraph.channels.binop", reason="langgraph core not installed")
pytest.importorskip("langgraph.graph", reason="langgraph core not installed")

from langgraph.channels.binop import BinaryOperatorAggregate  # type: ignore[import-untyped]  # noqa: E402,I001
from langgraph.channels.delta import DeltaChannel  # type: ignore[import-untyped]  # noqa: E402
from langgraph.graph import END, START, StateGraph  # type: ignore[import-untyped]  # noqa: E402
from typing_extensions import TypedDict  # noqa: E402

from langgraph.checkpoint.sqlite import SqliteSaver  # noqa: E402
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver  # noqa: E402

pytestmark = pytest.mark.anyio


def _noop(_state: Any) -> dict:
    return {}


def _list_concat(state: list, writes: list) -> list:
    result = list(state)
    for w in writes:
        result.extend(w if isinstance(w, list) else [w])
    return result


def _binop_graph(checkpointer: Any) -> Any:
    class BinopState(TypedDict):
        items: Annotated[list, BinaryOperatorAggregate(list, operator.add)]

    return (
        StateGraph(BinopState)
        .add_node("noop", _noop)
        .add_edge(START, "noop")
        .add_edge("noop", END)
        .compile(checkpointer=checkpointer)
    )


def _delta_graph(checkpointer: Any) -> Any:
    class DeltaState(TypedDict):
        items: Annotated[list, DeltaChannel(_list_concat)]

    return (
        StateGraph(DeltaState)
        .add_node("noop", _noop)
        .add_edge(START, "noop")
        .add_edge("noop", END)
        .compile(checkpointer=checkpointer)
    )


def _drive(graph: Any, config: RunnableConfig, tag: str, n: int) -> None:
    for i in range(n):
        graph.invoke({"items": [f"{tag}{i}"]}, config)


async def _adrive(graph: Any, config: RunnableConfig, tag: str, n: int) -> None:
    for i in range(n):
        await graph.ainvoke({"items": [f"{tag}{i}"]}, config)


def _settled_boundaries(history: list) -> list[tuple[RunnableConfig, list]]:
    """`(config, items)` for every checkpoint with `next == ('__start__',)`
    — the stable inter-invoke boundaries that round-trip predictably.
    """
    return [
        (s.config, list(s.values.get("items", [])))
        for s in history
        if s.next == ("__start__",)
    ]


def test_migration_preserves_pre_migration_state_sync() -> None:
    """Drive 3 invokes under `BinaryOperatorAggregate`, swap the
    annotation to `DeltaChannel` on the same sqlite-backed thread, and
    verify every settled pre-migration boundary round-trips exactly.

    The override's streaming walk must identify the plain accumulated
    list at each pre-migration ancestor as a valid `seed` even though
    no `_DeltaSnapshot` was ever written there.
    """
    with SqliteSaver.from_conn_string(":memory:") as saver:
        config: RunnableConfig = {"configurable": {"thread_id": "mig-sync"}}

        binop = _binop_graph(saver)
        _drive(binop, config, "u", 3)

        pre_boundaries = _settled_boundaries(list(binop.get_state_history(config)))
        assert len(pre_boundaries) >= 2, "expected multiple settled boundaries"

        delta = _delta_graph(saver)
        for cfg, items in pre_boundaries:
            snap = delta.get_state(cfg)
            assert list(snap.values.get("items", [])) == items, (
                f"snapshot mismatch at {cfg['configurable']['checkpoint_id']}: "
                f"expected {items}, got {snap.values.get('items', [])}"
            )


def test_migration_continued_thread_folds_deltas_on_seed_sync() -> None:
    """After migration, driving one more super-step extends the
    pre-migration accumulated state via the delta reducer — the seed
    plus a single new write.
    """
    with SqliteSaver.from_conn_string(":memory:") as saver:
        config: RunnableConfig = {"configurable": {"thread_id": "mig-continue-sync"}}

        binop = _binop_graph(saver)
        _drive(binop, config, "u", 3)

        pre_history = list(binop.get_state_history(config))
        pre_boundaries = _settled_boundaries(pre_history)
        # Latest settled boundary — the leaf pre-migration state.
        leaf_cfg, leaf_items = pre_boundaries[0]
        assert leaf_items, "expected non-empty pre-migration leaf"

        delta = _delta_graph(saver)
        delta.invoke({"items": ["after-migration"]}, leaf_cfg)
        new_state = delta.get_state(config).values["items"]
        assert new_state[: len(leaf_items)] == leaf_items
        assert "after-migration" in new_state


async def test_migration_preserves_pre_migration_state_async() -> None:
    """Async equivalent of the basic-migration round-trip check on
    `AsyncSqliteSaver`."""
    async with AsyncSqliteSaver.from_conn_string(":memory:") as saver:
        config: RunnableConfig = {"configurable": {"thread_id": "mig-async"}}

        binop = _binop_graph(saver)
        await _adrive(binop, config, "u", 3)

        pre_history = [s async for s in binop.aget_state_history(config)]
        pre_boundaries = _settled_boundaries(pre_history)
        assert len(pre_boundaries) >= 2

        delta = _delta_graph(saver)
        for cfg, items in pre_boundaries:
            snap = await delta.aget_state(cfg)
            assert list(snap.values.get("items", [])) == items, (
                f"async snapshot mismatch at {cfg['configurable']['checkpoint_id']}"
            )
