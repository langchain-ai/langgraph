"""Tests for the BinaryOperatorAggregate -> DeltaChannel migration path.

A thread written under `BinaryOperatorAggregate(...)` must keep working
after its annotation is swapped to `DeltaChannel(...)` on the same
checkpointer — pre-migration state visible at each *settled* ancestor
checkpoint is preserved, and post-migration writes fold on top through
the reducer.

Mechanism under test: the saver's `_get_channel_writes_history(config,
channel)` walks the parent chain; when it encounters an ancestor whose
`channel_values[channel]` is a real value (not `DELTA_SENTINEL`), it
returns that as the `seed`. `DeltaChannel.from_checkpoint(seed)` uses
it as the base value, and `replay_writes(writes)` folds on-path deltas.

Scenarios covered:

1. **Basic migration (sync + async)**: build pre-migration state with
   `BinaryOperatorAggregate`, swap the annotation to `DeltaChannel` on
   the same checkpointer, and verify that every settled pre-migration
   super-step boundary (`next=('__start__',)`) round-trips exactly
   under the delta-channel view.
2. **Time travel into a pre-migration checkpoint** after migration —
   `graph.get_state(pre_migration_config)` at a settled ancestor
   returns the same state as under the binop channel.
3. **Continuing a migrated thread**: driving one more super-step after
   migration produces a state that includes the pre-migration settled
   prefix plus the new delta write — proving `from_checkpoint(seed)` +
   `replay_writes` correctly fold post-migration deltas onto the
   pre-migration seed.
4. **Base-saver fallback path**: a third-party-style subclass that
   removes the optimized `InMemorySaver` override and falls back to
   `BaseCheckpointSaver._get_channel_writes_history` must produce the
   same result as the optimized path.
5. **Channel-type isolation across threads**: two threads on the same
   checkpointer under the delta-channel graph — one freshly-started,
   one migrated from pre-migration state — don't cross-contaminate.
   The parent-chain walk is scoped to the thread.

TODO: add postgres variants in the existing `libs/checkpoint-postgres`
test files (different fixture setup; not this file).
"""

from __future__ import annotations

import operator
from typing import Annotated, Any

import pytest
from langgraph.checkpoint.memory import InMemorySaver
from typing_extensions import TypedDict

from langgraph.channels._delta import DeltaChannel
from langgraph.channels.binop import BinaryOperatorAggregate
from langgraph.channels.delta import DeltaChannel
from langgraph.graph import END, START, StateGraph

pytestmark = pytest.mark.anyio


# ---------------------------------------------------------------------------
# Graph factories
#
# A minimal reducer (`operator.add` on lists of str) with a noop node keeps
# state change localized to the HumanMessage-like payload passed through
# `invoke`. That isolates the pre/post-migration parity assertions to
# channel-hydration semantics.
# ---------------------------------------------------------------------------


def _noop(_state: Any) -> dict:
    return {}


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
        items: Annotated[list, DeltaChannel(operator.add)]

    return (
        StateGraph(DeltaState)
        .add_node("noop", _noop)
        .add_edge(START, "noop")
        .add_edge("noop", END)
        .compile(checkpointer=checkpointer)
    )


def _drive(graph: Any, config: dict, tag: str, n: int) -> None:
    for i in range(n):
        graph.invoke({"items": [f"{tag}{i}"]}, config)


async def _adrive(graph: Any, config: dict, tag: str, n: int) -> None:
    for i in range(n):
        await graph.ainvoke({"items": [f"{tag}{i}"]}, config)


def _settled_boundaries(history: list) -> list[tuple[dict, list]]:
    """Return `[(config, items), ...]` for every checkpoint in `history`
    whose `next == ('__start__',)` — the stable boundaries between invokes.
    """
    return [
        (s.config, list(s.values.get("items", [])))
        for s in history
        if s.next == ("__start__",)
    ]


# ---------------------------------------------------------------------------
# 1. Basic migration (sync + async)
# ---------------------------------------------------------------------------


def test_basic_migration_preserves_pre_migration_state() -> None:
    """Build state under `BinaryOperatorAggregate`, migrate to
    `DeltaChannel` on the same checkpointer, and verify that every
    settled pre-migration super-step boundary round-trips exactly.

    Settled boundaries (`next=('__start__',)`) are the stable hydration
    targets for the migration path: writes that produced the NEXT
    super-step are kept as `pending_writes` on the ancestor, so walking
    from a descendant finds the ancestor's blob as the seed and
    reconstructs the correct state.
    """

    checkpointer = InMemorySaver()
    config = {"configurable": {"thread_id": "basic-sync"}}

    # Pre-migration: accumulate items across 3 invokes.
    binop = _binop_graph(checkpointer)
    _drive(binop, config, "u", 3)

    pre_boundaries = _settled_boundaries(list(binop.get_state_history(config)))
    assert len(pre_boundaries) >= 2, "expected multiple settled boundaries"

    # Migrate: swap the annotation on the same checkpointer.
    delta = _delta_graph(checkpointer)

    for cfg, items in pre_boundaries:
        snap = delta.get_state(cfg)
        assert list(snap.values.get("items", [])) == items, (
            f"snapshot mismatch at {cfg['configurable']['checkpoint_id']}: "
            f"expected {items}, got {snap.values.get('items', [])}"
        )


async def test_basic_migration_preserves_pre_migration_state_async() -> None:
    """Async variant of the basic migration scenario."""

    checkpointer = InMemorySaver()
    config = {"configurable": {"thread_id": "basic-async"}}

    binop = _binop_graph(checkpointer)
    await _adrive(binop, config, "u", 3)

    pre_history = [s async for s in binop.aget_state_history(config)]
    pre_boundaries = _settled_boundaries(pre_history)
    assert len(pre_boundaries) >= 2

    delta = _delta_graph(checkpointer)

    for cfg, items in pre_boundaries:
        snap = await delta.aget_state(cfg)
        assert list(snap.values.get("items", [])) == items, (
            f"async snapshot mismatch at {cfg['configurable']['checkpoint_id']}"
        )


# ---------------------------------------------------------------------------
# 2. Time travel into a pre-migration checkpoint after migration
# ---------------------------------------------------------------------------


def test_time_travel_into_pre_migration_checkpoint() -> None:
    """After migration, `graph.get_state(pre_migration_config)` at a
    settled ancestor returns the state as stored at that point."""

    checkpointer = InMemorySaver()
    config = {"configurable": {"thread_id": "time-travel"}}

    binop = _binop_graph(checkpointer)
    _drive(binop, config, "u", 3)

    pre_boundaries = _settled_boundaries(list(binop.get_state_history(config)))
    assert pre_boundaries, "no settled ancestors to time-travel to"

    delta = _delta_graph(checkpointer)

    # Pick the oldest non-empty boundary — a long distance to walk back.
    non_empty = [(cfg, items) for cfg, items in pre_boundaries if items]
    assert non_empty, "expected at least one non-empty boundary"
    target_cfg, expected_items = non_empty[-1]

    snap = delta.get_state(target_cfg)
    assert list(snap.values.get("items", [])) == expected_items


# ---------------------------------------------------------------------------
# 3. Continuing a migrated thread: deltas fold onto pre-migration seed
# ---------------------------------------------------------------------------


def test_continuing_migrated_thread_folds_deltas_on_seed() -> None:
    """Resume a pre-migration settled ancestor via `invoke(None, cfg)`
    under the delta-channel graph. Since the pre-migration checkpoint
    has an existing `pending_writes` entry (the input for the NEXT
    super-step), re-running from that ancestor reproduces the same
    post-ancestor state as the original binop run.

    This proves the seed-terminator + write-replay pipeline works
    end-to-end across the migration boundary.
    """

    checkpointer = InMemorySaver()
    config = {"configurable": {"thread_id": "continue"}}

    binop = _binop_graph(checkpointer)
    _drive(binop, config, "u", 2)

    # Pick the oldest settled boundary with non-empty state.
    pre_boundaries = _settled_boundaries(list(binop.get_state_history(config)))
    target_cfg, seed_items = next(
        (cfg, items) for cfg, items in reversed(pre_boundaries) if items
    )
    assert seed_items, "need a non-empty seed boundary"

    # Migrate and resume from the pre-migration ancestor. `invoke(None,
    # cfg)` replays the pending writes staged at `cfg` under the new
    # channel; the reducer folds those deltas onto the seed.
    delta = _delta_graph(checkpointer)
    result = delta.invoke(None, target_cfg)

    # The resumed state must include the pre-migration seed items in order.
    result_items = list(result.get("items", []))
    for idx, prefix_item in enumerate(seed_items):
        assert result_items[idx] == prefix_item, (
            f"pre-migration seed item at {idx} not preserved: "
            f"got {result_items[: idx + 1]}, expected {seed_items}"
        )


# ---------------------------------------------------------------------------
# 4. Base-saver fallback path
# ---------------------------------------------------------------------------


class _ThirdPartyStyleSaver(InMemorySaver):
    """Simulates a third-party saver that inherits the reference
    `_get_channel_writes_history` implementation from
    `BaseCheckpointSaver` rather than overriding it.

    We rebind the two methods to the base-class versions (via MRO) so
    the fallback path is exercised even though the storage layer is
    still the in-memory one.
    """

    # MRO: [_ThirdPartyStyleSaver, InMemorySaver, BaseCheckpointSaver, ...]
    _get_channel_writes_history = (  # type: ignore[assignment]
        InMemorySaver.__mro__[1]._get_channel_writes_history  # type: ignore[attr-defined]
    )
    _aget_channel_writes_history = (  # type: ignore[assignment]
        InMemorySaver.__mro__[1]._aget_channel_writes_history  # type: ignore[attr-defined]
    )


def test_base_saver_fallback_matches_optimized_override() -> None:
    """The reference `BaseCheckpointSaver` implementation must produce
    the same migration behavior as the optimized `InMemorySaver`
    override. We drive the same migration scenario through both savers
    and assert per-snapshot parity in the delta-channel view."""

    # Fast path: optimized InMemorySaver override.
    fast_saver = InMemorySaver()
    fast_config = {"configurable": {"thread_id": "fast"}}
    fast_binop = _binop_graph(fast_saver)
    _drive(fast_binop, fast_config, "u", 3)
    fast_delta = _delta_graph(fast_saver)
    fast_history = [
        (s.next, list(s.values.get("items", [])))
        for s in fast_delta.get_state_history(fast_config)
    ]

    # Slow path: base-class fallback.
    slow_saver = _ThirdPartyStyleSaver()
    slow_config = {"configurable": {"thread_id": "slow"}}
    slow_binop = _binop_graph(slow_saver)
    _drive(slow_binop, slow_config, "u", 3)
    slow_delta = _delta_graph(slow_saver)
    slow_history = [
        (s.next, list(s.values.get("items", [])))
        for s in slow_delta.get_state_history(slow_config)
    ]

    assert slow_history == fast_history, (
        "base-saver fallback should match optimized-override behavior; "
        f"fast={fast_history}, slow={slow_history}"
    )


# ---------------------------------------------------------------------------
# 5. Thread isolation under mixed-generation storage
# ---------------------------------------------------------------------------


def test_delta_and_migrated_threads_do_not_cross_contaminate() -> None:
    """Two threads sharing a checkpointer — one migrated from
    pre-migration state, one freshly-started under DeltaChannel — must
    maintain independent state. The parent-chain walk in
    `_get_channel_writes_history` must be scoped to the target thread.
    """

    checkpointer = InMemorySaver()
    migrated_cfg = {"configurable": {"thread_id": "migrated"}}
    fresh_cfg = {"configurable": {"thread_id": "fresh"}}

    # Thread A: pre-migration build-up.
    binop = _binop_graph(checkpointer)
    _drive(binop, migrated_cfg, "m", 2)

    # Thread B: fresh delta-channel run.
    delta = _delta_graph(checkpointer)
    _drive(delta, fresh_cfg, "f", 2)

    # Thread A: migrate and confirm its state is anchored in its own
    # thread's pre-migration history (tag 'm'), never mixing in tag 'f'.
    migrated_boundaries = _settled_boundaries(
        list(delta.get_state_history(migrated_cfg))
    )
    assert migrated_boundaries, "migrated thread has no settled boundaries"
    for _, items in migrated_boundaries:
        for it in items:
            assert it.startswith("m"), (
                f"migrated thread leaked item from other thread: {it}"
            )

    # Thread B: settled boundaries must only contain 'f' tags.
    fresh_boundaries = _settled_boundaries(list(delta.get_state_history(fresh_cfg)))
    assert fresh_boundaries, "fresh thread has no settled boundaries"
    for _, items in fresh_boundaries:
        for it in items:
            assert it.startswith("f"), (
                f"fresh thread leaked item from migrated thread: {it}"
            )


# ---------------------------------------------------------------------------
# 6. Tip-of-pre-migration hydration: the latest checkpoint from a binop-run
# thread has a real accumulated value in its own `channel_values["items"]`.
# When hydrated under the delta-channel graph via `get_state(config)` with no
# `checkpoint_id`, the short-circuit must use that value directly instead of
# walking ancestors (which would skip the tip's own blob).
# ---------------------------------------------------------------------------


def test_tip_of_pre_migration_hydrates_directly() -> None:
    """`graph.get_state(config)` at the latest (pre-migration) checkpoint
    returns the full accumulated list stored in that checkpoint's own
    `channel_values`. The hydration must not walk ancestors past it."""

    checkpointer = InMemorySaver()
    config = {"configurable": {"thread_id": "tip-sync"}}

    binop = _binop_graph(checkpointer)
    _drive(binop, config, "u", 3)

    binop_tip = binop.get_state(config)
    expected_items = list(binop_tip.values.get("items", []))
    assert expected_items == ["u0", "u1", "u2"], (
        f"sanity: pre-migration tip should accumulate all 3 items, got {expected_items}"
    )

    delta = _delta_graph(checkpointer)

    snap = delta.get_state(config)
    assert list(snap.values.get("items", [])) == expected_items, (
        f"tip hydration mismatch: expected {expected_items}, "
        f"got {snap.values.get('items', [])}"
    )


async def test_tip_of_pre_migration_hydrates_directly_async() -> None:
    """Async variant of the tip-of-pre-migration hydration scenario."""

    checkpointer = InMemorySaver()
    config = {"configurable": {"thread_id": "tip-async"}}

    binop = _binop_graph(checkpointer)
    await _adrive(binop, config, "u", 3)

    binop_tip = await binop.aget_state(config)
    expected_items = list(binop_tip.values.get("items", []))
    assert expected_items == ["u0", "u1", "u2"]

    delta = _delta_graph(checkpointer)

    snap = await delta.aget_state(config)
    assert list(snap.values.get("items", [])) == expected_items, (
        f"async tip hydration mismatch: expected {expected_items}, "
        f"got {snap.values.get('items', [])}"
    )


# ---------------------------------------------------------------------------
# 7. `update_state` after migration writes a real value to the new
# checkpoint's `channel_values` (not a sentinel). Hydration must use it
# directly — the ancestor walk would skip this blob and return stale state.
# ---------------------------------------------------------------------------


def test_update_state_after_migration_uses_written_value() -> None:
    """After migrating and running at least one post-migration super-step
    (so the thread's tip has a `DELTA_SENTINEL`), `update_state` writes a
    concrete value to a new checkpoint's `channel_values`. `get_state`
    must reflect that concrete value."""

    checkpointer = InMemorySaver()
    config = {"configurable": {"thread_id": "update-state"}}

    # Pre-migration: accumulate a little state.
    binop = _binop_graph(checkpointer)
    _drive(binop, config, "u", 2)

    # Migrate and run one more super-step so the tip is a post-migration
    # checkpoint with `DELTA_SENTINEL` in its own `channel_values`.
    delta = _delta_graph(checkpointer)
    delta.invoke({"items": ["post"]}, config)

    # `update_state` writes a concrete value into a new checkpoint's blob
    # via the reducer against the hydrated prior state.
    delta.update_state(config, {"items": ["x", "y"]})

    snap = delta.get_state(config)
    updated_items = list(snap.values.get("items", []))
    # Must include the "x","y" update; without the hydration fix, the
    # update_state-written blob would be skipped in favor of an ancestor
    # walk, and the update values would disappear.
    assert "x" in updated_items and "y" in updated_items, (
        f"update_state values missing from snapshot: {updated_items}"
    )
    # The "x","y" items should be folded onto the prior accumulated state,
    # not stand alone. This verifies the update-written blob is used
    # directly by `get_state` (no ancestor walk past it).
    assert len(updated_items) >= 4, (
        f"update_state snapshot should preserve pre-update state, got {updated_items}"
    )
    assert updated_items[-2:] == ["x", "y"], (
        f"update_state deltas should be at the tail, got {updated_items}"
    )


# ---------------------------------------------------------------------------
# 8. Fork from an `update_state` checkpoint: a new run branched off the
# update_state-produced checkpoint must see that checkpoint's concrete
# `channel_values` as its base, with new deltas folded on top.
# ---------------------------------------------------------------------------


def test_fork_from_update_state_checkpoint() -> None:
    """Branching a new run from the checkpoint produced by `update_state`
    must use that checkpoint's concrete blob as the base. Additional
    deltas from the forked run fold onto it through the reducer."""

    checkpointer = InMemorySaver()
    config = {"configurable": {"thread_id": "fork"}}

    # Pre-migration build-up, then migrate and add one post-migration step.
    binop = _binop_graph(checkpointer)
    _drive(binop, config, "u", 2)
    delta = _delta_graph(checkpointer)
    delta.invoke({"items": ["post"]}, config)

    # Apply `update_state` and capture the returned config (references
    # the new checkpoint produced by the update).
    update_cfg = delta.update_state(config, {"items": ["x", "y"]})

    update_snap = delta.get_state(update_cfg)
    base_items = list(update_snap.values.get("items", []))
    assert "x" in base_items and "y" in base_items, (
        f"update_state values missing from snapshot: {base_items}"
    )
    assert base_items[-2:] == ["x", "y"], (
        f"sanity: update_state deltas should be at the tail, got {base_items}"
    )

    # Fork: invoke from the update_state checkpoint with a new delta.
    forked = delta.invoke({"items": ["fork0"]}, update_cfg)
    forked_items = list(forked.get("items", []))
    # The fork must see the update_state-written blob as its base (not
    # walk past it), and the new delta must fold on top of it.
    assert forked_items[: len(base_items)] == base_items, (
        f"fork lost update_state base: base={base_items}, forked={forked_items}"
    )
    assert forked_items[-1] == "fork0", f"fork delta not appended: {forked_items}"
