"""DELTA_CHANNEL_RECONSTRUCTION capability tests — end-to-end round-trip.

Exercises: aput + aput_writes + aget_delta_channel_history + reconstruction.
This catches the most common silent-corruption mode: failing to round-trip
`_DeltaSnapshot` blobs through serialization.

NOTE: This test does NOT import from `langgraph` (which is not a dependency
of checkpoint-conformance). Instead it inlines a minimal reconstruction
equivalent: seed + fold writes through a simple list-append reducer.
"""

from __future__ import annotations

import traceback
from collections.abc import Callable
from typing import Any
from uuid import uuid4

from langgraph.checkpoint.base import BaseCheckpointSaver

from langgraph.checkpoint.conformance.spec._delta_fixtures import build_delta_chain


def _reconstruct(seed: Any, writes: list) -> list:
    """Minimal DeltaChannel reconstruction: list-append reducer.

    Mirrors DeltaChannel.from_checkpoint(seed) + replay_writes(writes).
    """
    from langgraph.checkpoint.serde.types import _DeltaSnapshot

    if seed is None:
        base: list = []
    elif isinstance(seed, _DeltaSnapshot):
        base = list(seed.value)
    else:
        base = list(seed)
    for _task_id, _ch, value in writes:
        base = base + value
    return base


async def test_reconstruction_basic(
    saver: BaseCheckpointSaver,
) -> None:
    """Reconstruct delta channel value from history matches expected."""
    tid = str(uuid4())
    # 5 steps: snapshot at 0 (value=[0]), writes at 1,2,3,4.
    # Head = step 4. Walk from parent (step 3) collects writes 1,2,3.
    configs = await build_delta_chain(
        saver,
        thread_id=tid,
        channel="msgs",
        snapshots_at_steps=[0],
        total_steps=5,
        write_value_fn=lambda step: [step],
    )
    head = configs[-1]
    result = await saver.aget_delta_channel_history(config=head, channels=["msgs"])
    history = result["msgs"]

    seed = history.get("seed")
    reconstructed = _reconstruct(seed, history["writes"])

    # seed=[0] from step 0, writes from steps 1,2,3
    expected = [0] + [1] + [2] + [3]
    assert reconstructed == expected, (
        f"Reconstructed {reconstructed} != expected {expected}"
    )


async def test_reconstruction_mid_chain_snapshot(
    saver: BaseCheckpointSaver,
) -> None:
    """Reconstruction works when snapshot is mid-chain."""
    tid = str(uuid4())
    # 6 steps: snapshots at 0 and 3, writes at 1,2,4,5.
    # Head = step 5. Walk from step 4 stops at step 3 (snapshot).
    # Collects writes from step 4.
    configs = await build_delta_chain(
        saver,
        thread_id=tid,
        channel="msgs",
        snapshots_at_steps=[0, 3],
        total_steps=6,
        write_value_fn=lambda step: [step],
    )
    head = configs[-1]
    result = await saver.aget_delta_channel_history(config=head, channels=["msgs"])
    history = result["msgs"]

    seed = history.get("seed")
    reconstructed = _reconstruct(seed, history["writes"])

    # Snapshot at step 3 = [3], write from step 4
    expected = [3] + [4]
    assert reconstructed == expected, (
        f"Reconstructed {reconstructed} != expected {expected}"
    )


async def test_reconstruction_no_snapshot(
    saver: BaseCheckpointSaver,
) -> None:
    """Reconstruction from root (no snapshot) gives all writes accumulated."""
    tid = str(uuid4())
    # 4 steps: no snapshot, writes at 0,1,2,3.
    # Head = step 3. Walk from step 2 collects writes 0,1,2.
    configs = await build_delta_chain(
        saver,
        thread_id=tid,
        channel="msgs",
        snapshots_at_steps=[],
        total_steps=4,
        write_value_fn=lambda step: [step],
    )
    head = configs[-1]
    result = await saver.aget_delta_channel_history(config=head, channels=["msgs"])
    history = result["msgs"]

    seed = history.get("seed")
    reconstructed = _reconstruct(seed, history["writes"])

    # No seed → start empty, writes from steps 0,1,2
    expected = [0] + [1] + [2]
    assert reconstructed == expected, (
        f"Reconstructed {reconstructed} != expected {expected}"
    )


ALL_DELTA_CHANNEL_RECONSTRUCTION_TESTS = [
    test_reconstruction_basic,
    test_reconstruction_mid_chain_snapshot,
    test_reconstruction_no_snapshot,
]


async def run_delta_channel_reconstruction_tests(
    saver: BaseCheckpointSaver,
    on_test_result: Callable[[str, str, bool, str | None], None] | None = None,
) -> tuple[int, int, list[str]]:
    """Run all reconstruction tests. Returns (passed, failed, failure_names)."""
    passed = 0
    failed = 0
    failures: list[str] = []
    for test_fn in ALL_DELTA_CHANNEL_RECONSTRUCTION_TESTS:
        try:
            await test_fn(saver)
            passed += 1
            if on_test_result:
                on_test_result(
                    "delta_channel_reconstruction", test_fn.__name__, True, None
                )
        except Exception:
            failed += 1
            msg = f"{test_fn.__name__}: {traceback.format_exc()}"
            failures.append(msg)
            if on_test_result:
                on_test_result(
                    "delta_channel_reconstruction",
                    test_fn.__name__,
                    False,
                    traceback.format_exc(),
                )
    return passed, failed, failures
