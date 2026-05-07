"""DELTA_CHANNEL_RECONSTRUCTION capability tests — end-to-end round-trip.

Exercises: aput + aput_writes + aget_delta_channel_history + from_checkpoint +
replay_writes. This catches the most common silent-corruption mode: failing to
round-trip `_DeltaSnapshot` blobs through serialization.
"""

from __future__ import annotations

import traceback
from collections.abc import Callable
from uuid import uuid4

from langgraph.checkpoint.base import BaseCheckpointSaver

from langgraph.checkpoint.conformance.spec._delta_fixtures import build_delta_chain


def _list_reducer(state: list, writes: list) -> list:
    """Simple append reducer for testing."""
    return state + writes


async def test_reconstruction_basic(
    saver: BaseCheckpointSaver,
) -> None:
    """Reconstruct delta channel value from history matches expected."""
    from langgraph.channels.delta import DeltaChannel

    tid = str(uuid4())
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
    from langgraph._internal._typing import MISSING

    if seed is None:
        seed = MISSING

    ch = DeltaChannel(_list_reducer, list)
    replay_ch = ch.from_checkpoint(seed)
    replay_ch.replay_writes(history["writes"])
    reconstructed = replay_ch.get()

    # Expected: snapshot at step 0 = [0], then writes at steps 1,2,3,4
    expected = [0] + [1] + [2] + [3] + [4]
    assert reconstructed == expected, (
        f"Reconstructed {reconstructed} != expected {expected}"
    )


async def test_reconstruction_mid_chain_snapshot(
    saver: BaseCheckpointSaver,
) -> None:
    """Reconstruction works when snapshot is mid-chain."""
    from langgraph.channels.delta import DeltaChannel

    tid = str(uuid4())
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
    from langgraph._internal._typing import MISSING

    if seed is None:
        seed = MISSING

    ch = DeltaChannel(_list_reducer, list)
    replay_ch = ch.from_checkpoint(seed)
    replay_ch.replay_writes(history["writes"])
    reconstructed = replay_ch.get()

    # Snapshot at step 3 = [3], writes at steps 4,5
    expected = [3] + [4] + [5]
    assert reconstructed == expected, (
        f"Reconstructed {reconstructed} != expected {expected}"
    )


async def test_reconstruction_no_snapshot(
    saver: BaseCheckpointSaver,
) -> None:
    """Reconstruction from root (no snapshot) gives all writes accumulated."""
    from langgraph.channels.delta import DeltaChannel

    tid = str(uuid4())
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

    from langgraph._internal._typing import MISSING

    seed = history.get("seed", MISSING)

    ch = DeltaChannel(_list_reducer, list)
    replay_ch = ch.from_checkpoint(seed)
    replay_ch.replay_writes(history["writes"])
    reconstructed = replay_ch.get()

    expected = [0] + [1] + [2] + [3]
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
