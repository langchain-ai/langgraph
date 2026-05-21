"""`thread.values` against the integration API."""

from __future__ import annotations

import pytest

from .conftest import ASSISTANT_ID, EXPECTED_TERMINAL_ITEMS

pytestmark = pytest.mark.integration


async def test_values_async(async_threads) -> None:
    threads, _ = async_threads
    async with threads.stream(assistant_id=ASSISTANT_ID) as thread:
        await thread.run.start(input={"messages": [], "value": "init", "items": []})

        # `_signal_paused` pushes None to the values subscription on the
        # rising edge of `interrupted`, so this loop exits at the interrupt.
        snapshots: list[dict] = []
        async for snap in thread.values:
            snapshots.append(snap)

        assert thread.interrupted, f"expected interrupt; got {len(snapshots)} snapshots"
        await thread.run.respond("yes")

        final = await thread.output
        assert final.get("items") == EXPECTED_TERMINAL_ITEMS
        assert snapshots, "expected pre-interrupt snapshots"


def test_values_sync(sync_threads) -> None:
    threads, _ = sync_threads
    with threads.stream(assistant_id=ASSISTANT_ID) as thread:
        thread.run.start(input={"messages": [], "value": "init", "items": []})

        snapshots: list[dict] = []
        for snap in thread.values:
            snapshots.append(snap)

        assert thread.interrupted, f"expected interrupt; got {len(snapshots)} snapshots"
        thread.run.respond("yes")

        final = thread.output
        assert final.get("items") == EXPECTED_TERMINAL_ITEMS
        assert snapshots, "expected pre-interrupt snapshots"
