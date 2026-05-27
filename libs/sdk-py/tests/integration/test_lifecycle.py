"""`thread.interrupted` / `thread.interrupts` / `run.respond` lifecycle."""

from __future__ import annotations

import pytest

from .conftest import ASSISTANT_ID, EXPECTED_TERMINAL_ITEMS

pytestmark = pytest.mark.integration


async def test_lifecycle_async(async_threads) -> None:
    threads, _ = async_threads
    async with threads.stream(assistant_id=ASSISTANT_ID) as thread:
        await thread.run.start(input={"messages": [], "value": "init", "items": []})

        async for _snap in thread.values:
            if thread.interrupted:
                break

        assert thread.interrupted, "expected an interrupt"
        assert thread.interrupts, "expected interrupts list to be populated"

        await thread.run.respond("yes")
        final = await thread.output
        assert "asked" in final.get("items", [])
        assert final.get("items") == EXPECTED_TERMINAL_ITEMS


def test_lifecycle_sync(sync_threads) -> None:
    threads, _ = sync_threads
    with threads.stream(assistant_id=ASSISTANT_ID) as thread:
        thread.run.start(input={"messages": [], "value": "init", "items": []})

        for _snap in thread.values:
            if thread.interrupted:
                break

        assert thread.interrupted, "expected an interrupt"
        assert thread.interrupts, "expected interrupts list to be populated"

        thread.run.respond("yes")
        final = thread.output
        assert "asked" in final.get("items", [])
        assert final.get("items") == EXPECTED_TERMINAL_ITEMS
