"""`thread.extensions[name]` channel against the integration API."""

from __future__ import annotations

import pytest

from .conftest import ASSISTANT_ID

pytestmark = pytest.mark.integration

_EXPECTED_PRE_INTERRUPT_STEPS = [
    "stream_message",
    "stream_message",
    "call_tool",
    "call_tool",
    "ask_human",
]


async def test_extensions_async(async_threads) -> None:
    threads, _ = async_threads
    async with threads.stream(assistant_id=ASSISTANT_ID) as thread:
        await thread.run.start(input={"messages": [], "value": "init", "items": []})

        events: list[dict] = []
        async for event in thread.extensions["progress"]:
            events.append(event)

        # Iterator exits at the `ask_human` interrupt via `_signal_paused`,
        # so we capture exactly the pre-interrupt progress sequence.
        steps = [e.get("step") for e in events]
        assert steps == _EXPECTED_PRE_INTERRUPT_STEPS, (
            f"unexpected step sequence: {steps}"
        )


def test_extensions_sync(sync_threads) -> None:
    threads, _ = sync_threads
    with threads.stream(assistant_id=ASSISTANT_ID) as thread:
        thread.run.start(input={"messages": [], "value": "init", "items": []})

        events: list[dict] = []
        for event in thread.extensions["progress"]:
            events.append(event)

        steps = [e.get("step") for e in events]
        assert steps == _EXPECTED_PRE_INTERRUPT_STEPS, (
            f"unexpected step sequence: {steps}"
        )
