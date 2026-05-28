"""`thread.messages` projection (outer iter + inner `.text` token deltas)."""

from __future__ import annotations

import pytest

from .conftest import ASSISTANT_ID

pytestmark = pytest.mark.integration


async def test_messages_async(async_threads) -> None:
    threads, _ = async_threads
    async with threads.stream(assistant_id=ASSISTANT_ID) as thread:
        await thread.run.start(input={"messages": [], "value": "init", "items": []})

        # Drain the outer iterator first; iterating each inner `stream.text`
        # while the outer is suspended deadlocks.
        streams = [s async for s in thread.messages]
        assert streams, "expected at least one streamed message"

        for stream in streams:
            text = "".join([t async for t in stream.text])
            assert text == "Hello, world!", f"unexpected message text: {text!r}"


def test_messages_sync(sync_threads) -> None:
    threads, _ = sync_threads
    with threads.stream(assistant_id=ASSISTANT_ID) as thread:
        thread.run.start(input={"messages": [], "value": "init", "items": []})

        streams = list(thread.messages)
        assert streams, "expected at least one streamed message"

        for stream in streams:
            text = "".join(list(stream.text))
            assert text == "Hello, world!", f"unexpected message text: {text!r}"
