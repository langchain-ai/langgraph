"""Concurrent `threads.stream()` against the integration API."""

from __future__ import annotations

import asyncio
import threading
import time
from typing import Any

import pytest

from .conftest import ASSISTANT_ID, EXPECTED_TERMINAL_ITEMS

pytestmark = pytest.mark.integration
_INTERRUPT_TIMEOUT = 10.0


async def _drive_one_async(threads: Any) -> dict[str, Any]:
    async with threads.stream(assistant_id=ASSISTANT_ID) as thread:
        await thread.run.start(input={"messages": [], "value": "init", "items": []})

        async for _ in thread.values:
            if thread.interrupted:
                break
        deadline = time.monotonic() + _INTERRUPT_TIMEOUT
        while not thread.interrupted:
            assert time.monotonic() < deadline, "expected an interrupt before output"
            await asyncio.sleep(0.01)
        await thread.run.respond("yes")

        final = await thread.output
        return {"thread_id": thread.thread_id, "items": final.get("items")}


async def test_concurrent_streams_async(async_threads) -> None:
    threads, _ = async_threads
    a, b = await asyncio.gather(_drive_one_async(threads), _drive_one_async(threads))
    assert a["items"] == EXPECTED_TERMINAL_ITEMS
    assert b["items"] == EXPECTED_TERMINAL_ITEMS
    assert a["thread_id"] != b["thread_id"], (
        f"concurrent streams collided on thread_id {a['thread_id']!r}"
    )


def _drive_one_sync(
    threads: Any, label: str, results: dict[str, dict[str, Any]]
) -> None:
    with threads.stream(assistant_id=ASSISTANT_ID) as thread:
        thread.run.start(input={"messages": [], "value": "init", "items": []})

        for _ in thread.values:
            if thread.interrupted:
                break
        deadline = time.monotonic() + _INTERRUPT_TIMEOUT
        while not thread.interrupted:
            assert time.monotonic() < deadline, "expected an interrupt before output"
            time.sleep(0.01)
        thread.run.respond("yes")

        final = thread.output
        results[label] = {"thread_id": thread.thread_id, "items": final.get("items")}


def test_concurrent_streams_sync(sync_threads) -> None:
    threads, _ = sync_threads
    results: dict[str, dict[str, Any]] = {}
    workers = [
        threading.Thread(
            target=_drive_one_sync,
            args=(threads, label, results),
            daemon=True,
            name=f"sync-stream-{label}",
        )
        for label in ("A", "B")
    ]
    for w in workers:
        w.start()
    for w in workers:
        w.join(timeout=60)
        assert not w.is_alive(), f"worker {w.name} did not finish within 60s"

    a = results.get("A")
    b = results.get("B")
    assert a is not None and a["items"] == EXPECTED_TERMINAL_ITEMS
    assert b is not None and b["items"] == EXPECTED_TERMINAL_ITEMS
    assert a["thread_id"] != b["thread_id"], (
        f"concurrent streams collided on thread_id {a['thread_id']!r}"
    )
