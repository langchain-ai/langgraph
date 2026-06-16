"""Exercise concurrent `threads.stream()` against the integration API.

Two distinct threads.stream() contexts run in parallel against the same
client. Each context is independent (different thread_id minted by the
SDK, separate controller, separate auto-responder). Invariants:

1. Both runs reach the canonical terminal state independently
   (`items == ['streamed','tool','asked','sub']`).
2. Their thread_ids differ (no thread-id collision when minting client-side).
3. Neither raises during iteration.

This catches regressions where the two streams might share controller
state or where minted ids could collide under concurrent ``__aenter__``.
"""

from __future__ import annotations

import asyncio
import contextlib
import threading
from typing import Any

from _common import (
    ASSISTANT_ID,
    auto_respond_async,
    auto_respond_sync,
    check_api_reachable,
    header,
    make_async_client,
    make_sync_client,
)

_EXPECTED_TERMINAL_ITEMS = ["streamed", "tool", "asked", "sub"]


async def _drive_one_async(threads: Any, label: str) -> dict[str, Any]:
    async with threads.stream(assistant_id=ASSISTANT_ID) as thread:
        await thread.run.start(input={"messages": [], "value": "init", "items": []})
        responder = auto_respond_async(thread)
        # Just drain values until terminal; we only care about the final state.
        async for _ in thread.values:
            pass
        await responder
        final = await thread.output
        print(f"  [{label}] thread_id={thread.thread_id} items={final.get('items')!r}")
        return {"thread_id": thread.thread_id, "items": final.get("items")}


async def run_async() -> None:
    header("async concurrent threads.stream (x2)")
    threads, raw = make_async_client()
    try:
        results = await asyncio.gather(
            _drive_one_async(threads, "A"),
            _drive_one_async(threads, "B"),
        )
        a, b = results
        assert a["items"] == _EXPECTED_TERMINAL_ITEMS, (
            f"stream A failed to reach terminal: {a!r}"
        )
        assert b["items"] == _EXPECTED_TERMINAL_ITEMS, (
            f"stream B failed to reach terminal: {b!r}"
        )
        assert a["thread_id"] != b["thread_id"], (
            f"concurrent streams collided on thread_id {a['thread_id']!r}"
        )
    finally:
        await raw.aclose()


def _drive_one_sync(
    threads: Any, label: str, results: dict[str, dict[str, Any]]
) -> None:
    with threads.stream(assistant_id=ASSISTANT_ID) as thread:
        thread.run.start(input={"messages": [], "value": "init", "items": []})
        responder = auto_respond_sync(thread)
        for _ in thread.values:
            pass
        responder.join(timeout=10)
        final = thread.output
        print(f"  [{label}] thread_id={thread.thread_id} items={final.get('items')!r}")
        results[label] = {"thread_id": thread.thread_id, "items": final.get("items")}


def run_sync() -> None:
    header("sync concurrent threads.stream (x2)")
    threads, raw = make_sync_client()
    try:
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
        assert a is not None and a["items"] == _EXPECTED_TERMINAL_ITEMS, (
            f"stream A failed to reach terminal: {a!r}"
        )
        assert b is not None and b["items"] == _EXPECTED_TERMINAL_ITEMS, (
            f"stream B failed to reach terminal: {b!r}"
        )
        assert a["thread_id"] != b["thread_id"], (
            f"concurrent streams collided on thread_id {a['thread_id']!r}"
        )
    finally:
        with contextlib.suppress(Exception):
            raw.close()


def main() -> None:
    check_api_reachable()
    asyncio.run(run_async())
    run_sync()


if __name__ == "__main__":
    main()
