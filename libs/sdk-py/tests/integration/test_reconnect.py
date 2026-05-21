"""Mid-iteration SSE close + terminal-state recovery via REST."""

from __future__ import annotations

import asyncio
import contextlib
import functools
import time
from typing import Any

import pytest

from .conftest import ASSISTANT_ID, EXPECTED_TERMINAL_ITEMS

pytestmark = pytest.mark.integration

_INTERRUPT_WAIT_SECONDS = 5.0


def _instrument_dedup_async(controller: Any) -> dict[str, int]:
    """Wrap `_dedup_iter` so duplicate event_ids are counted (no asserts here)."""
    counter = {"drops": 0, "yields": 0}
    original = controller._dedup_iter.__func__  # type: ignore[attr-defined]

    @functools.wraps(original)
    async def _counted(self, source):  # type: ignore[no-untyped-def]
        async for event in source:
            event_id = event.get("event_id")
            if event_id is not None:
                if event_id in self._seen_event_ids:
                    counter["drops"] += 1
                    continue
                self._seen_event_ids.add(event_id)
            counter["yields"] += 1
            yield event

    # ty doesn't see through `@functools.wraps` to the descriptor protocol; this
    # is the canonical method-binding pattern.
    controller._dedup_iter = _counted.__get__(controller, type(controller))  # ty: ignore[unresolved-attribute]
    return counter


def _instrument_dedup_sync(controller: Any) -> dict[str, int]:
    counter = {"drops": 0, "yields": 0}
    original = controller._dedup_iter.__func__  # type: ignore[attr-defined]

    @functools.wraps(original)
    def _counted(self, source):  # type: ignore[no-untyped-def]
        for event in source:
            event_id = event.get("event_id")
            if event_id is not None:
                if event_id in self._seen_event_ids:
                    counter["drops"] += 1
                    continue
                self._seen_event_ids.add(event_id)
            counter["yields"] += 1
            yield event

    # ty doesn't see through `@functools.wraps` to the descriptor protocol; this
    # is the canonical method-binding pattern.
    controller._dedup_iter = _counted.__get__(controller, type(controller))  # ty: ignore[unresolved-attribute]
    return counter


async def test_close_mid_iteration_async(async_threads) -> None:
    """A client-initiated SSE close mid-iteration must not corrupt durable state."""
    threads, _ = async_threads
    async with threads.stream(assistant_id=ASSISTANT_ID) as thread:
        counter = _instrument_dedup_async(thread)
        await thread.run.start(input={"messages": [], "value": "init", "items": []})

        snapshots: list[dict] = []
        dropped = False
        iteration_error: BaseException | None = None
        try:
            async for snap in thread.values:
                snapshots.append(snap)
                if not dropped and thread._shared_stream is not None:
                    await thread._shared_stream.close()
                    dropped = True
        except BaseException as err:
            iteration_error = err

        # The values iterator exits via the None sentinel pushed by `close()`,
        # which can land before the lifecycle watcher observes `input.requested`.
        # Poll briefly so the interrupt has a chance to arrive on its own SSE
        # before we ask for terminal state.
        deadline = asyncio.get_running_loop().time() + _INTERRUPT_WAIT_SECONDS
        while not thread.interrupted and asyncio.get_running_loop().time() < deadline:
            await asyncio.sleep(0.1)

        if thread.interrupted:
            with contextlib.suppress(Exception):
                await thread.run.respond("yes")

        final = await thread.output.with_timeout(_INTERRUPT_WAIT_SECONDS)
        assert dropped, "expected to drop the shared stream during iteration"
        assert snapshots, "expected at least one snapshot before the drop"
        assert iteration_error is None, (
            f"values iterator raised on stream close: {iteration_error!r}"
        )
        assert final.get("items") == EXPECTED_TERMINAL_ITEMS
        # Graceful close should not produce duplicate event_ids since the SDK
        # only reconnects (via `since=<cursor>`) on a non-cancelled `shared.done`.
        assert counter["drops"] == 0, (
            f"unexpected dedup activity (drops={counter['drops']}); "
            "no rotation occurred so no overlap was expected"
        )


def test_close_mid_iteration_sync(sync_threads) -> None:
    threads, _ = sync_threads
    with threads.stream(assistant_id=ASSISTANT_ID) as thread:
        controller = thread._controller
        counter = _instrument_dedup_sync(controller)
        thread.run.start(input={"messages": [], "value": "init", "items": []})

        snapshots: list[dict] = []
        dropped = False
        iteration_error: BaseException | None = None
        try:
            for snap in thread.values:
                snapshots.append(snap)
                if (
                    not dropped
                    and controller is not None
                    and controller._shared_stream is not None
                ):
                    controller._shared_stream.close()
                    dropped = True
        except BaseException as err:
            iteration_error = err

        deadline = time.monotonic() + _INTERRUPT_WAIT_SECONDS
        while not thread.interrupted and time.monotonic() < deadline:
            time.sleep(0.1)

        if thread.interrupted:
            with contextlib.suppress(Exception):
                thread.run.respond("yes")

        final = thread.output
        assert dropped, "expected to drop the shared stream during iteration"
        assert snapshots, "expected at least one snapshot before the drop"
        assert iteration_error is None, (
            f"values iterator raised on stream close: {iteration_error!r}"
        )
        assert final.get("items") == EXPECTED_TERMINAL_ITEMS
        assert counter["drops"] == 0, (
            f"unexpected dedup activity (drops={counter['drops']})"
        )
