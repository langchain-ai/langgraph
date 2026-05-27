"""Exercise stream-handle close + recovery against the integration API.

The SDK's "reconnect on transport drop" code path (controller
`_reconnect_shared_stream`) only fires when `shared.done` resolves to a
non-cancelled error — i.e. genuine network/server failures, not graceful
client-initiated closes. Reliably faking such an error against a real
server is brittle, so this script asserts the next-strongest invariant:
**a client-initiated stream close mid-iteration must not corrupt durable
state**.

Concretely:

1. Start the run; let the auto-responder unblock the interrupt.
2. Drop the shared SSE handle after the first snapshot.
3. The values projection iterator may end early (the close drains the
   sub queue with `None`), but `thread.output` must still resolve to the
   canonical terminal state via the REST fallback path.
4. No exception escapes the iteration.

We also instrument `_dedup_iter` to count any duplicate event_ids and
print the counter for visibility. A future regression that
double-delivers events through the controller would surface here.
"""

from __future__ import annotations

import asyncio
import contextlib
import functools
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


def _instrument_dedup_async(controller: Any) -> dict[str, int]:
    """Wrap `_dedup_iter` so duplicate event_ids are counted."""
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

    controller._dedup_iter = _counted.__get__(controller, type(controller))
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

    controller._dedup_iter = _counted.__get__(controller, type(controller))
    return counter


async def run_async() -> None:
    header("async stream-close mid-iteration (terminal state via REST)")
    threads, raw = make_async_client()
    try:
        async with threads.stream(assistant_id=ASSISTANT_ID) as thread:
            counter = _instrument_dedup_async(thread)
            await thread.run.start(input={"messages": [], "value": "init", "items": []})

            responder = auto_respond_async(thread)

            snapshots: list[dict] = []
            dropped = False
            iteration_error: BaseException | None = None
            try:
                async for snap in thread.values:
                    snapshots.append(snap)
                    if not dropped and thread._shared_stream is not None:
                        print(f"  dropping shared stream (cursor={thread._cursor})...")
                        await thread._shared_stream.close()
                        dropped = True
            except BaseException as err:
                iteration_error = err

            await responder

            final = await thread.output
            print(f"  snapshots seen before drop: {len(snapshots)}")
            print(f"  final items={final.get('items')!r}")
            print(f"  dedup drops={counter['drops']} yields={counter['yields']}")
            print(f"  iteration_error={iteration_error!r}")

            assert dropped, "expected to drop the shared stream during iteration"
            assert snapshots, "expected at least one snapshot before the drop"
            assert iteration_error is None, (
                f"values iterator raised on stream close: {iteration_error!r}"
            )
            assert final.get("items") == _EXPECTED_TERMINAL_ITEMS, (
                f"terminal state not reached via REST after drop: "
                f"items={final.get('items')!r}"
            )
            assert counter["drops"] == 0, (
                f"unexpected dedup activity (drops={counter['drops']}); "
                "no rotation occurred so no overlap was expected"
            )
    finally:
        await raw.aclose()


def run_sync() -> None:
    header("sync stream-close mid-iteration (terminal state via REST)")
    threads, raw = make_sync_client()
    try:
        with threads.stream(assistant_id=ASSISTANT_ID) as thread:
            controller = thread._controller
            counter = _instrument_dedup_sync(controller)
            thread.run.start(input={"messages": [], "value": "init", "items": []})

            responder = auto_respond_sync(thread)

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
                        print(
                            f"  dropping shared stream (cursor={controller._cursor})..."
                        )
                        controller._shared_stream.close()
                        dropped = True
            except BaseException as err:
                iteration_error = err

            responder.join(timeout=10)

            final = thread.output
            print(f"  snapshots seen before drop: {len(snapshots)}")
            print(f"  final items={final.get('items')!r}")
            print(f"  dedup drops={counter['drops']} yields={counter['yields']}")
            print(f"  iteration_error={iteration_error!r}")

            assert dropped, "expected to drop the shared stream during iteration"
            assert snapshots, "expected at least one snapshot before the drop"
            assert iteration_error is None, (
                f"values iterator raised on stream close: {iteration_error!r}"
            )
            assert final.get("items") == _EXPECTED_TERMINAL_ITEMS, (
                f"terminal state not reached via REST after drop: "
                f"items={final.get('items')!r}"
            )
            assert counter["drops"] == 0, (
                f"unexpected dedup activity (drops={counter['drops']}); "
                "no rotation occurred so no overlap was expected"
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
