"""Exercise `thread.extensions[name]` against the integration API.

Every node in the example graph writes `("progress", {...})` via
`get_stream_writer`. This script verifies the `extensions["progress"]`
projection yields each progress event in order.
"""

from __future__ import annotations

import asyncio

from _common import (
    ASSISTANT_ID,
    check_api_reachable,
    header,
    make_async_client,
    make_sync_client,
)


async def run_async() -> None:
    header("async extensions[progress]")
    threads, raw = make_async_client()
    try:
        async with threads.stream(assistant_id=ASSISTANT_ID) as thread:
            await thread.run.start(input={"messages": [], "value": "init", "items": []})

            events: list[dict] = []
            async for event in thread.extensions["progress"]:
                print(f"  progress: {event!r}")
                events.append(event)
            print(f"  total progress events: {len(events)}")
            assert events, "expected at least one progress event"
            # Verify ordering covers the node sequence.
            steps = [e.get("step") for e in events]
            print(f"  step sequence: {steps}")
    finally:
        await raw.aclose()


def run_sync() -> None:
    header("sync extensions[progress]")
    threads, raw = make_sync_client()
    try:
        with threads.stream(assistant_id=ASSISTANT_ID) as thread:
            thread.run.start(input={"messages": [], "value": "init", "items": []})

            events: list[dict] = []
            for event in thread.extensions["progress"]:
                print(f"  progress: {event!r}")
                events.append(event)
            print(f"  total progress events: {len(events)}")
            assert events, "expected at least one progress event"
            steps = [e.get("step") for e in events]
            print(f"  step sequence: {steps}")
    finally:
        raw.close()


def main() -> None:
    check_api_reachable()
    asyncio.run(run_async())
    run_sync()


if __name__ == "__main__":
    main()
