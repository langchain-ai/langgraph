"""Exercise `thread.values` against the integration API.

The integration graph's `ask_human` node interrupts mid-run. The
projection iterators (`thread.values`, `.messages`, `.tool_calls`,
`.subgraphs`) do not terminate on interrupt — they're paused, waiting
for more events. To drain the full run end-to-end we use a background
auto-responder that watches `thread.interrupted` and calls
`thread.run.respond(...)` so the run continues to the terminal.

Run after `docker compose up -d` from `libs/sdk-py/integration/`:

    uv run python integration/scripts/test_values.py
"""

from __future__ import annotations

import asyncio

from _common import (
    ASSISTANT_ID,
    auto_respond_async,
    auto_respond_sync,
    check_api_reachable,
    header,
    make_async_client,
    make_sync_client,
)


async def run_async() -> None:
    header("async values")
    threads, raw = make_async_client()
    try:
        async with threads.stream(assistant_id=ASSISTANT_ID) as thread:
            await thread.run.start(input={"messages": [], "value": "init", "items": []})

            # Background task: respond to the interrupt so the iterator
            # eventually sees terminal-completion events.
            responder = auto_respond_async(thread)

            snapshots: list[dict] = []
            async for snap in thread.values:
                snapshots.append(snap)
                print(
                    f"  values snapshot: items={snap.get('items')!r} value={snap.get('value')!r}"
                )

            await responder

            final = await thread.output
            print(f"  final output items={final.get('items')!r}")
            assert "sub" in final.get("items", []), "expected subgraph to have run"
    finally:
        await raw.aclose()


def run_sync() -> None:
    header("sync values")
    threads, raw = make_sync_client()
    try:
        with threads.stream(assistant_id=ASSISTANT_ID) as thread:
            thread.run.start(input={"messages": [], "value": "init", "items": []})

            responder = auto_respond_sync(thread)

            snapshots: list[dict] = []
            for snap in thread.values:
                snapshots.append(snap)
                print(
                    f"  values snapshot: items={snap.get('items')!r} value={snap.get('value')!r}"
                )

            responder.join(timeout=5)

            final = thread.output
            print(f"  final output items={final.get('items')!r}")
            assert "sub" in final.get("items", []), "expected subgraph to have run"
    finally:
        raw.close()


def main() -> None:
    check_api_reachable()
    asyncio.run(run_async())
    run_sync()


if __name__ == "__main__":
    main()
