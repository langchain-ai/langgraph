"""Exercise the WebSocket transport against the integration API.

Equivalent to `test_values.py` but with `transport="websocket"`.
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
    header("async websocket transport")
    threads, raw = make_async_client()
    try:
        async with threads.stream(
            assistant_id=ASSISTANT_ID,
            transport="websocket",
        ) as thread:
            from langgraph_sdk.stream.transport import ProtocolWebSocketTransport

            assert isinstance(thread._transport, ProtocolWebSocketTransport), (
                f"expected ws transport, got {type(thread._transport).__name__}"
            )

            await thread.run.start(input={"messages": [], "value": "init", "items": []})

            # The graph interrupts at `ask_human`; without a background
            # responder the values iterator would pause indefinitely.
            responder = auto_respond_async(thread)

            snapshots: list[dict] = []
            async for snap in thread.values:
                snapshots.append(snap)
                print(f"  ws values snapshot items={snap.get('items')!r}")

            await responder

            final = await thread.output
            print(f"  final via ws: items={final.get('items')!r}")
            assert "sub" in final.get("items", []), (
                "expected subgraph to have run via ws transport"
            )
    finally:
        await raw.aclose()


def run_sync() -> None:
    header("sync websocket transport")
    threads, raw = make_sync_client()
    try:
        with threads.stream(
            assistant_id=ASSISTANT_ID,
            transport="websocket",
        ) as thread:
            thread.run.start(input={"messages": [], "value": "init", "items": []})

            responder = auto_respond_sync(thread)

            snapshots: list[dict] = []
            for snap in thread.values:
                snapshots.append(snap)
                print(f"  ws values snapshot items={snap.get('items')!r}")

            responder.join(timeout=5)

            final = thread.output
            print(f"  final via ws: items={final.get('items')!r}")
            assert "sub" in final.get("items", []), (
                "expected subgraph to have run via ws transport"
            )
    finally:
        raw.close()


def main() -> None:
    check_api_reachable()
    asyncio.run(run_async())
    run_sync()


if __name__ == "__main__":
    main()
