"""WebSocket transport against the integration API."""

from __future__ import annotations

import pytest

from .conftest import ASSISTANT_ID, EXPECTED_TERMINAL_ITEMS

pytestmark = pytest.mark.integration


async def test_websocket_async(async_threads) -> None:
    threads, _ = async_threads
    async with threads.stream(
        assistant_id=ASSISTANT_ID, transport="websocket"
    ) as thread:
        from langgraph_sdk.stream.transport import ProtocolWebSocketTransport

        assert isinstance(thread._transport, ProtocolWebSocketTransport)

        await thread.run.start(input={"messages": [], "value": "init", "items": []})

        snapshots: list[dict] = []
        async for snap in thread.values:
            snapshots.append(snap)

        assert thread.interrupted, "expected interrupt over ws"
        await thread.run.respond("yes")

        final = await thread.output
        assert final.get("items") == EXPECTED_TERMINAL_ITEMS


def test_websocket_sync(sync_threads) -> None:
    threads, _ = sync_threads
    with threads.stream(assistant_id=ASSISTANT_ID, transport="websocket") as thread:
        from langgraph_sdk.stream.transport import SyncProtocolWebSocketTransport

        assert isinstance(thread._transport, SyncProtocolWebSocketTransport)

        thread.run.start(input={"messages": [], "value": "init", "items": []})

        snapshots: list[dict] = []
        for snap in thread.values:
            snapshots.append(snap)

        assert thread.interrupted, "expected interrupt over ws"
        thread.run.respond("yes")

        final = thread.output
        assert final.get("items") == EXPECTED_TERMINAL_ITEMS
