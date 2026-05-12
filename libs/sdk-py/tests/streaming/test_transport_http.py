from __future__ import annotations

import asyncio

import httpx

from langgraph_sdk.stream.transport.http import EventStreamHandle, ProtocolSseTransport


async def test_event_stream_handle_constructs_with_open_state():
    ready: asyncio.Future[None] = asyncio.get_running_loop().create_future()
    closed = False

    async def aiter_events():
        if False:
            yield  # pragma: no cover

    async def closer():
        nonlocal closed
        closed = True

    handle = EventStreamHandle(events=aiter_events(), ready=ready, close=closer)
    assert handle.ready is ready
    await handle.close()
    assert closed is True


async def test_send_command_posts_json_and_returns_response():
    from streaming._fake_server import FakeServer

    fake = FakeServer()
    transport = httpx.ASGITransport(app=fake.app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        sse = ProtocolSseTransport(client=client, thread_id="t-1")
        result = await sse.send_command(
            {
                "command_id": 7,
                "method": "run.start",
                "params": {"input": {"x": 1}},
            }
        )
    assert result == {"command_id": 7, "result": {"run_id": "run-1"}}
    assert fake.received_commands[0]["command_id"] == 7


async def test_send_command_returns_none_on_202():
    from starlette.applications import Starlette
    from starlette.responses import Response
    from starlette.routing import Route

    received: list[dict] = []

    async def commands(_request):
        received.append({})
        return Response(status_code=202)

    app = Starlette(
        routes=[Route("/threads/{thread_id}/commands", commands, methods=["POST"])]
    )
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        sse = ProtocolSseTransport(client=client, thread_id="t-1")
        result = await sse.send_command(
            {"command_id": 1, "method": "noop", "params": {}}
        )
    assert result is None
    assert len(received) == 1
