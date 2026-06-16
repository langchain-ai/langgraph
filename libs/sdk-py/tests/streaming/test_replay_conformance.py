from __future__ import annotations

import httpx

from langgraph_sdk.stream.transport.http import ProtocolSseTransport
from streaming._fake_server import FakeServer
from streaming.assert_transport_replays import assert_transport_replays


class _Harness:
    def __init__(self, fake: FakeServer, transport: ProtocolSseTransport) -> None:
        self._fake = fake
        self.transport = transport

    def script_buffered(self, events):
        self._fake.script(events)


async def test_fake_server_replays_buffered_events():
    fake = FakeServer()
    asgi = httpx.ASGITransport(app=fake.app)
    async with httpx.AsyncClient(transport=asgi, base_url="http://test") as client:
        transport = ProtocolSseTransport(client=client, thread_id="t-1")
        await assert_transport_replays(_Harness(fake, transport))
