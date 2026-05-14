"""Tests for the lifecycle watcher: `interrupted` / `interrupts` state."""

from __future__ import annotations

import asyncio

import httpx

from langgraph_sdk._async.http import HttpClient
from langgraph_sdk._async.threads import ThreadsClient
from streaming._events import input_requested_event
from streaming._fake_server import FakeServer


async def test_interrupted_starts_false():
    async with httpx.AsyncClient(base_url="http://test") as raw:
        threads = ThreadsClient(HttpClient(raw))
        async with threads.stream(thread_id="t-1", assistant_id="agent") as thread:
            assert thread.interrupted is False
            assert thread.interrupts == []


async def test_interrupts_populated_from_input_requested_event():
    fake = FakeServer()
    fake.script([input_requested_event(seq=0)])
    asgi = httpx.ASGITransport(app=fake.app)
    async with httpx.AsyncClient(transport=asgi, base_url="http://test") as raw:
        threads = ThreadsClient(HttpClient(raw))
        async with threads.stream(thread_id="t-1", assistant_id="agent") as thread:
            await thread.run.start(input={})
            # Lifecycle watcher consumes asynchronously — poll briefly.
            for _ in range(20):
                if thread.interrupted:
                    break
                await asyncio.sleep(0.05)
    assert thread.interrupted is True
    assert len(thread.interrupts) == 1
    assert thread.interrupts[0]["interrupt_id"] == "i-1"
