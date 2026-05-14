"""Tests for `thread.tool_calls` - typed async tool-call projection."""

from __future__ import annotations

import httpx
import pytest

from langgraph_sdk._async.http import HttpClient
from langgraph_sdk._async.threads import ThreadsClient
from streaming._events import (
    lifecycle_completed_event,
    lifecycle_errored_event,
    lifecycle_started_event,
    tool_error_event,
    tool_finished_event,
    tool_output_delta_event,
    tool_started_event,
)
from streaming._fake_server import FakeServer


async def test_tool_calls_subscribes_to_tools_channel():
    fake = FakeServer()
    fake.script([lifecycle_completed_event(seq=1)])
    asgi = httpx.ASGITransport(app=fake.app)
    async with httpx.AsyncClient(transport=asgi, base_url="http://test") as raw:
        threads = ThreadsClient(HttpClient(raw))
        async with threads.stream(thread_id="t-1", assistant_id="agent") as thread:
            await thread.run.start(input={})
            _ = [call async for call in thread.tool_calls]

    assert any(
        "tools" in body.get("channels", []) for body in fake.stream_request_bodies
    )


async def test_tool_calls_yields_handle_deltas_and_output():
    fake = FakeServer()
    fake.script(
        [
            lifecycle_started_event(seq=0),
            tool_started_event(
                seq=1,
                tool_call_id="call-1",
                tool_name="search",
                input={"query": "sf weather"},
            ),
            tool_output_delta_event(seq=2, tool_call_id="call-1", delta="part "),
            tool_output_delta_event(seq=3, tool_call_id="call-1", delta="two"),
            tool_finished_event(
                seq=4,
                tool_call_id="call-1",
                output={"temperature": 68},
            ),
            lifecycle_completed_event(seq=5),
        ]
    )
    asgi = httpx.ASGITransport(app=fake.app)
    async with httpx.AsyncClient(transport=asgi, base_url="http://test") as raw:
        threads = ThreadsClient(HttpClient(raw))
        async with threads.stream(thread_id="t-1", assistant_id="agent") as thread:
            await thread.run.start(input={})
            calls = [call async for call in thread.tool_calls]

    assert len(calls) == 1
    call = calls[0]
    assert call.tool_call_id == "call-1"
    assert call.name == "search"
    assert call.input == {"query": "sf weather"}
    assert call.namespace == []
    assert call.done is True
    assert [delta async for delta in call.deltas] == ["part ", "two"]
    assert await call.output == {"temperature": 68}


async def test_tool_calls_multiple_concurrent_calls_route_by_id():
    fake = FakeServer()
    fake.script(
        [
            lifecycle_started_event(seq=0),
            tool_started_event(seq=1, tool_call_id="call-a", tool_name="alpha"),
            tool_started_event(seq=2, tool_call_id="call-b", tool_name="beta"),
            tool_output_delta_event(seq=3, tool_call_id="call-b", delta="b1"),
            tool_output_delta_event(seq=4, tool_call_id="call-a", delta="a1"),
            tool_finished_event(seq=5, tool_call_id="call-a", output="A"),
            tool_finished_event(seq=6, tool_call_id="call-b", output="B"),
            lifecycle_completed_event(seq=7),
        ]
    )
    asgi = httpx.ASGITransport(app=fake.app)
    async with httpx.AsyncClient(transport=asgi, base_url="http://test") as raw:
        threads = ThreadsClient(HttpClient(raw))
        async with threads.stream(thread_id="t-1", assistant_id="agent") as thread:
            await thread.run.start(input={})
            calls = [call async for call in thread.tool_calls]

    by_id = {call.tool_call_id: call for call in calls}
    assert set(by_id) == {"call-a", "call-b"}
    assert [delta async for delta in by_id["call-a"].deltas] == ["a1"]
    assert [delta async for delta in by_id["call-b"].deltas] == ["b1"]
    assert await by_id["call-a"].output == "A"
    assert await by_id["call-b"].output == "B"


async def test_tool_calls_ignores_nested_namespace_for_root_projection():
    fake = FakeServer()
    fake.script(
        [
            lifecycle_started_event(seq=0),
            tool_started_event(seq=1, namespace=["child:1"], tool_call_id="nested"),
            tool_finished_event(seq=2, namespace=["child:1"], tool_call_id="nested"),
            lifecycle_completed_event(seq=3),
        ]
    )
    asgi = httpx.ASGITransport(app=fake.app)
    async with httpx.AsyncClient(transport=asgi, base_url="http://test") as raw:
        threads = ThreadsClient(HttpClient(raw))
        async with threads.stream(thread_id="t-1", assistant_id="agent") as thread:
            await thread.run.start(input={})
            calls = [call async for call in thread.tool_calls]

    assert calls == []


async def test_tool_calls_error_event_fails_output_and_deltas():
    fake = FakeServer()
    fake.script(
        [
            lifecycle_started_event(seq=0),
            tool_started_event(seq=1, tool_call_id="call-1"),
            tool_output_delta_event(seq=2, tool_call_id="call-1", delta="before"),
            tool_error_event(seq=3, tool_call_id="call-1", message="boom"),
            lifecycle_completed_event(seq=4),
        ]
    )
    asgi = httpx.ASGITransport(app=fake.app)
    async with httpx.AsyncClient(transport=asgi, base_url="http://test") as raw:
        threads = ThreadsClient(HttpClient(raw))
        async with threads.stream(thread_id="t-1", assistant_id="agent") as thread:
            await thread.run.start(input={})
            calls = [call async for call in thread.tool_calls]

    assert len(calls) == 1
    assert [delta async for delta in calls[0].deltas] == ["before"]
    with pytest.raises(RuntimeError, match="boom"):
        await calls[0].output


async def test_tool_calls_run_error_fails_active_handle():
    fake = FakeServer()
    fake.script(
        [
            lifecycle_started_event(seq=0),
            tool_started_event(seq=1, tool_call_id="call-1"),
            lifecycle_errored_event(seq=2, error="run failed"),
        ]
    )
    asgi = httpx.ASGITransport(app=fake.app)
    async with httpx.AsyncClient(transport=asgi, base_url="http://test") as raw:
        threads = ThreadsClient(HttpClient(raw))
        async with threads.stream(thread_id="t-1", assistant_id="agent") as thread:
            await thread.run.start(input={})
            calls = [call async for call in thread.tool_calls]

    assert len(calls) == 1
    with pytest.raises(RuntimeError, match="Run errored: run failed"):
        await calls[0].output


async def test_tool_calls_stream_end_fails_active_handle():
    fake = FakeServer()
    fake.script(
        [
            lifecycle_started_event(seq=0),
            tool_started_event(seq=1, tool_call_id="call-1"),
            lifecycle_completed_event(seq=2),
        ]
    )
    asgi = httpx.ASGITransport(app=fake.app)
    async with httpx.AsyncClient(transport=asgi, base_url="http://test") as raw:
        threads = ThreadsClient(HttpClient(raw))
        async with threads.stream(thread_id="t-1", assistant_id="agent") as thread:
            await thread.run.start(input={})
            calls = [call async for call in thread.tool_calls]

    assert len(calls) == 1
    with pytest.raises(RuntimeError, match="closed before terminal tool event"):
        await calls[0].output
