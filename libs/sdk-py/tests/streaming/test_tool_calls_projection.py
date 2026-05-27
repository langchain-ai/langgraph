"""Tests for `thread.tool_calls` - typed async tool-call projection."""

from __future__ import annotations

import time

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


async def test_tool_calls_explicit_aclose_does_not_block_1s():
    """Explicitly closing the tool_calls iterator must return in <500ms.

    The old finally block did `await asyncio.wait_for(asyncio.shield(run_done),
    timeout=1.0)` unconditionally. When the caller explicitly calls aclose() on
    the generator before any lifecycle terminal event arrives, this caused a
    mandatory 1-second stall per iterator close.
    """
    fake = FakeServer()
    # Script has a started lifecycle and one tool, but NO terminal lifecycle.
    # If the shield-wait is present, aclose() will block for 1s.
    fake.script(
        [
            lifecycle_started_event(seq=0),
            tool_started_event(seq=1, tool_call_id="call-1"),
        ]
    )
    asgi = httpx.ASGITransport(app=fake.app)
    async with httpx.AsyncClient(transport=asgi, base_url="http://test") as raw:
        threads = ThreadsClient(HttpClient(raw))
        async with threads.stream(thread_id="t-1", assistant_id="agent") as thread:
            await thread.run.start(input={})
            # _tool_calls_iter() is an AsyncGenerator; cast so the type checker
            # knows aclose() is available without a bare AsyncIterator protocol.
            from collections.abc import AsyncGenerator

            gen: AsyncGenerator = thread.tool_calls._tool_calls_iter()
            _call = await gen.__anext__()  # receive the one tool-started handle
            start = time.monotonic()
            await gen.aclose()  # explicitly close — must not stall 1s
            elapsed = time.monotonic() - start
    assert elapsed < 0.5, f"tool_calls aclose() took {elapsed:.3f}s (expected <0.5s)"


def test_tool_call_handle_deltas_queue_is_bounded():
    """ToolCallHandle._deltas must be constructed with a bounded asyncio.Queue.

    Unbounded queues allow producers to enqueue indefinitely, causing memory
    growth when consumers are slow.
    """
    import asyncio

    # We need a running loop to create the Future inside ToolCallHandle.__init__.
    async def _make() -> None:
        from langgraph_sdk._async.stream import ToolCallHandle

        handle_default = ToolCallHandle(tool_call_id="tc1", name="foo")
        assert handle_default._deltas.maxsize > 0, (
            "default maxsize must be positive (bounded)"
        )

        handle_custom = ToolCallHandle(tool_call_id="tc2", name="bar", max_queue_size=8)
        assert handle_custom._deltas.maxsize == 8

    asyncio.run(_make())


def test_tool_call_handle_deltas_single_consumer_guard():
    """Accessing `handle.deltas` a second time must raise immediately.

    `_deltas` is a single-consumer queue; fanning out to multiple consumers
    would cause each consumer to miss events already consumed by the other.
    The property must raise before returning the iterator so the caller
    sees the error even without iterating.
    """
    import asyncio

    async def _run() -> None:
        from langgraph_sdk._async.stream import ToolCallHandle

        handle = ToolCallHandle(tool_call_id="tc1", name="foo")

        # First access: fine — returns the iterator.
        _iter_1 = handle.deltas

        # Second access: must raise immediately (before any iteration).
        with pytest.raises(RuntimeError, match="single consumer"):
            _ = handle.deltas

    asyncio.run(_run())
