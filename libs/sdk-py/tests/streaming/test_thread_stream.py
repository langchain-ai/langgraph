from __future__ import annotations

import re
import uuid

import httpx
import pytest

from langgraph_sdk._async.http import HttpClient
from langgraph_sdk._async.stream import AsyncThreadStream
from langgraph_sdk._async.threads import ThreadsClient
from streaming._fake_server import FakeServer


async def test_thread_stream_stores_thread_id_and_assistant_id():
    async with httpx.AsyncClient(base_url="http://test") as client:
        stream = AsyncThreadStream(
            client=client,
            thread_id="t-1",
            assistant_id="agent",
        )
        assert stream.thread_id == "t-1"
        assert stream.assistant_id == "agent"


async def test_aenter_returns_self():
    async with httpx.AsyncClient(base_url="http://test") as client:
        stream = AsyncThreadStream(client=client, thread_id="t-1", assistant_id="agent")
        async with stream as entered:
            assert entered is stream


async def test_aexit_marks_closed():
    async with httpx.AsyncClient(base_url="http://test") as client:
        stream = AsyncThreadStream(client=client, thread_id="t-1", assistant_id="agent")
        async with stream:
            assert stream._closed is False
        assert stream._closed is True


async def test_close_is_idempotent():
    async with httpx.AsyncClient(base_url="http://test") as client:
        stream = AsyncThreadStream(client=client, thread_id="t-1", assistant_id="agent")
        await stream.close()
        await stream.close()  # must not raise
        assert stream._closed is True


async def test_threads_stream_returns_async_thread_stream_with_explicit_id():
    async with httpx.AsyncClient(base_url="http://test") as raw:
        threads = ThreadsClient(HttpClient(raw))
        stream = threads.stream(thread_id="my-thread", assistant_id="agent")
        assert stream.thread_id == "my-thread"
        assert stream.assistant_id == "agent"


async def test_threads_stream_mints_uuid4_when_thread_id_none():
    async with httpx.AsyncClient(base_url="http://test") as raw:
        threads = ThreadsClient(HttpClient(raw))
        stream = threads.stream(assistant_id="agent")
        # uuid4 format: 8-4-4-4-12 hex
        assert re.fullmatch(
            r"[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}",
            stream.thread_id,
        )
        # And it's actually parseable as a v4 UUID.
        assert uuid.UUID(stream.thread_id).version == 4


async def test_threads_stream_requires_assistant_id():
    async with httpx.AsyncClient(base_url="http://test") as raw:
        threads = ThreadsClient(HttpClient(raw))
        with pytest.raises(TypeError):
            threads.stream(thread_id="t-1")  # ty: ignore[missing-argument]


async def test_threads_stream_accepts_headers_kwarg():
    """Forward-compat: `headers` is accepted now even though it isn't plumbed yet."""
    from langgraph_sdk._async.http import HttpClient
    from langgraph_sdk._async.threads import ThreadsClient

    async with httpx.AsyncClient(base_url="http://test") as raw:
        threads = ThreadsClient(HttpClient(raw))
        stream = threads.stream(
            thread_id="t-1",
            assistant_id="agent",
            headers={"X-Foo": "bar"},
        )
        assert stream.thread_id == "t-1"


async def test_aenter_constructs_transport_with_thread_id():
    fake = FakeServer()
    transport = httpx.ASGITransport(app=fake.app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as raw:
        from langgraph_sdk._async.http import HttpClient
        from langgraph_sdk._async.threads import ThreadsClient

        threads = ThreadsClient(HttpClient(raw))
        stream = threads.stream(thread_id="t-1", assistant_id="agent")
        async with stream:
            assert stream._transport is not None
            assert stream._transport.thread_id == "t-1"


async def test_aexit_closes_transport():
    fake = FakeServer()
    transport = httpx.ASGITransport(app=fake.app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as raw:
        from langgraph_sdk._async.http import HttpClient
        from langgraph_sdk._async.threads import ThreadsClient

        threads = ThreadsClient(HttpClient(raw))
        stream = threads.stream(thread_id="t-1", assistant_id="agent")
        async with stream:
            inner_transport = stream._transport
        assert inner_transport is not None
        assert inner_transport._closed is True


async def test_run_start_sends_command_with_assistant_id():
    fake = FakeServer()
    transport = httpx.ASGITransport(app=fake.app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as raw:
        from langgraph_sdk._async.http import HttpClient
        from langgraph_sdk._async.threads import ThreadsClient

        threads = ThreadsClient(HttpClient(raw))
        async with threads.stream(thread_id="t-1", assistant_id="agent") as thread:
            result = await thread.run.start(input={"x": 1})
    assert result == {"run_id": "run-1"}
    command = fake.received_commands[0]
    assert command["method"] == "run.start"
    assert command["params"]["assistant_id"] == "agent"
    assert command["params"]["input"] == {"x": 1}
    assert command["id"] == 1


async def test_command_ids_are_monotonic():
    fake = FakeServer()
    transport = httpx.ASGITransport(app=fake.app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as raw:
        from langgraph_sdk._async.http import HttpClient
        from langgraph_sdk._async.threads import ThreadsClient

        threads = ThreadsClient(HttpClient(raw))
        async with threads.stream(thread_id="t-1", assistant_id="agent") as thread:
            await thread.run.start(input={"x": 1})
            await thread.run.start(input={"x": 2})
    assert [c["id"] for c in fake.received_commands] == [1, 2]


async def test_run_start_forwards_config_and_metadata():
    fake = FakeServer()
    transport = httpx.ASGITransport(app=fake.app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as raw:
        from langgraph_sdk._async.http import HttpClient
        from langgraph_sdk._async.threads import ThreadsClient

        threads = ThreadsClient(HttpClient(raw))
        async with threads.stream(thread_id="t-1", assistant_id="agent") as thread:
            await thread.run.start(
                input={"x": 1},
                config={"recursion_limit": 5},
                metadata={"trace": "abc"},
            )
    params = fake.received_commands[0]["params"]
    assert params["config"] == {"recursion_limit": 5}
    assert params["metadata"] == {"trace": "abc"}


async def test_run_start_raises_outside_context_manager():
    import pytest

    async with httpx.AsyncClient(base_url="http://test") as raw:
        stream = AsyncThreadStream(client=raw, thread_id="t-1", assistant_id="agent")
        with pytest.raises(RuntimeError, match="async with"):
            await stream.run.start(input={"x": 1})


async def test_run_start_raises_on_error_envelope():
    from starlette.applications import Starlette
    from starlette.responses import JSONResponse
    from starlette.routing import Route

    async def commands(_request):
        return JSONResponse(
            {
                "type": "error",
                "id": 1,
                "error": "invalid_argument",
                "message": "run.start requires an assistant_id.",
            }
        )

    app = Starlette(
        routes=[Route("/threads/{thread_id}/commands", commands, methods=["POST"])]
    )
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as raw:
        import pytest

        from langgraph_sdk._async.http import HttpClient
        from langgraph_sdk._async.threads import ThreadsClient

        threads = ThreadsClient(HttpClient(raw))
        async with threads.stream(thread_id="t-1", assistant_id="agent") as thread:
            with pytest.raises(RuntimeError, match="invalid_argument"):
                await thread.run.start(input={"x": 1})
