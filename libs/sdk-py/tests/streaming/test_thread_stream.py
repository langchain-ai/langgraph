from __future__ import annotations

import re
import uuid

import httpx
import pytest

from langgraph_sdk._async.http import HttpClient
from langgraph_sdk._async.stream import AsyncThreadStream
from langgraph_sdk._async.threads import ThreadsClient
from streaming._events import lifecycle_event, values_event
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
    """`headers` is accepted as a kwarg even though it isn't forwarded yet."""
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


async def test_events_yields_raw_events_after_run_start():
    fake = FakeServer()
    fake.script(
        [
            lifecycle_event(seq=0),
            values_event(seq=1),
        ]
    )
    transport = httpx.ASGITransport(app=fake.app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as raw:
        from langgraph_sdk._async.http import HttpClient
        from langgraph_sdk._async.threads import ThreadsClient

        threads = ThreadsClient(HttpClient(raw))
        async with threads.stream(thread_id="t-1", assistant_id="agent") as thread:
            await thread.run.start(input={})
            received = [e async for e in thread.events]
    methods = [e["method"] for e in received]
    assert methods == ["lifecycle", "values"]


async def test_events_subscribes_to_all_channels():
    fake = FakeServer()
    fake.script([])
    transport = httpx.ASGITransport(app=fake.app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as raw:
        from langgraph_sdk._async.http import HttpClient
        from langgraph_sdk._async.threads import ThreadsClient

        threads = ThreadsClient(HttpClient(raw))
        async with threads.stream(thread_id="t-1", assistant_id="agent") as thread:
            await thread.run.start(input={})
            _ = [e async for e in thread.events]
    body = fake.stream_request_bodies[0]
    assert set(body["channels"]) == {
        "values",
        "updates",
        "messages",
        "tools",
        "lifecycle",
        "input",
        "checkpoints",
        "tasks",
        "custom",
    }


async def test_events_terminates_on_aexit():
    import asyncio

    import pytest

    fake = FakeServer()
    fake.script([lifecycle_event(seq=i) for i in range(5)])
    transport = httpx.ASGITransport(app=fake.app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as raw:
        from langgraph_sdk._async.http import HttpClient
        from langgraph_sdk._async.threads import ThreadsClient

        threads = ThreadsClient(HttpClient(raw))
        stream = threads.stream(thread_id="t-1", assistant_id="agent")
        async with stream as thread:
            await thread.run.start(input={})
            handle_events = thread.events
        # After __aexit__, further iteration must terminate cleanly.
        with pytest.raises(StopAsyncIteration):
            await asyncio.wait_for(handle_events.__anext__(), timeout=1.0)


async def test_events_raises_outside_context_manager():
    async with httpx.AsyncClient(base_url="http://test") as raw:
        stream = AsyncThreadStream(client=raw, thread_id="t-1", assistant_id="agent")
        with pytest.raises(RuntimeError, match="async with"):
            _ = stream.events


async def test_aexit_preserves_original_exception_if_close_raises():
    """If the body of `async with` raises, AND close() also raises, the
    body's exception must propagate. close()'s error is suppressed (chained
    as context on close_err, but does not replace the original)."""
    async with httpx.AsyncClient(base_url="http://test") as raw:
        thread = AsyncThreadStream(client=raw, thread_id="t-1", assistant_id="agent")

        async def failing_close():
            raise RuntimeError("close failed")

        thread.close = failing_close  # ty:ignore[invalid-assignment]

        with pytest.raises(ValueError, match="original"):
            async with thread:
                raise ValueError("original")


async def test_events_property_returns_fresh_iterator_each_access():
    """Two separate accesses of `thread.events` must return independent
    subscriptions — the second access should produce a fresh iterator,
    even if both are accessed before either is drained."""
    fake = FakeServer()
    fake.script([])
    transport = httpx.ASGITransport(app=fake.app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as raw:
        from langgraph_sdk._async.http import HttpClient
        from langgraph_sdk._async.threads import ThreadsClient

        threads = ThreadsClient(HttpClient(raw))
        async with threads.stream(thread_id="t-1", assistant_id="agent") as thread:
            first_iter = thread.events
            second_iter = thread.events
            # Each property access must return a distinct iterator object.
            assert first_iter is not second_iter


async def test_fresh_thread_happy_path_end_to_end():
    """User passes no thread_id; SDK mints one and uses it in all URLs.

    Validates the thread-stream surface end-to-end:
      - uuid4 minted at client.threads.stream()
      - run.start posted to /threads/<minted-id>/commands
      - events SSE opened at /threads/<minted-id>/stream/events
      - scripted events delivered to the user iterator
    """
    fake = FakeServer()
    fake.script([lifecycle_event(seq=0), values_event(seq=1)])

    posted_paths: list[str] = []

    class _PathSpyTransport(httpx.ASGITransport):
        async def handle_async_request(self, request):
            posted_paths.append(str(request.url.path))
            return await super().handle_async_request(request)

    spy = _PathSpyTransport(app=fake.app)
    async with httpx.AsyncClient(transport=spy, base_url="http://test") as raw:
        threads = ThreadsClient(HttpClient(raw))
        async with threads.stream(assistant_id="agent") as thread:
            assert uuid.UUID(thread.thread_id).version == 4
            result = await thread.run.start(input={"x": 1})
            assert result == {"run_id": "run-1"}
            received = [e async for e in thread.events]
    assert [e["method"] for e in received] == ["lifecycle", "values"]
    # Both POSTs must include the minted thread_id in the path.
    minted_id_paths = [p for p in posted_paths if thread.thread_id in p]
    assert any(p.endswith("/commands") for p in minted_id_paths)
    assert any(p.endswith("/stream/events") for p in minted_id_paths)


async def test_aenter_raises_after_close():
    async with httpx.AsyncClient(base_url="http://test") as raw:
        stream = AsyncThreadStream(client=raw, thread_id="t-1", assistant_id="agent")
        async with stream:
            pass
        # After exit, the stream is closed; re-entering must raise rather than
        # silently constructing a new transport that would leak on the next exit.
        with pytest.raises(RuntimeError, match="closed and cannot be re-entered"):
            async with stream:
                pass


async def test_register_subscription_assigns_monotonic_ids():
    async with httpx.AsyncClient(base_url="http://test") as raw:
        stream = AsyncThreadStream(client=raw, thread_id="t-1", assistant_id="agent")
        async with stream:
            sub_a = stream._register_subscription({"channels": ["values"]})
            sub_b = stream._register_subscription({"channels": ["messages"]})
            assert sub_a.id == 1
            assert sub_b.id == 2
            assert stream._subscriptions[sub_a.id] is sub_a
            assert stream._subscriptions[sub_b.id] is sub_b


async def test_unregister_subscription_removes_from_registry():
    async with httpx.AsyncClient(base_url="http://test") as raw:
        stream = AsyncThreadStream(client=raw, thread_id="t-1", assistant_id="agent")
        async with stream:
            sub = stream._register_subscription({"channels": ["values"]})
            stream._unregister_subscription(sub.id)
            assert sub.id not in stream._subscriptions
