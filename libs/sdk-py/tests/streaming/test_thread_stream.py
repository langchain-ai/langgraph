from __future__ import annotations

import asyncio
import contextlib
import re
import uuid
from typing import Any

import httpx
import pytest

from langgraph_sdk._async.http import HttpClient
from langgraph_sdk._async.stream import AsyncThreadStream
from langgraph_sdk._async.threads import ThreadsClient
from langgraph_sdk.stream.transport import (
    ProtocolSseTransport,
    ProtocolWebSocketTransport,
)
from streaming._events import (
    lifecycle_completed_event,
    lifecycle_event,
    lifecycle_started_event,
    values_event,
)
from streaming._fake_server import FakeServer


async def test_thread_agent_get_tree_fetches_assistant_graph():
    fake = FakeServer()
    fake.set_graph(
        {
            "nodes": [{"id": "agent", "type": "runnable", "data": {"name": "agent"}}],
            "edges": [{"source": "agent", "target": "__end__"}],
        }
    )
    transport = httpx.ASGITransport(app=fake.app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as raw:
        threads = ThreadsClient(HttpClient(raw))
        async with threads.stream(
            thread_id="t-1",
            assistant_id="agent",
            headers={"X-Custom-Header": "my-value"},
        ) as thread:
            graph = await thread.agent.get_tree(xray=True)

    assert graph["nodes"][0]["id"] == "agent"
    assert graph["edges"] == [{"source": "agent", "target": "__end__"}]
    assert fake.graph_request_params == [{"xray": "true"}]
    assert fake.graph_request_headers[0].get("x-custom-header") == "my-value"


async def test_thread_agent_get_tree_raises_after_close():
    async with httpx.AsyncClient(base_url="http://test") as raw:
        threads = ThreadsClient(HttpClient(raw))
        stream = threads.stream(thread_id="t-1", assistant_id="agent")
        await stream.close()
        with pytest.raises(RuntimeError, match="closed"):
            await stream.agent.get_tree()


async def test_extensions_projection_empty_name_raises():
    async with httpx.AsyncClient(base_url="http://test") as raw:
        threads = ThreadsClient(HttpClient(raw))
        stream = threads.stream(thread_id="t-1", assistant_id="agent")
        with pytest.raises(ValueError, match="non-empty"):
            stream.extensions[""]


async def test_extensions_projection_closed_stream_yields_nothing():
    async with httpx.AsyncClient(base_url="http://test") as raw:
        threads = ThreadsClient(HttpClient(raw))
        # Enter and immediately exit so _controller is set but _closed is True.
        async with threads.stream(thread_id="t-1", assistant_id="agent") as stream:
            pass
        payloads = [p async for p in stream.extensions["progress"]]
    assert payloads == []


async def test_thread_stream_stores_thread_id_and_assistant_id():
    async with httpx.AsyncClient(base_url="http://test") as client:
        stream = AsyncThreadStream(
            http=HttpClient(client),
            thread_id="t-1",
            assistant_id="agent",
        )
        assert stream.thread_id == "t-1"
        assert stream.assistant_id == "agent"


async def test_aenter_returns_self():
    async with httpx.AsyncClient(base_url="http://test") as client:
        stream = AsyncThreadStream(
            http=HttpClient(client), thread_id="t-1", assistant_id="agent"
        )
        async with stream as entered:
            assert entered is stream


async def test_aexit_marks_closed():
    async with httpx.AsyncClient(base_url="http://test") as client:
        stream = AsyncThreadStream(
            http=HttpClient(client), thread_id="t-1", assistant_id="agent"
        )
        async with stream:
            assert stream._closed is False
        assert stream._closed is True


async def test_close_is_idempotent():
    async with httpx.AsyncClient(base_url="http://test") as client:
        stream = AsyncThreadStream(
            http=HttpClient(client), thread_id="t-1", assistant_id="agent"
        )
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


async def test_threads_stream_headers_forwarded_to_commands():
    """Headers passed to `threads.stream()` are forwarded to /commands requests."""
    fake = FakeServer()
    transport = httpx.ASGITransport(app=fake.app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as raw:
        threads = ThreadsClient(HttpClient(raw))
        async with threads.stream(
            thread_id="t-1",
            assistant_id="agent",
            headers={"X-Custom-Header": "my-value"},
        ) as thread:
            await thread.run.start(input={})
    assert fake.command_request_headers, "no command requests captured"
    assert fake.command_request_headers[0].get("x-custom-header") == "my-value"


async def test_threads_stream_headers_forwarded_to_stream_events():
    """Headers passed to `threads.stream()` are forwarded to /stream/events requests."""
    fake = FakeServer()
    fake.script([lifecycle_event(seq=0)])
    transport = httpx.ASGITransport(app=fake.app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as raw:
        threads = ThreadsClient(HttpClient(raw))
        async with threads.stream(
            thread_id="t-1",
            assistant_id="agent",
            headers={"X-Custom-Header": "my-value"},
        ) as thread:
            await thread.run.start(input={})
            _ = [e async for e in thread.subscribe(["lifecycle"])]
    assert fake.stream_request_headers_list, "no stream/events requests captured"
    assert fake.stream_request_headers_list[0].get("x-custom-header") == "my-value"


async def test_no_headers_by_default():
    """When `headers` is omitted, `_headers` is an empty dict and no custom
    headers appear in command or stream requests.
    """
    fake = FakeServer()
    fake.script([lifecycle_event(seq=0)])
    transport = httpx.ASGITransport(app=fake.app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as raw:
        threads = ThreadsClient(HttpClient(raw))
        async with threads.stream(thread_id="t-1", assistant_id="agent") as thread:
            assert thread._headers == {}
            await thread.run.start(input={})
            _ = [e async for e in thread.subscribe(["lifecycle"])]
    # No custom header keys beyond the protocol-required / transport-required ones.
    protocol_keys = {
        "content-type",
        "accept",
        "cache-control",
        "host",
        "user-agent",
        "accept-encoding",
        "connection",
        "transfer-encoding",
        "content-length",
    }
    extra_command = {
        k for k in fake.command_request_headers[0] if k.lower() not in protocol_keys
    }
    extra_stream = {
        k for k in fake.stream_request_headers_list[0] if k.lower() not in protocol_keys
    }
    assert extra_command == set(), f"unexpected command headers: {extra_command}"
    assert extra_stream == set(), f"unexpected stream headers: {extra_stream}"


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


async def test_aenter_selects_websocket_transport():
    async with httpx.AsyncClient(base_url="http://test") as raw:
        from langgraph_sdk._async.http import HttpClient
        from langgraph_sdk._async.threads import ThreadsClient

        threads = ThreadsClient(HttpClient(raw))
        stream = threads.stream(
            thread_id="t-1", assistant_id="agent", transport="websocket"
        )
        async with stream:
            assert isinstance(stream._transport, ProtocolWebSocketTransport)


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
        assert isinstance(inner_transport, ProtocolSseTransport)
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
        stream = AsyncThreadStream(
            http=HttpClient(raw), thread_id="t-1", assistant_id="agent"
        )
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
        stream = AsyncThreadStream(
            http=HttpClient(raw), thread_id="t-1", assistant_id="agent"
        )
        with pytest.raises(RuntimeError, match="async with"):
            _ = stream.events


async def test_aexit_preserves_original_exception_if_close_raises():
    """If the body of `async with` raises, AND close() also raises, the
    body's exception must propagate. close()'s error is suppressed (chained
    as context on close_err, but does not replace the original)."""
    async with httpx.AsyncClient(base_url="http://test") as raw:
        thread = AsyncThreadStream(
            http=HttpClient(raw), thread_id="t-1", assistant_id="agent"
        )

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

    Validates core surface end-to-end:
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
        stream = AsyncThreadStream(
            http=HttpClient(raw), thread_id="t-1", assistant_id="agent"
        )
        async with stream:
            pass
        # After exit, the stream is closed; re-entering must raise rather than
        # silently constructing a new transport that would leak on the next exit.
        with pytest.raises(RuntimeError, match="closed and cannot be re-entered"):
            async with stream:
                pass


async def test_register_subscription_assigns_monotonic_ids():
    async with httpx.AsyncClient(base_url="http://test") as raw:
        stream = AsyncThreadStream(
            http=HttpClient(raw), thread_id="t-1", assistant_id="agent"
        )
        async with stream:
            sub_a = stream._register_subscription({"channels": ["values"]})
            sub_b = stream._register_subscription({"channels": ["messages"]})
            assert sub_a.id == 1
            assert sub_b.id == 2
            assert stream._subscriptions[sub_a.id] is sub_a
            assert stream._subscriptions[sub_b.id] is sub_b


async def test_unregister_subscription_removes_from_registry():
    async with httpx.AsyncClient(base_url="http://test") as raw:
        stream = AsyncThreadStream(
            http=HttpClient(raw), thread_id="t-1", assistant_id="agent"
        )
        async with stream:
            sub = stream._register_subscription({"channels": ["values"]})
            stream._unregister_subscription(sub.id)
            assert sub.id not in stream._subscriptions


async def test_await_run_start_gate_honors_timeout():
    """Gate must raise asyncio.TimeoutError if run.start never completes
    within the configured timeout."""
    import asyncio

    async with httpx.AsyncClient(base_url="http://test") as raw:
        threads = ThreadsClient(HttpClient(raw))
        async with threads.stream(thread_id="t-1", assistant_id="agent") as thread:
            # Install a never-resolving gate to simulate an in-flight
            # run.start that will not complete within the timeout window.
            loop = asyncio.get_running_loop()
            thread._run_start_ready = loop.create_future()
            with pytest.raises(asyncio.TimeoutError):
                await thread._await_run_start_gate(timeout=0.1)
            # Gate must still be pending after the timeout (no side effects).
            assert thread._run_start_ready is not None
            assert not thread._run_start_ready.done()


async def test_await_run_start_gate_returns_when_gate_resolves_in_time():
    """With a generous timeout and a gate that resolves promptly, the
    gate returns without raising."""
    import asyncio

    async with httpx.AsyncClient(base_url="http://test") as raw:
        threads = ThreadsClient(HttpClient(raw))
        async with threads.stream(thread_id="t-1", assistant_id="agent") as thread:
            loop = asyncio.get_running_loop()
            gate: asyncio.Future[None] = loop.create_future()
            thread._run_start_ready = gate
            loop.call_later(0.01, lambda: gate.set_result(None))
            await thread._await_run_start_gate(timeout=1.0)


async def test_run_start_timeout_constructor_kwarg_forwarded_to_gate():
    """`run_start_timeout` constructor kwarg is stored and consulted by
    `_reconcile_stream` via `_await_run_start_gate`."""
    import asyncio

    async with httpx.AsyncClient(base_url="http://test") as raw:
        stream = AsyncThreadStream(
            http=HttpClient(raw),
            thread_id="t-1",
            assistant_id="agent",
            run_start_timeout=0.1,
        )
        async with stream as thread:
            loop = asyncio.get_running_loop()
            # Install a never-resolving gate.
            thread._run_start_ready = loop.create_future()
            with pytest.raises(asyncio.TimeoutError):
                # Reconcile must surface the timeout from the gate.
                await thread._reconcile_stream({"channels": ["lifecycle"]})


async def test_subscribe_waits_for_run_start_to_commit():
    """Subscribing before run.start commits must not race the server.

    With the gate: subscribers wait for run.start to return before opening
    their SSE. Without it, a fast subscribe would 404 against a thread the
    server hasn't created yet.
    """
    import asyncio

    fake = FakeServer()
    fake.script([])
    transport = httpx.ASGITransport(app=fake.app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as raw:
        threads = ThreadsClient(HttpClient(raw))
        async with threads.stream(thread_id="t-1", assistant_id="agent") as thread:
            # Kick run.start without awaiting — concurrently subscribe.
            run_task = asyncio.create_task(thread.run.start(input={}))
            sub_iter = thread.subscribe(["lifecycle"])
            # Drain one event or hit EOF. The iterator's first __anext__
            # awaits _reconcile_stream which awaits the gate.
            async for _ in sub_iter:
                break
            # If the gate works, run.start completed before the subscription
            # opened its SSE (and thus before iteration finished).
            assert run_task.done()


async def test_run_respond_dispatches_input_respond_command():
    fake = FakeServer()
    asgi = httpx.ASGITransport(app=fake.app)
    async with httpx.AsyncClient(transport=asgi, base_url="http://test") as raw:
        threads = ThreadsClient(HttpClient(raw))
        async with threads.stream(thread_id="t-1", assistant_id="agent") as thread:
            await thread.run.start(input={})
            # Simulate one outstanding interrupt.
            thread.interrupts.append(
                {"interrupt_id": "i-1", "value": None, "namespace": []}
            )
            thread.interrupted = True
            await thread.run.respond("yes")
    command = fake.received_commands[-1]
    assert command["method"] == "input.respond"
    assert command["params"]["interrupt_id"] == "i-1"
    assert command["params"]["response"] == "yes"
    assert command["params"]["namespace"] == []


async def test_run_respond_with_explicit_interrupt_id():
    fake = FakeServer()
    asgi = httpx.ASGITransport(app=fake.app)
    async with httpx.AsyncClient(transport=asgi, base_url="http://test") as raw:
        threads = ThreadsClient(HttpClient(raw))
        async with threads.stream(thread_id="t-1", assistant_id="agent") as thread:
            await thread.run.start(input={})
            thread.interrupts.extend(
                [
                    {"interrupt_id": "a", "value": None, "namespace": []},
                    {"interrupt_id": "b", "value": None, "namespace": []},
                ]
            )
            thread.interrupted = True
            await thread.run.respond("pick", interrupt_id="b")
    assert fake.received_commands[-1]["params"]["interrupt_id"] == "b"
    assert fake.received_commands[-1]["params"]["namespace"] == []


async def test_run_respond_raises_when_no_outstanding_interrupts():
    async with httpx.AsyncClient(base_url="http://test") as raw:
        threads = ThreadsClient(HttpClient(raw))
        async with threads.stream(thread_id="t-1", assistant_id="agent") as thread:
            with pytest.raises(RuntimeError, match="no outstanding interrupt"):
                await thread.run.respond("yes")


async def test_run_respond_raises_when_ambiguous_interrupt_id():
    async with httpx.AsyncClient(base_url="http://test") as raw:
        threads = ThreadsClient(HttpClient(raw))
        async with threads.stream(thread_id="t-1", assistant_id="agent") as thread:
            thread.interrupts.extend(
                [
                    {"interrupt_id": "a", "value": None, "namespace": []},
                    {"interrupt_id": "b", "value": None, "namespace": []},
                ]
            )
            thread.interrupted = True
            with pytest.raises(RuntimeError, match=r"ambiguous|interrupt_id"):
                await thread.run.respond("yes")


async def test_run_respond_snapshots_interrupts_under_lock():
    """`respond()` must take a snapshot of `interrupts` under the
    `_interrupts_lock`, so a concurrent terminal-event clear cannot
    invalidate the in-flight dispatch.

    Verifies: if `_interrupts_lock` is held when `respond()` is called,
    `respond()` blocks until the lock is released — proving it serializes
    with the terminal-clear path that takes the same lock.
    """
    import asyncio

    fake = FakeServer()
    asgi = httpx.ASGITransport(app=fake.app)
    async with httpx.AsyncClient(transport=asgi, base_url="http://test") as raw:
        threads = ThreadsClient(HttpClient(raw))
        async with threads.stream(thread_id="t-1", assistant_id="agent") as thread:
            await thread.run.start(input={})
            thread.interrupts.append(
                {"interrupt_id": "i-1", "value": None, "namespace": []}
            )
            thread.interrupted = True
            # Take the interrupts lock externally to block `respond()`.
            assert hasattr(thread, "_interrupts_lock"), (
                "AsyncThreadStream must expose _interrupts_lock"
            )
            await thread._interrupts_lock.acquire()
            try:
                # `respond()` must NOT complete while we hold the lock.
                task = asyncio.create_task(thread.run.respond("yes"))
                # Give the task a chance to start and reach the lock.
                await asyncio.sleep(0.05)
                assert not task.done(), (
                    "respond() should be blocked waiting for _interrupts_lock"
                )
            finally:
                thread._interrupts_lock.release()
            # Now `respond()` should complete.
            await asyncio.wait_for(task, timeout=1.0)
    command = fake.received_commands[-1]
    assert command["method"] == "input.respond"
    assert command["params"]["interrupt_id"] == "i-1"


async def test_terminal_lifecycle_clear_acquires_interrupts_lock():
    """Terminal lifecycle event clears `interrupts` under the same lock
    that `respond()` uses, preventing TOCTOU between snapshot and
    dispatch."""
    import asyncio

    fake = FakeServer()
    # No scripted events; we exercise `_apply_lifecycle_event` directly.
    fake.script([])
    asgi = httpx.ASGITransport(app=fake.app)
    async with httpx.AsyncClient(transport=asgi, base_url="http://test") as raw:
        threads = ThreadsClient(HttpClient(raw))
        async with threads.stream(thread_id="t-1", assistant_id="agent") as thread:
            thread.interrupts.append(
                {"interrupt_id": "i-1", "value": None, "namespace": []}
            )
            thread.interrupted = True
            # Hold the lock; a completion event must block on it before
            # clearing interrupts.
            await thread._interrupts_lock.acquire()
            try:
                from typing import cast

                from langchain_protocol import Event

                terminal_event = cast(
                    Event,
                    {
                        "type": "event",
                        "method": "lifecycle",
                        "params": {
                            "namespace": [],
                            "data": {"event": "completed"},
                        },
                        "seq": 99,
                        "event_id": "evt-99",
                    },
                )
                clear_task = asyncio.create_task(
                    thread._apply_lifecycle_event(terminal_event)
                )
                await asyncio.sleep(0.05)
                # Interrupts must still be present — clear is blocked.
                assert thread.interrupted is True
                assert len(thread.interrupts) == 1
                assert not clear_task.done()
            finally:
                thread._interrupts_lock.release()
            await asyncio.wait_for(clear_task, timeout=1.0)
            assert thread.interrupted is False
            assert thread.interrupts == []


async def test_run_respond_raises_when_explicit_interrupt_id_not_outstanding():
    async with httpx.AsyncClient(base_url="http://test") as raw:
        threads = ThreadsClient(HttpClient(raw))
        async with threads.stream(thread_id="t-1", assistant_id="agent") as thread:
            thread.interrupts.append(
                {"interrupt_id": "a", "value": None, "namespace": []}
            )
            thread.interrupted = True
            with pytest.raises(RuntimeError, match="does not match"):
                await thread.run.respond("yes", interrupt_id="nonexistent")


async def test_output_cancellation_does_not_trigger_new_fetch():
    """When the in-flight fetch task for thread.output is cancelled, a
    subsequent call must NOT spawn a fresh task and issue a new REST GET;
    `_get_task` should return the same (cancelled) task so awaiters share
    the CancelledError outcome.
    """
    fake = FakeServer()
    # No lifecycle terminal event — the fetch task will park on _run_done.
    fake.script([])
    fake.set_state({"messages": ["hello"]})
    asgi = httpx.ASGITransport(app=fake.app)
    async with httpx.AsyncClient(transport=asgi, base_url="http://test") as raw:
        threads = ThreadsClient(HttpClient(raw))
        async with threads.stream(thread_id="t-1", assistant_id="agent") as thread:
            await thread.run.start(input={})
            output_awaitable = thread.output

            # Materialize the underlying task and let it park on _run_done.
            shared_task = output_awaitable._get_task()
            for _ in range(20):
                await asyncio.sleep(0)
                if not shared_task.done():
                    break

            # Simulate an in-flight fetch being cancelled.
            shared_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await shared_task
            assert shared_task.done()
            assert shared_task.cancelled()

            # A subsequent _get_task() must return the SAME cancelled task —
            # no respawn, no fresh REST fetch.
            second_task = output_awaitable._get_task()
            assert second_task is shared_task, (
                "expected cancelled task to be reused; got a fresh task"
            )
            # No state GET should have been issued (lifecycle never completed).
            assert fake.state_request_count == 0


async def test_output_with_timeout_raises_timeout_error():
    """thread.output.with_timeout(s) raises TimeoutError when the lifecycle
    never resolves within the budget."""
    fake = FakeServer()
    # Hold the stream open with a long inter-event delay so the lifecycle
    # watcher parks on the iterator (mid-sleep before a non-terminal event)
    # and `_run_done` never resolves before the timeout fires. Without the
    # delay, a clean EOF would resolve `_run_done` with an errored
    # `_RunTerminal` (per PR 7821) and `with_timeout` would raise that error
    # instead of `asyncio.TimeoutError`.
    fake.script([lifecycle_started_event(seq=0)], delay=10.0)
    fake.set_state({"never": "reached"})
    asgi = httpx.ASGITransport(app=fake.app)
    async with httpx.AsyncClient(transport=asgi, base_url="http://test") as raw:
        threads = ThreadsClient(HttpClient(raw))
        async with threads.stream(thread_id="t-1", assistant_id="agent") as thread:
            await thread.run.start(input={})
            with pytest.raises(asyncio.TimeoutError):
                await thread.output.with_timeout(0.1)


async def test_output_with_timeout_returns_value_when_lifecycle_completes_in_time():
    """thread.output.with_timeout(s) returns the values dict when the lifecycle
    resolves within the budget."""
    fake = FakeServer()
    fake.script([lifecycle_completed_event(seq=0)])
    fake.set_state({"messages": ["hello"]})
    asgi = httpx.ASGITransport(app=fake.app)
    async with httpx.AsyncClient(transport=asgi, base_url="http://test") as raw:
        threads = ThreadsClient(HttpClient(raw))
        async with threads.stream(thread_id="t-1", assistant_id="agent") as thread:
            await thread.run.start(input={})
            result = await thread.output.with_timeout(2.0)
    assert result == {"messages": ["hello"]}


async def test_output_with_timeout_returns_new_awaitable_not_self():
    """with_timeout() returns a fresh awaitable, leaving the original untouched."""
    async with httpx.AsyncClient(base_url="http://test") as raw:
        threads = ThreadsClient(HttpClient(raw))
        async with threads.stream(thread_id="t-1", assistant_id="agent") as thread:
            bounded = thread.output.with_timeout(0.5)
            assert bounded is not thread.output
            assert bounded._timeout == 0.5
            assert thread.output._timeout is None


async def test_threads_stream_accepts_websocket_transport_option():
    async with httpx.AsyncClient(base_url="http://test") as raw:
        threads = ThreadsClient(HttpClient(raw))
        stream = threads.stream(
            thread_id="t-1",
            assistant_id="agent",
            transport="websocket",
        )
    assert stream._transport_kind == "websocket"


async def test_threads_stream_rejects_unknown_transport_option():
    async with httpx.AsyncClient(base_url="http://test") as raw:
        threads = ThreadsClient(HttpClient(raw))
        with pytest.raises(ValueError, match="transport"):
            threads.stream(
                thread_id="t-1",
                assistant_id="agent",
                transport="bogus",  # ty: ignore[invalid-argument-type]
            )


async def test_v3_streaming_async_surface_smoke():
    import asyncio

    from streaming._events import (
        custom_event,
        lifecycle_completed_event,
        message_finish_event,
        message_start_event,
        message_text_delta_event,
        message_text_finish_event,
        tool_finished_event,
        tool_started_event,
        values_event,
    )

    fake = FakeServer()
    fake.set_state({"final": True})
    fake.script(
        [
            values_event(seq=1, values={"step": 1}),
            message_start_event(seq=2, message_id="msg-1"),
            message_text_delta_event(seq=3, text="hi", message_id="msg-1"),
            message_text_finish_event(seq=4, text="hi", message_id="msg-1"),
            message_finish_event(seq=5, message_id="msg-1"),
            tool_started_event(seq=6, tool_call_id="call-1", tool_name="search"),
            tool_finished_event(seq=7, tool_call_id="call-1", output={"ok": True}),
            custom_event(seq=8, name="progress", step=1),
            lifecycle_completed_event(seq=9),
        ]
    )
    async with httpx.AsyncClient(
        transport=fake.transport, base_url="http://test"
    ) as raw:
        threads = ThreadsClient(HttpClient(raw))
        async with threads.stream(thread_id="t-1", assistant_id="agent") as thread:
            start = await thread.run.start(
                input={"messages": [{"role": "user", "content": "hi"}]}
            )

            # Start all consumers concurrently so their subscriptions are all
            # registered before the first SSE reconciliation. This means ONE
            # shared SSE opens with the union filter, dedup never rejects events.
            async def _get_first_values() -> Any:
                async for v in thread.values:
                    return v

            async def _get_message_streams():
                return [s async for s in thread.messages]

            async def _get_tool_calls():
                return [call async for call in thread.tool_calls]

            async def _get_progress():
                return [p async for p in thread.extensions["progress"]]

            first_values, message_streams, tool_calls, progress = await asyncio.gather(
                _get_first_values(),
                _get_message_streams(),
                _get_tool_calls(),
                _get_progress(),
            )
            # stream.text is already fully accumulated after gather completes.
            message_texts = [await s.text for s in message_streams]
            final = await thread.output

    assert start == {"run_id": "run-1"}
    assert first_values == fake.state["values"]
    assert message_texts == ["hi"]
    assert tool_calls[0].name == "search"
    assert progress == [{"name": "progress", "step": 1}]
    assert final == {"final": True}
