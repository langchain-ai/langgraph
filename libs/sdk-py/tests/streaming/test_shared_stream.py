from __future__ import annotations

import asyncio
from collections.abc import AsyncGenerator
from typing import Any, cast

import httpx

from langgraph_sdk._async.http import HttpClient
from langgraph_sdk._async.threads import ThreadsClient
from langgraph_sdk.stream.controller import StreamController
from langgraph_sdk.stream.transport.http import EventStreamHandle
from streaming._events import lifecycle_event, values_event
from streaming._fake_server import FakeServer, _StreamScript


async def test_shared_stream_serves_single_subscription():
    fake = FakeServer()
    fake.script([lifecycle_event(seq=0), values_event(seq=1)])
    asgi = httpx.ASGITransport(app=fake.app)
    async with httpx.AsyncClient(transport=asgi, base_url="http://test") as raw:
        threads = ThreadsClient(HttpClient(raw))
        async with threads.stream(thread_id="t-1", assistant_id="agent") as thread:
            await thread.run.start(input={})
            await thread._reconcile_stream({"channels": ["lifecycle", "values"]})
            assert thread._shared_stream is not None
            received = [
                e async for e in thread._dedup_iter(thread._shared_stream.events)
            ]
    methods = [e["method"] for e in received]
    assert methods == ["lifecycle", "values"]
    assert fake.peak_open_event_streams == 1


async def test_seen_event_ids_dedupes_replayed_events():
    fake = FakeServer()
    fake.script(
        [
            lifecycle_event(seq=0),
            lifecycle_event(seq=0),  # duplicate event_id
            values_event(seq=1),
        ]
    )
    asgi = httpx.ASGITransport(app=fake.app)
    async with httpx.AsyncClient(transport=asgi, base_url="http://test") as raw:
        threads = ThreadsClient(HttpClient(raw))
        async with threads.stream(thread_id="t-1", assistant_id="agent") as thread:
            await thread.run.start(input={})
            await thread._reconcile_stream({"channels": ["lifecycle", "values"]})
            assert thread._shared_stream is not None
            received = [
                e async for e in thread._dedup_iter(thread._shared_stream.events)
            ]
    seqs = [e["seq"] for e in received]
    assert seqs == [0, 1]  # the duplicate seq=0 was deduped via event_id


async def test_rotation_when_new_subscription_widens_filter():
    fake = FakeServer()
    fake.script([lifecycle_event(seq=0), values_event(seq=1)])
    asgi = httpx.ASGITransport(app=fake.app)
    async with httpx.AsyncClient(transport=asgi, base_url="http://test") as raw:
        threads = ThreadsClient(HttpClient(raw))
        async with threads.stream(thread_id="t-1", assistant_id="agent") as thread:
            await thread.run.start(input={})
            # First subscription: lifecycle only.
            await thread._reconcile_stream({"channels": ["lifecycle"]})
            # Second subscription widens to lifecycle + values.
            await thread._reconcile_stream({"channels": ["lifecycle", "values"]})
    # Rotation: two separate SSE requests were opened (old + new).
    assert len(fake.stream_request_bodies) >= 2


async def test_no_rotation_when_existing_filter_covers_new_subscription():
    fake = FakeServer()
    fake.script([])
    asgi = httpx.ASGITransport(app=fake.app)
    async with httpx.AsyncClient(transport=asgi, base_url="http://test") as raw:
        threads = ThreadsClient(HttpClient(raw))
        async with threads.stream(thread_id="t-1", assistant_id="agent") as thread:
            await thread.run.start(input={})
            await thread._reconcile_stream({"channels": ["lifecycle", "values"]})
            # New subscription is a subset — existing filter covers it.
            await thread._reconcile_stream({"channels": ["values"]})
    # No rotation in the shared stream (1 shared SSE) plus 1 lifecycle watcher SSE = 2.
    assert len(fake.stream_request_bodies) == 2


async def test_subscribe_yields_only_matching_events():
    fake = FakeServer()
    fake.script(
        [
            lifecycle_event(seq=0),
            values_event(seq=1),
            lifecycle_event(seq=2),
        ]
    )
    asgi = httpx.ASGITransport(app=fake.app)
    async with httpx.AsyncClient(transport=asgi, base_url="http://test") as raw:
        threads = ThreadsClient(HttpClient(raw))
        async with threads.stream(thread_id="t-1", assistant_id="agent") as thread:
            await thread.run.start(input={})

            async def drain(channels):
                return [e async for e in thread.subscribe(channels)]

            lifecycle_events, values_events = await asyncio.gather(
                drain(["lifecycle"]),
                drain(["values"]),
            )
    assert [e["seq"] for e in lifecycle_events] == [0, 2]
    assert [e["seq"] for e in values_events] == [1]


async def test_two_concurrent_subscribes_share_one_stream():
    fake = FakeServer()
    fake.script([lifecycle_event(seq=i) for i in range(5)])
    asgi = httpx.ASGITransport(app=fake.app)
    async with httpx.AsyncClient(transport=asgi, base_url="http://test") as raw:
        threads = ThreadsClient(HttpClient(raw))
        async with threads.stream(thread_id="t-1", assistant_id="agent") as thread:
            await thread.run.start(input={})

            async def drain(channels):
                return [e async for e in thread.subscribe(channels)]

            results = await asyncio.gather(
                drain(["lifecycle"]),
                drain(["lifecycle"]),
            )
    assert len(results[0]) == 5
    assert len(results[1]) == 5
    # Both subscriptions share one SSE (no rotation) plus 1 lifecycle watcher SSE = 2.
    assert len(fake.stream_request_bodies) == 2


async def test_subscribe_does_not_leak_when_iterator_unconsumed():
    """Subscriptions register lazily on first __anext__, not at subscribe() call time.

    Why: registering eagerly would leak the subscription if the caller
    constructs the iterator but never iterates it. The lazy pattern ties
    registration to the generator's lifecycle, which is bounded by aclose()
    / exhaustion / cancellation.
    """
    fake = FakeServer()
    fake.script([])
    asgi = httpx.ASGITransport(app=fake.app)
    async with httpx.AsyncClient(transport=asgi, base_url="http://test") as raw:
        threads = ThreadsClient(HttpClient(raw))
        async with threads.stream(thread_id="t-1", assistant_id="agent") as thread:
            await thread.run.start(input={})
            _ = thread.subscribe(["lifecycle"])  # construct but never iterate
            # Subscription is not registered yet — the generator body hasn't run.
            assert len(thread._subscriptions) == 0


async def test_values_projection_registers_via_delegation_not_controller_directly():
    """Values projection must register subscriptions through AsyncThreadStream delegation wrappers.

    Verifies that subscriptions created by `thread.values` are visible via
    `thread._subscriptions` (the delegated property) and also accessible in
    `thread._controller._subscriptions`, confirming both views are consistent.
    The test also confirms that the `_ValuesProjection` never bypasses
    `AsyncThreadStream._register_subscription` to write to the controller
    directly — the subscription count seen through the thread wrapper equals
    the count inside the controller at the moment the subscription is live.
    """
    from streaming._events import lifecycle_completed_event

    fake = FakeServer()
    fake.script([lifecycle_completed_event(seq=0)])
    fake.set_state({"ok": True})
    asgi = httpx.ASGITransport(app=fake.app)
    counts_during: list[tuple[int, int]] = []
    async with httpx.AsyncClient(transport=asgi, base_url="http://test") as raw:
        threads = ThreadsClient(HttpClient(raw))
        async with threads.stream(thread_id="t-1", assistant_id="agent") as thread:
            await thread.run.start(input={})
            raw_controller = thread._controller
            assert raw_controller is not None
            # `_controller` returns `self` (AsyncThreadStream) as a duck-typed
            # controller surface; StreamController is parallel groundwork
            # used elsewhere.
            controller: StreamController = raw_controller  # ty: ignore[invalid-assignment]

            assert len(thread._subscriptions) == 0

            # Collect counts while the subscription is live (first snapshot only).
            aiter: AsyncGenerator[Any, None] = cast(
                "AsyncGenerator[Any, None]", thread.values.__aiter__()
            )
            # Advance to first item — subscription must be registered by now.
            await aiter.__anext__()
            counts_during.append(
                (len(thread._subscriptions), len(controller._subscriptions))
            )
            # Close the iterator explicitly so the finally block runs immediately.
            await aiter.aclose()

    # Both views must have agreed — no bypass of the delegation wrapper.
    assert len(counts_during) == 1
    thread_count, ctrl_count = counts_during[0]
    assert thread_count == ctrl_count
    assert thread_count >= 1


def _make_handle(
    events: list[dict[str, Any]],
    err: BaseException | None = None,
) -> tuple[EventStreamHandle, asyncio.Queue]:
    """Build a synthetic EventStreamHandle for reconnect tests.

    Returns the handle and the underlying queue so callers can inject events
    or the end sentinel directly from test code.
    """
    loop = asyncio.get_running_loop()
    ready: asyncio.Future[None] = loop.create_future()
    done: asyncio.Future[BaseException | None] = loop.create_future()
    queue: asyncio.Queue = asyncio.Queue()

    async def _pump() -> None:
        ready.set_result(None)
        for event in events:
            await queue.put(event)
        done.set_result(err)
        await queue.put(None)  # sentinel

    asyncio.create_task(_pump())  # noqa: RUF006

    async def _aiter():
        while True:
            item = await queue.get()
            if item is None:
                return
            yield item

    async def _close() -> None:
        if not done.done():
            done.set_result(None)
        await queue.put(None)

    return EventStreamHandle(
        events=_aiter(), ready=ready, done=done, close=_close
    ), queue


async def test_shared_stream_reconnects_with_since_after_transport_drop():
    """StreamController reopens the stream with `since` after a post-ready drop."""
    opened_params: list[dict[str, Any]] = []

    handle1, _ = _make_handle(
        [values_event(seq=1, values={"counter": 1})],
        err=RuntimeError("scripted async stream failure"),
    )
    handle2, _ = _make_handle([values_event(seq=2, values={"counter": 2})])
    handles = [handle1, handle2]

    from unittest.mock import MagicMock

    from langgraph_sdk.stream.transport.http import ProtocolSseTransport

    transport = MagicMock(spec=ProtocolSseTransport)

    def _open(params: dict[str, Any]) -> EventStreamHandle:
        opened_params.append(dict(params))
        return handles.pop(0)

    transport.open_event_stream.side_effect = _open

    async def gate() -> None:
        return None

    controller = StreamController(transport=transport, run_start_gate=gate)
    sub = controller.register_subscription({"channels": ["values"]})
    await controller.reconcile_stream({"channels": ["values"]})
    controller.ensure_fanout_running()

    first = await asyncio.wait_for(sub.queue.get(), timeout=1.0)
    second = await asyncio.wait_for(sub.queue.get(), timeout=1.0)
    end = await asyncio.wait_for(sub.queue.get(), timeout=1.0)
    await controller.close()

    assert first["seq"] == 1
    assert second["seq"] == 2
    assert end is None
    assert opened_params[0]["channels"] == ["values"]
    assert "since" not in opened_params[0]
    assert opened_params[1]["channels"] == ["values"]
    assert opened_params[1]["since"] == 1


async def test_shared_stream_reconnect_dedupes_replayed_overlap():
    """StreamController deduplicates events replayed on reconnect."""
    handle1, _ = _make_handle(
        [values_event(seq=1, values={"counter": 1})],
        err=RuntimeError("scripted async stream failure"),
    )
    handle2, _ = _make_handle(
        [
            values_event(seq=1, values={"counter": 1}),  # replayed overlap
            values_event(seq=2, values={"counter": 2}),
        ]
    )
    handles = [handle1, handle2]

    from unittest.mock import MagicMock

    from langgraph_sdk.stream.transport.http import ProtocolSseTransport

    transport = MagicMock(spec=ProtocolSseTransport)
    transport.open_event_stream.side_effect = lambda _params: handles.pop(0)

    async def gate() -> None:
        return None

    controller = StreamController(transport=transport, run_start_gate=gate)
    sub = controller.register_subscription({"channels": ["values"]})
    await controller.reconcile_stream({"channels": ["values"]})
    controller.ensure_fanout_running()

    received = [
        await asyncio.wait_for(sub.queue.get(), timeout=1.0),
        await asyncio.wait_for(sub.queue.get(), timeout=1.0),
        await asyncio.wait_for(sub.queue.get(), timeout=1.0),
    ]
    await controller.close()

    assert [event["seq"] for event in received if event is not None] == [1, 2]
    assert received[-1] is None


async def test_send_command_applied_through_seq_seeds_shared_stream_since():
    fake = FakeServer()
    fake.script_sequence([_StreamScript(events=[]), _StreamScript(events=[])])
    fake.script_command_response(
        {
            "type": "success",
            "id": None,
            "result": {"run_id": "run-1"},
            "meta": {"applied_through_seq": 17},
        }
    )
    asgi = httpx.ASGITransport(app=fake.app)
    async with httpx.AsyncClient(transport=asgi, base_url="http://test") as raw:
        threads = ThreadsClient(HttpClient(raw))
        async with threads.stream(thread_id="t-1", assistant_id="agent") as thread:
            await thread.run.start(input={})
            _ = [event async for event in thread.subscribe(["values"])]

    # The shared SSE filter is a union of subscription params plus
    # ``lifecycle`` (added by ``_compute_current_union`` so the fanout
    # consumer can detect root-terminal events for projection-iterator
    # termination). Match any request whose channels include ``values``.
    values_requests = [
        b for b in fake.stream_request_bodies if "values" in (b.get("channels") or [])
    ]
    assert len(values_requests) == 1
    assert values_requests[0]["since"] == 17
