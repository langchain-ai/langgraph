from __future__ import annotations

import asyncio
from collections.abc import AsyncGenerator
from typing import Any, cast

import httpx

from langgraph_sdk._async.http import HttpClient
from langgraph_sdk._async.threads import ThreadsClient
from langgraph_sdk.stream.controller import StreamController
from streaming._events import lifecycle_event, values_event
from streaming._fake_server import FakeServer


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
            controller: StreamController = raw_controller

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
