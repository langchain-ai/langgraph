"""Tests for StreamController and related stream/controller.py types."""

from __future__ import annotations

import pytest

from langgraph_sdk.stream.controller import StreamController

# ---------------------------------------------------------------------------
# Task 3.1: bounded subscription queues
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_subscription_queue_bounded_by_max_queue_size():
    """`StreamController` must create per-subscription queues bounded by `max_queue_size`."""
    import httpx

    from langgraph_sdk.stream.transport.http import ProtocolSseTransport

    transport = ProtocolSseTransport(
        client=httpx.AsyncClient(base_url="http://test"),
        thread_id="t-1",
    )
    controller = StreamController(transport=transport, max_queue_size=4)
    sub = controller._register_subscription({"channels": ["values"]})
    assert sub.queue.maxsize == 4


@pytest.mark.asyncio
async def test_subscription_queue_default_max_queue_size_is_1024():
    """`StreamController` default `max_queue_size` is 1024."""
    import httpx

    from langgraph_sdk.stream.transport.http import ProtocolSseTransport

    transport = ProtocolSseTransport(
        client=httpx.AsyncClient(base_url="http://test"),
        thread_id="t-1",
    )
    controller = StreamController(transport=transport)
    sub = controller._register_subscription({"channels": ["values"]})
    assert sub.queue.maxsize == 1024
