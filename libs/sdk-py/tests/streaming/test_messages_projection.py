"""Tests for `thread.messages` - typed async message projection."""

from __future__ import annotations

import httpx
import pytest
from langchain_core.language_models.chat_model_stream import AsyncChatModelStream
from langchain_core.messages import AIMessage

from langgraph_sdk._async.http import HttpClient
from langgraph_sdk._async.threads import ThreadsClient
from streaming._events import (
    lifecycle_completed_event,
    lifecycle_started_event,
    message_error_event,
    message_finish_event,
    message_start_event,
    message_text_delta_event,
    message_text_finish_event,
)
from streaming._fake_server import FakeServer


async def test_messages_subscribes_to_messages_channel():
    fake = FakeServer()
    fake.script([lifecycle_completed_event(seq=1)])
    asgi = httpx.ASGITransport(app=fake.app)
    async with httpx.AsyncClient(transport=asgi, base_url="http://test") as raw:
        threads = ThreadsClient(HttpClient(raw))
        async with threads.stream(thread_id="t-1", assistant_id="agent") as thread:
            await thread.run.start(input={})
            _ = [message async for message in thread.messages]

    assert any(
        "messages" in body.get("channels", []) for body in fake.stream_request_bodies
    )


async def test_messages_yields_async_chat_model_stream_and_text_deltas():
    fake = FakeServer()
    fake.script(
        [
            lifecycle_started_event(seq=0),
            message_start_event(seq=1, message_id="msg-1", run_id="run-1"),
            message_text_delta_event(seq=2, text="hel", message_id="msg-1"),
            message_text_delta_event(seq=3, text="lo", message_id="msg-1"),
            message_text_finish_event(seq=4, text="hello", message_id="msg-1"),
            message_finish_event(
                seq=5, input_tokens=2, output_tokens=3, message_id="msg-1"
            ),
            lifecycle_completed_event(seq=6),
        ]
    )
    asgi = httpx.ASGITransport(app=fake.app)
    async with httpx.AsyncClient(transport=asgi, base_url="http://test") as raw:
        threads = ThreadsClient(HttpClient(raw))
        async with threads.stream(thread_id="t-1", assistant_id="agent") as thread:
            await thread.run.start(input={})
            streams = [message async for message in thread.messages]

    assert len(streams) == 1
    message = streams[0]
    assert isinstance(message, AsyncChatModelStream)
    assert message.message_id == "msg-1"
    assert [delta async for delta in message.text] == ["hel", "lo"]
    assert await message.text == "hello"
    output = await message.output
    assert isinstance(output, AIMessage)
    assert output.id == "msg-1"
    assert output.content == [{"type": "text", "text": "hello", "index": 0}]
    assert output.usage_metadata == {
        "input_tokens": 2,
        "output_tokens": 3,
        "total_tokens": 5,
    }


async def test_messages_multiple_messages_are_distinct_streams():
    fake = FakeServer()
    fake.script(
        [
            lifecycle_started_event(seq=0),
            message_start_event(seq=1, message_id="msg-1", run_id="run-1"),
            message_text_delta_event(seq=2, text="one", message_id="msg-1"),
            message_text_finish_event(seq=3, text="one", message_id="msg-1"),
            message_finish_event(seq=4, message_id="msg-1"),
            message_start_event(seq=5, message_id="msg-2", run_id="run-2"),
            message_text_delta_event(seq=6, text="two", message_id="msg-2"),
            message_text_finish_event(seq=7, text="two", message_id="msg-2"),
            message_finish_event(seq=8, message_id="msg-2"),
            lifecycle_completed_event(seq=9),
        ]
    )
    asgi = httpx.ASGITransport(app=fake.app)
    async with httpx.AsyncClient(transport=asgi, base_url="http://test") as raw:
        threads = ThreadsClient(HttpClient(raw))
        async with threads.stream(thread_id="t-1", assistant_id="agent") as thread:
            await thread.run.start(input={})
            streams = [message async for message in thread.messages]

    assert [stream.message_id for stream in streams] == ["msg-1", "msg-2"]
    assert [await stream.text for stream in streams] == ["one", "two"]


async def test_messages_ignores_nested_namespace_for_root_projection():
    fake = FakeServer()
    fake.script(
        [
            lifecycle_started_event(seq=0),
            message_start_event(seq=1, namespace=["child:1"], message_id="nested"),
            message_text_delta_event(seq=2, namespace=["child:1"], text="nested"),
            message_finish_event(seq=3, namespace=["child:1"]),
            lifecycle_completed_event(seq=4),
        ]
    )
    asgi = httpx.ASGITransport(app=fake.app)
    async with httpx.AsyncClient(transport=asgi, base_url="http://test") as raw:
        threads = ThreadsClient(HttpClient(raw))
        async with threads.stream(thread_id="t-1", assistant_id="agent") as thread:
            await thread.run.start(input={})
            streams = [message async for message in thread.messages]

    assert streams == []


async def test_messages_error_event_fails_active_stream():
    fake = FakeServer()
    fake.script(
        [
            lifecycle_started_event(seq=0),
            message_start_event(seq=1, message_id="msg-1"),
            message_error_event(seq=2, message="model failed", message_id="msg-1"),
            lifecycle_completed_event(seq=3),
        ]
    )
    asgi = httpx.ASGITransport(app=fake.app)
    async with httpx.AsyncClient(transport=asgi, base_url="http://test") as raw:
        threads = ThreadsClient(HttpClient(raw))
        async with threads.stream(thread_id="t-1", assistant_id="agent") as thread:
            await thread.run.start(input={})
            streams = [message async for message in thread.messages]

    assert len(streams) == 1
    with pytest.raises(RuntimeError, match="model failed"):
        await streams[0].output


async def test_messages_concurrent_same_run_id_route_independently():
    """Two messages sharing a run_id must route to independent streams.

    The old `_message_route_key` keyed on `run_id` when present, so both
    message-start events mapped to the same `active` slot and the second
    overwrote the first. Subsequent deltas and finish events all routed to
    the wrong (or missing) stream.
    """
    fake = FakeServer()
    fake.script(
        [
            lifecycle_started_event(seq=0),
            # Both messages share run_id="run-1" (same agent turn)
            message_start_event(seq=1, message_id="msg-A", run_id="run-1"),
            message_start_event(seq=2, message_id="msg-B", run_id="run-1"),
            message_text_delta_event(seq=3, text="alpha", message_id="msg-A"),
            message_text_delta_event(seq=4, text="beta", message_id="msg-B"),
            message_finish_event(seq=5, message_id="msg-A"),
            message_finish_event(seq=6, message_id="msg-B"),
            lifecycle_completed_event(seq=7),
        ]
    )
    asgi = httpx.ASGITransport(app=fake.app)
    async with httpx.AsyncClient(transport=asgi, base_url="http://test") as raw:
        threads = ThreadsClient(HttpClient(raw))
        async with threads.stream(thread_id="t-1", assistant_id="agent") as thread:
            await thread.run.start(input={})
            streams = [msg async for msg in thread.messages]

    assert [s.message_id for s in streams] == ["msg-A", "msg-B"]
    assert [await s.text for s in streams] == ["alpha", "beta"]


async def test_messages_orphan_delta_without_matching_key_is_dropped():
    """A delta whose message_id doesn't match any active stream must be dropped.

    The old code fell back to routing the event to the only active stream when
    `len(active) == 1`, causing orphan/mismatched deltas to silently corrupt
    an unrelated stream's content.
    """
    fake = FakeServer()
    fake.script(
        [
            lifecycle_started_event(seq=0),
            message_start_event(seq=1, message_id="msg-A"),
            # Delta with a mismatched message_id — must be dropped, not routed to msg-A.
            message_text_delta_event(seq=2, text="orphan", message_id="msg-UNKNOWN"),
            message_text_delta_event(seq=3, text="real", message_id="msg-A"),
            message_finish_event(seq=4, message_id="msg-A"),
            lifecycle_completed_event(seq=5),
        ]
    )
    asgi = httpx.ASGITransport(app=fake.app)
    async with httpx.AsyncClient(transport=asgi, base_url="http://test") as raw:
        threads = ThreadsClient(HttpClient(raw))
        async with threads.stream(thread_id="t-1", assistant_id="agent") as thread:
            await thread.run.start(input={})
            streams = [msg async for msg in thread.messages]

    assert len(streams) == 1
    # Only the correctly-keyed delta "real" must appear; "orphan" must be dropped.
    assert await streams[0].text == "real"
