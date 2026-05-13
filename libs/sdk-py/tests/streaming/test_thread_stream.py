from __future__ import annotations

import re
import uuid

import httpx
import pytest

from langgraph_sdk._async.http import HttpClient
from langgraph_sdk._async.stream import AsyncThreadStream
from langgraph_sdk._async.threads import ThreadsClient


def test_thread_stream_stores_thread_id_and_assistant_id():
    stream = AsyncThreadStream(
        client=None,  # transport not built until __aenter__
        thread_id="t-1",
        assistant_id="agent",
    )
    assert stream.thread_id == "t-1"
    assert stream.assistant_id == "agent"


async def test_aenter_returns_self():
    stream = AsyncThreadStream(client=None, thread_id="t-1", assistant_id="agent")
    async with stream as entered:
        assert entered is stream


async def test_aexit_marks_closed():
    stream = AsyncThreadStream(client=None, thread_id="t-1", assistant_id="agent")
    async with stream:
        assert stream._closed is False
    assert stream._closed is True


async def test_close_is_idempotent():
    stream = AsyncThreadStream(client=None, thread_id="t-1", assistant_id="agent")
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
