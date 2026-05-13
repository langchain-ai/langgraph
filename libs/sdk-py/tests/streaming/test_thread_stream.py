from __future__ import annotations

from langgraph_sdk._async.stream import AsyncThreadStream


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
