"""Child lifecycle completion must not finish the parent thread's output."""

from __future__ import annotations

import asyncio
from typing import cast

import httpx
import pytest
from langchain_protocol import Event

from langgraph_sdk._async.http import HttpClient
from langgraph_sdk._async.threads import ThreadsClient
from langgraph_sdk._sync.http import SyncHttpClient
from langgraph_sdk._sync.stream import _BlockingResult
from langgraph_sdk._sync.threads import SyncThreadsClient
from streaming._events import input_requested_event, lifecycle_event


@pytest.mark.parametrize("phase", ["completed", "failed"])
async def test_async_child_terminal_does_not_finish_root(phase):
    async with httpx.AsyncClient(base_url="http://test") as raw:
        thread = ThreadsClient(HttpClient(raw)).stream(
            thread_id="t-1", assistant_id="agent"
        )
        thread._run_done = asyncio.get_running_loop().create_future()
        await thread._apply_lifecycle_event(
            cast(Event, input_requested_event(namespace=["sub:1"]))
        )
        await thread._apply_lifecycle_event(
            cast(Event, lifecycle_event(namespace=["sub:1"], event=phase))
        )
        assert not thread._run_done.done()
        assert thread.interrupted
        assert thread.interrupts[0]["namespace"] == ["sub:1"]

        await thread._apply_lifecycle_event(
            cast(Event, lifecycle_event(event="completed"))
        )
        assert thread._run_done.done()
        assert (await thread._run_done).error is None
        assert not thread.interrupted
        assert thread.interrupts == []


@pytest.mark.parametrize("phase", ["completed", "failed"])
def test_sync_child_terminal_does_not_finish_root(phase):
    with httpx.Client(base_url="http://test") as raw:
        thread = SyncThreadsClient(SyncHttpClient(raw)).stream(
            thread_id="t-1", assistant_id="agent"
        )
        thread._run_done = _BlockingResult()
        thread._apply_lifecycle_event(
            cast(Event, input_requested_event(namespace=["sub:1"]))
        )
        thread._apply_lifecycle_event(
            cast(Event, lifecycle_event(namespace=["sub:1"], event=phase))
        )
        assert not thread._run_done.done()
        assert thread.interrupted
        assert thread.interrupts[0]["namespace"] == ["sub:1"]

        thread._apply_lifecycle_event(cast(Event, lifecycle_event(event="completed")))
        assert thread._run_done.done()
        assert thread._run_done.result().error is None
        assert not thread.interrupted
        assert thread.interrupts == []
