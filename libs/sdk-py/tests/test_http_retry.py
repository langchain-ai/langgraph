"""Tests for transient network-error retries in the SDK HTTP clients.

Issue #5819: the SDK clients propagated raw `httpx.ReadError` (and other
network-level errors) without any retry. `HttpClient`/`SyncHttpClient` now
retry those transient errors with exponential backoff.
"""

import asyncio

import httpx
import pytest

from langgraph_sdk._async.http import HttpClient
from langgraph_sdk._sync.http import SyncHttpClient


class _TransientTransport(httpx.MockTransport):
    """Raise `httpx.ReadError` for the first `fail_times` calls, then succeed."""

    def __init__(self, fail_times: int) -> None:
        self.fail_times = fail_times
        self.calls = 0
        super().__init__(self._handler)

    def _handler(self, _request: httpx.Request) -> httpx.Response:
        self.calls += 1
        if self.calls <= self.fail_times:
            raise httpx.ReadError("simulated network drop")
        return httpx.Response(200, json={"ok": True})


async def test_async_retries_transient_read_error() -> None:
    t = _TransientTransport(fail_times=2)
    async with httpx.AsyncClient(transport=t, base_url="http://test") as client:
        hc = HttpClient(client, max_retries=3)
        assert await hc.get("/foo") == {"ok": True}
    assert t.calls == 3  # 2 failures + 1 success


async def test_async_gives_up_after_max_retries() -> None:
    t = _TransientTransport(fail_times=10)
    async with httpx.AsyncClient(transport=t, base_url="http://test") as client:
        hc = HttpClient(client, max_retries=3)
        with pytest.raises(httpx.ReadError):
            await hc.get("/foo")
    assert t.calls == 3


def test_sync_retries_transient_read_error() -> None:
    t = _TransientTransport(fail_times=1)
    with httpx.Client(transport=t, base_url="http://test") as client:
        hc = SyncHttpClient(client, max_retries=3)
        assert hc.get("/foo") == {"ok": True}
    assert t.calls == 2  # 1 failure + 1 success


def test_sync_gives_up_after_max_retries() -> None:
    t = _TransientTransport(fail_times=10)
    with httpx.Client(transport=t, base_url="http://test") as client:
        hc = SyncHttpClient(client, max_retries=3)
        with pytest.raises(httpx.ReadError):
            hc.get("/foo")
    assert t.calls == 3


def test_event_loop_retry() -> None:
    asyncio.run(test_async_retries_transient_read_error())
    asyncio.run(test_async_gives_up_after_max_retries())
