"""An in-process transport must still honour the client's timeout.

``get_client()`` with no URL talks to the server through ``httpx.ASGITransport``, which
dispatches directly into the ASGI app. That transport has no socket, so it never looks at
``request.extensions["timeout"]``: a caller who passes ``timeout=0.1`` to bound a runaway
graph waits for the graph instead. These tests use a stand-in transport with the same shape
(a coroutine that never returns in time) so they need no server installed.
"""

from __future__ import annotations

import asyncio
import time

import httpx
import pytest

from langgraph_sdk._shared.utilities import TimeoutEnforcingTransport


class _HungTransport(httpx.AsyncBaseTransport):
    """Accepts the request, then does what a stuck graph does: nothing, for a long time."""

    def __init__(self, delay: float) -> None:
        self.delay = delay
        self.aclosed = False

    async def handle_async_request(
        self,
        request: httpx.Request,  # noqa: ARG002 - the name is fixed by httpx
    ) -> httpx.Response:
        await asyncio.sleep(self.delay)
        return httpx.Response(200, json={"ok": True})

    async def aclose(self) -> None:
        self.aclosed = True


async def test_in_process_transport_honours_the_read_timeout():
    inner = _HungTransport(delay=10)
    transport = TimeoutEnforcingTransport(inner)
    async with httpx.AsyncClient(
        base_url="http://api", transport=transport, timeout=0.2
    ) as client:
        started = time.monotonic()
        with pytest.raises(httpx.ReadTimeout):
            await client.get("/threads")
        elapsed = time.monotonic() - started

    assert elapsed < 2, f"timeout did not bound the request: took {elapsed:.1f}s"
    assert inner.aclosed, "aclose must reach the wrapped transport"


async def test_a_fast_response_is_untouched():
    transport = TimeoutEnforcingTransport(_HungTransport(delay=0))
    async with httpx.AsyncClient(
        base_url="http://api", transport=transport, timeout=5
    ) as client:
        response = await client.get("/threads")

    assert response.status_code == 200


async def test_no_timeout_means_no_deadline():
    """``timeout=None`` must keep meaning "wait indefinitely", not "expire immediately"."""
    transport = TimeoutEnforcingTransport(_HungTransport(delay=0.4))
    async with httpx.AsyncClient(
        base_url="http://api", transport=transport, timeout=None
    ) as client:
        response = await client.get("/threads")

    assert response.status_code == 200


class _StreamingTransport(httpx.AsyncBaseTransport):
    """Hands back the response at the headers and drips the body afterwards.

    This is the shape of the loopback transport inside a LangGraph deployment
    (``langgraph_api.asgi_transport.ASGITransport``): ``handle_async_request`` returns as soon
    as the app sends ``http.response.start``, and body chunks follow through a queue.
    """

    def __init__(self, chunks: int, gap: float, header_delay: float = 0.0) -> None:
        self.chunks = chunks
        self.gap = gap
        self.header_delay = header_delay

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:  # noqa: ARG002
        queue: asyncio.Queue[bytes | None] = asyncio.Queue()
        started = asyncio.Event()

        async def app() -> None:
            await asyncio.sleep(self.header_delay)
            started.set()
            for index in range(self.chunks):
                await asyncio.sleep(self.gap)
                queue.put_nowait(b"data: %d\n\n" % index)
            queue.put_nowait(None)

        task = asyncio.create_task(app())
        try:
            await started.wait()
        except asyncio.CancelledError:
            task.cancel()
            raise
        return httpx.Response(200, stream=_QueueStream(queue, task))


class _QueueStream(httpx.AsyncByteStream):
    def __init__(self, queue: asyncio.Queue, task: asyncio.Task) -> None:
        self._queue = queue
        self._task = task

    async def __aiter__(self):
        while True:
            chunk = await self._queue.get()
            if chunk is None:
                return
            yield chunk

    async def aclose(self) -> None:
        self._task.cancel()


async def test_a_stream_that_keeps_producing_outlives_the_budget():
    """The budget is per read, not per request: a long stream of live chunks must not be cut."""
    transport = TimeoutEnforcingTransport(_StreamingTransport(chunks=6, gap=0.1))
    async with httpx.AsyncClient(
        base_url="http://api", transport=transport, timeout=0.25
    ) as client:
        started = time.monotonic()
        async with client.stream("GET", "/runs/stream") as response:
            chunks = [chunk async for chunk in response.aiter_raw()]
        elapsed = time.monotonic() - started

    assert len(chunks) == 6, f"stream was truncated after {len(chunks)} chunks"
    assert elapsed > 0.25, f"the run did not outlast the budget ({elapsed:.2f}s)"


async def test_a_stream_that_goes_quiet_mid_body_raises_read_timeout():
    """A body that stops arriving is a read timeout, exactly as it is over a socket."""
    transport = TimeoutEnforcingTransport(_StreamingTransport(chunks=1, gap=5))
    started = time.monotonic()
    async with (
        httpx.AsyncClient(
            base_url="http://api", transport=transport, timeout=0.2
        ) as client,
        client.stream("GET", "/runs/stream") as response,
    ):
        with pytest.raises(httpx.ReadTimeout):
            async for _ in response.aiter_raw():
                pass
        elapsed = time.monotonic() - started

    assert elapsed < 2, f"a stalled body was not bounded: {elapsed:.1f}s"
