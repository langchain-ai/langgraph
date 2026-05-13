"""Async thread-centric streaming surface for the v3 protocol.

Phase 2 skeleton: `AsyncThreadStream` is an async context manager that owns
a `ProtocolSseTransport` for one thread, dispatches `run.start` commands,
and exposes a raw `events` async iterable. No projections yet — those land
in Phases 3+.

Direct port of `libs/sdk/src/client/stream/index.ts` (skeleton subset).
"""

from __future__ import annotations

from typing import Any

import httpx

from langgraph_sdk.stream.transport import ProtocolSseTransport


class AsyncThreadStream:
    """Async context manager for one thread's v3 streaming session.

    Construct via `client.threads.stream(thread_id=None, *, assistant_id, ...)`
    rather than instantiating directly.
    """

    def __init__(
        self,
        *,
        client: httpx.AsyncClient,
        thread_id: str,
        assistant_id: str,
    ) -> None:
        self._http_client = client
        self.thread_id = thread_id
        self.assistant_id = assistant_id
        self._closed = False
        self._transport: ProtocolSseTransport | None = None

    async def __aenter__(self) -> AsyncThreadStream:
        self._transport = ProtocolSseTransport(
            client=self._http_client,
            thread_id=self.thread_id,
        )
        return self

    async def __aexit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        await self.close()

    async def close(self) -> None:
        """Tear down the thread stream. Idempotent."""
        if self._closed:
            return
        self._closed = True
        if self._transport is not None:
            await self._transport.close()
