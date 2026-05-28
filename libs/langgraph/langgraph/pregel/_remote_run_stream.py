# libs/langgraph/langgraph/pregel/_remote_run_stream.py
from __future__ import annotations

import logging
import sys
from collections.abc import AsyncIterator, Iterator
from types import TracebackType
from typing import Any

from langchain_core.runnables import RunnableConfig
from langgraph_sdk._async.stream import AsyncThreadStream
from langgraph_sdk._sync.stream import SyncThreadStream
from langgraph_sdk.client import LangGraphClient, SyncLangGraphClient

logger = logging.getLogger(__name__)


class _RemoteGraphRunStream:
    """Sync adapter: SyncThreadStream -> GraphRunStream surface."""

    def __init__(
        self,
        *,
        sync_client: SyncLangGraphClient,
        sdk_thread: SyncThreadStream,
        input: Any,
        config: RunnableConfig | None,
        metadata: dict[str, Any] | None,
    ) -> None:
        self._client = sync_client
        self._sdk = sdk_thread
        self._start_kwargs: dict[str, Any] = {
            "input": input,
            "config": config,
            "metadata": metadata,
        }
        self._run_id: str | None = None
        self._closed = False
        self._events_iter: Iterator[Any] | None = None

    def __enter__(self) -> _RemoteGraphRunStream:
        if self._closed:
            raise RuntimeError("_RemoteGraphRunStream already closed")
        self._sdk.__enter__()
        try:
            result = self._sdk.run.start(**self._start_kwargs)
        except BaseException:
            self._sdk.__exit__(*sys.exc_info())
            raise
        self._run_id = result["run_id"]
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        if self._closed:
            return
        self._closed = True
        self._sdk.__exit__(exc_type, exc, tb)

    @property
    def output(self) -> Any:
        return self._sdk.output

    @property
    def interrupted(self) -> bool:
        """Whether the remote run is currently paused at an interrupt.

        Reads the SDK's current value without blocking. This differs from
        local `GraphRunStream.interrupted`, which drives the run to terminal
        before returning the flag. Sync callers needing a wait-for-interrupt
        pattern should switch to the async API and drain a projection.
        """
        return self._sdk.interrupted

    @property
    def interrupts(self) -> list[Any]:
        """Current outstanding interrupt payloads (non-blocking snapshot)."""
        return list(self._sdk.interrupts)

    def abort(self) -> None:
        if self._closed:
            return
        self._closed = True
        if self._run_id is not None:
            try:
                self._client.runs.cancel(self._sdk.thread_id, self._run_id, wait=False)
            except Exception:
                logger.debug("abort: runs.cancel failed", exc_info=True)
        try:
            self._sdk.close()
        except Exception:
            logger.debug("abort: sdk.close failed", exc_info=True)

    def __iter__(self) -> Iterator[Any]:
        if self._events_iter is None:
            self._events_iter = iter(self._sdk.events)
        return self._events_iter

    def interleave(self, *names: str) -> Iterator[tuple[str, Any]]:
        """Not yet implemented on RemoteGraph.

        Local `GraphRunStream.interleave` yields typed wrapper objects
        (chat-model streams, tool-call handles, snapshot dicts) pushed by
        in-process transformers. The remote analog requires an SDK-side
        refactor so projection decoders can run off a single shared
        subscription instead of each one owning its own queue. Tracked as
        a follow-up sdk-py PR; once shipped, this method becomes a
        one-line passthrough to the new SDK primitive.
        """
        raise NotImplementedError(
            "RemoteGraph.stream_events(version='v3').interleave() is not "
            "yet implemented; requires an upcoming sdk-py refactor that "
            "extracts per-channel decoders from the projection iterators."
        )


class _AsyncRemoteGraphRunStream:
    """Async adapter: AsyncThreadStream -> AsyncGraphRunStream surface."""

    def __init__(
        self,
        *,
        client: LangGraphClient,
        sdk_thread: AsyncThreadStream,
        input: Any,
        config: RunnableConfig | None,
        metadata: dict[str, Any] | None,
    ) -> None:
        self._client = client
        self._sdk = sdk_thread
        self._start_kwargs: dict[str, Any] = {
            "input": input,
            "config": config,
            "metadata": metadata,
        }
        self._run_id: str | None = None
        self._closed = False
        self._events_aiter: AsyncIterator[Any] | None = None

    async def __aenter__(self) -> _AsyncRemoteGraphRunStream:
        if self._closed:
            raise RuntimeError("_AsyncRemoteGraphRunStream already closed")
        await self._sdk.__aenter__()
        try:
            result = await self._sdk.run.start(**self._start_kwargs)
        except BaseException:
            await self._sdk.__aexit__(*sys.exc_info())
            raise
        self._run_id = result["run_id"]
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        if self._closed:
            return
        self._closed = True
        await self._sdk.__aexit__(exc_type, exc, tb)

    @property
    def output(self) -> Any:
        return self._sdk.output

    @property
    async def interrupted(self) -> bool:
        """Whether the remote run is currently paused at an interrupt.

        Reads the SDK's current value without blocking. This differs from
        local `AsyncGraphRunStream.interrupted`, which drives the run to
        terminal before returning the flag. Callers that need a
        wait-for-interrupt pattern should drain a projection (e.g.,
        `async for snap in stream._sdk.values`) until the SDK's paused
        sentinel fires, then check this property.
        """
        return self._sdk.interrupted

    @property
    async def interrupts(self) -> list[Any]:
        """Current outstanding interrupt payloads.

        Non-blocking; reads the SDK's current snapshot. See `interrupted`
        for the divergence from local v3 semantics.
        """
        return list(self._sdk.interrupts)

    async def abort(self) -> None:
        if self._closed:
            return
        self._closed = True
        if self._run_id is not None:
            try:
                await self._client.runs.cancel(
                    self._sdk.thread_id, self._run_id, wait=False
                )
            except Exception:
                logger.debug("abort: runs.cancel failed", exc_info=True)
        try:
            await self._sdk.close()
        except Exception:
            logger.debug("abort: sdk.close failed", exc_info=True)

    def __aiter__(self) -> AsyncIterator[Any]:
        if self._events_aiter is None:
            self._events_aiter = self._sdk.events.__aiter__()
        return self._events_aiter

    # Note: deliberately no `interleave()` on the async adapter. Local
    # `AsyncGraphRunStream` doesn't have one either (async callers compose
    # with `asyncio.gather` / `asyncio.as_completed`). The sync adapter
    # provides `interleave()` because sync callers have no comparable
    # primitive for iterating multiple iterators concurrently.
