# libs/langgraph/langgraph/pregel/_remote_run_stream.py
from __future__ import annotations

import logging
import sys
from collections.abc import AsyncIterator, Iterator, Mapping
from types import TracebackType
from typing import Any, cast

from langchain_core.runnables import RunnableConfig
from langgraph_sdk._async.stream import AsyncThreadStream
from langgraph_sdk._sync.stream import SyncThreadStream
from langgraph_sdk.client import LangGraphClient, SyncLangGraphClient
from langgraph_sdk.stream.decoders import DataDecoder

from langgraph.types import Command

logger = logging.getLogger(__name__)


def _translate_command_input(input: Any) -> Any:
    """Translate a local `Command` into the v3 wire `input`, else passthrough.

    The v3 server decides start-vs-resume from thread state (an interrupted
    run or pending interrupts) and, on resume, wraps the whole `input` as
    `{"resume": input}` itself. So a resume `Command` must surface its raw
    `resume` value as the wire `input` (not the serialized dataclass, which
    the server would double-wrap). The v3 `run.start` path has no `goto` /
    `update` channel, so those are rejected.

    `langgraph_sdk` is upstream of `langgraph`, so this `Command`-aware
    marshalling lives here on the adapter (langgraph) side of the boundary.
    """
    if isinstance(input, Command):
        if input.goto or input.update:
            raise NotImplementedError(
                "RemoteGraph v3 streaming supports `Command(resume=...)` only; "
                "`goto` / `update` are not supported by the v3 `run.start` path."
            )
        return input.resume
    return input


class _ChannelProjection:
    """Decoded projection for a wire channel the SDK doesn't type natively.

    Subscribes to `channel` and decodes each event's `params["data"]` through the
    SDK's `DataDecoder` — the same decoder the SDK's own plain-payload projections
    (`values` / `updates` / `checkpoints` / `tasks`) use, which yields the item
    shape that local's `UpdatesTransformer` / `CheckpointsTransformer` /
    `TasksTransformer` / `CustomTransformer` push, so iterating this matches the
    corresponding local projection. Iterate with `for` against a sync stream and
    `async for` against an async stream (matching the underlying SDK). Opening
    the subscription requires the stream to be entered (`with` / `async with`).
    """

    def __init__(self, sdk: AsyncThreadStream | SyncThreadStream, channel: str) -> None:
        self._sdk = sdk
        self._channel = channel

    def __iter__(self) -> Iterator[Any]:
        # Sync lane: the sync adapter's SDK returns a sync iterator here.
        decoder = DataDecoder(self._channel)
        events = cast(Iterator[Any], self._sdk.subscribe([self._channel]))
        for event in events:
            yield from decoder.feed(event)

    def __aiter__(self) -> AsyncIterator[Any]:
        return self._aiter()

    async def _aiter(self) -> AsyncIterator[Any]:
        # Async lane: the async adapter's SDK returns an async iterator here.
        decoder = DataDecoder(self._channel)
        events = cast(AsyncIterator[Any], self._sdk.subscribe([self._channel]))
        async for event in events:
            for item in decoder.feed(event):
                yield item


class _ProjectionRegistry(Mapping[str, Any]):
    """Read-only name -> projection registry mirroring local `GraphRunStream.extensions`.

    Resolution follows the langchain-protocol wire channels, and every entry
    yields the same decoded item shape local does (`params.data`):

    - `values` / `messages` / `tool_calls` / `subgraphs` resolve to the SDK's
      decoded typed projections. `tool_calls` is the `tools` channel — tool
      *execution* events, distinct from the tool-call *inputs* inside `messages`.
    - `updates` / `checkpoints` / `tasks` / `custom` have no typed SDK
      projection, so they resolve to a `_ChannelProjection` that subscribes to
      the channel and yields `params.data` — matching the local transformer
      output for those channels.
    - any other name is a specific custom-extension channel
      (`thread.extensions[name]`, i.e. `custom:<name>`).

    `lifecycle` is intentionally absent: local derives a status payload from it
    rather than yielding `params.data`, and the SDK consumes it as control-plane
    (driving `output` / `interrupted`), so its shape can't be matched — it
    remains reachable via the raw `events` iterator. `debug` is absent too: it
    is not a v3 wire channel.
    """

    # Channels the SDK decodes into typed projections.
    _TYPED = ("values", "messages", "tool_calls", "subgraphs")
    # Wire channels with no typed SDK projection — decoded here to match local.
    _DECODED = ("updates", "checkpoints", "tasks", "custom")
    _NATIVE = _TYPED + _DECODED

    def __init__(self, sdk: AsyncThreadStream | SyncThreadStream) -> None:
        self._sdk = sdk

    def __getitem__(self, name: str) -> Any:
        if name in self._TYPED:
            return getattr(self._sdk, name)
        if name in self._DECODED:
            return _ChannelProjection(self._sdk, name)
        return self._sdk.extensions[name]

    def __iter__(self) -> Iterator[str]:
        return iter(self._NATIVE)

    def __len__(self) -> int:
        return len(self._NATIVE)


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
            "input": _translate_command_input(input),
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

    @property
    def values(self) -> Any:
        """Live state-snapshot projection (mirrors local `run.values`)."""
        return self._sdk.values

    @property
    def messages(self) -> Any:
        """Live message-stream projection (mirrors local `run.messages`)."""
        return self._sdk.messages

    @property
    def subgraphs(self) -> Any:
        """Subgraph-handle projection (mirrors local `run.subgraphs`)."""
        return self._sdk.subgraphs

    @property
    def tool_calls(self) -> Any:
        """Tool-execution projection (the `tools` channel).

        These are tool *execution* events (started / output / finished),
        distinct from the tool-call *inputs* carried inside `messages`.
        """
        return self._sdk.tool_calls

    @property
    def extensions(self) -> Mapping[str, Any]:
        """Name -> projection registry (mirrors local `run.extensions`)."""
        return _ProjectionRegistry(self._sdk)

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
        yield from self._sdk.interleave_projections(list(names))


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
            "input": _translate_command_input(input),
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

    async def output(self) -> Any:
        """Drive the remote run to completion and return the final state.

        Awaits the SDK's terminal-state awaitable, matching local
        `AsyncGraphRunStream.output()` (a method, not a property, so
        `run.output` without `await` fails at type-check time rather than
        silently yielding a coroutine).
        """
        return await self._sdk.output

    async def interrupted(self) -> bool:
        """Whether the remote run is currently paused at an interrupt.

        Reads the SDK's current value without blocking. This differs from
        local `AsyncGraphRunStream.interrupted()`, which drives the run to
        terminal before returning the flag. Callers that need a
        wait-for-interrupt pattern should drain a projection (e.g.,
        `async for snap in stream._sdk.values`) until the SDK's paused
        sentinel fires, then call this method.
        """
        return self._sdk.interrupted

    async def interrupts(self) -> list[Any]:
        """Current outstanding interrupt payloads.

        Non-blocking; reads the SDK's current snapshot. See `interrupted`
        for the divergence from local v3 semantics.
        """
        return list(self._sdk.interrupts)

    @property
    def values(self) -> Any:
        """Live state-snapshot projection (mirrors local `run.values`)."""
        return self._sdk.values

    @property
    def messages(self) -> Any:
        """Live message-stream projection (mirrors local `run.messages`)."""
        return self._sdk.messages

    @property
    def subgraphs(self) -> Any:
        """Subgraph-handle projection (mirrors local `run.subgraphs`)."""
        return self._sdk.subgraphs

    @property
    def tool_calls(self) -> Any:
        """Tool-execution projection (the `tools` channel).

        These are tool *execution* events (started / output / finished),
        distinct from the tool-call *inputs* carried inside `messages`.
        """
        return self._sdk.tool_calls

    @property
    def extensions(self) -> Mapping[str, Any]:
        """Name -> projection registry (mirrors local `run.extensions`)."""
        return _ProjectionRegistry(self._sdk)

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
