"""Central event dispatcher with transformer pipeline for StreamingHandler.

``StreamMux`` is the sync-safe core: it holds the main
:class:`EventLog`, tracks discovered namespaces for subgraph stream
creation, and pipes every event through the registered
:class:`StreamTransformer` pipeline before appending it to the log.

``AsyncStreamMux`` extends the base with async subscription endpoints
(output futures, namespace waiters, filtered event iteration).
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import AsyncIterator
from typing import Any

from langgraph.stream._event_log import EventLog
from langgraph.stream._types import InterruptPayload, ProtocolEvent, StreamTransformer
from langgraph.stream.stream_channel import StreamChannel, is_stream_channel


class StreamMux:
    """Sync-safe event dispatcher for the StreamingHandler infrastructure.

    The mux owns the main event log, applies the transformer pipeline to
    every incoming event, and tracks namespace discovery and latest values.

    For async subscription endpoints (output futures, namespace waiters,
    filtered event iteration), use :class:`AsyncStreamMux`.
    """

    def __init__(self, transformers: list[StreamTransformer] | None = None) -> None:
        self._event_log: EventLog[ProtocolEvent] = EventLog()
        self._transformers: list[StreamTransformer] = list(transformers or [])
        self._channels: list[StreamChannel[Any]] = []
        self._current_namespace: list[str] = []
        self._next_emit_seq: int = 0

        # Namespace discovery: maps top-level ns segment → True
        self._discovered_ns: dict[str, bool] = {}

        # Latest values per namespace (list-of-strings key)
        self._latest_values: dict[str, Any] = {}

        # Interrupt tracking
        self._interrupts: list[InterruptPayload] = []
        self._interrupted = False

        # Closed state
        self._closed = False
        self._error: BaseException | None = None

    # -- Producer API -------------------------------------------------------

    def push(self, event: ProtocolEvent) -> None:
        """Push an event through the transformer pipeline and into the log.

        Each registered transformer's ``process()`` is called in order.
        If any transformer returns ``False``, the event is suppressed
        (not appended to the main log).
        """
        if self._closed:
            return

        # Mux is the sole seq assigner — ensures all events in the log
        # (including those from StreamChannel forwarders) share a single
        # monotonically increasing counter.
        event["seq"] = self._next_emit_seq
        self._next_emit_seq += 1

        # Track namespace
        ns = event["params"].get("namespace", [])
        if ns:
            top_segment = ns[0]
            if top_segment not in self._discovered_ns:
                self._discovered_ns[top_segment] = True
                self._on_ns_discovered(top_segment)

        # Track values
        if event["method"] == "values":
            ns_key = _ns_key(ns)
            self._latest_values[ns_key] = event["params"]["data"]

        # Track interrupts from values events
        if event["method"] == "values":
            data = event["params"]["data"]
            if isinstance(data, dict) and "__interrupt__" in data:
                interrupt_info = data["__interrupt__"]
                if isinstance(interrupt_info, (list, tuple)):
                    for item in interrupt_info:
                        iid = getattr(item, "id", None) or str(id(item))
                        self._interrupts.append(
                            InterruptPayload(
                                interrupt_id=iid,
                                payload=item,
                            )
                        )
                    self._interrupted = True

        # Run transformer pipeline
        self._current_namespace = ns
        keep = True
        for transformer in self._transformers:
            result = transformer.process(event)
            if result is False:
                keep = False
        self._current_namespace = []

        # Append to main log if not suppressed
        if keep:
            self._event_log.append(event)

    def close(self, output: Any = None) -> None:
        """Close the mux, finalizing transformers and the event log."""
        if self._closed:
            return
        self._closed = True

        # Finalize transformers (optional method)
        for transformer in self._transformers:
            if hasattr(transformer, "finalize"):
                transformer.finalize()

        # Close wired channels
        for channel in self._channels:
            channel._close()

        # Close the event log
        self._event_log.close()

    def fail(self, error: BaseException) -> None:
        """Fail the mux, propagating the error to transformers and channels."""
        if self._closed:
            return
        self._closed = True
        self._error = error

        # Fail transformers (optional method)
        for transformer in self._transformers:
            if hasattr(transformer, "fail"):
                transformer.fail(error)

        # Fail wired channels
        for channel in self._channels:
            channel._fail(error)

        # Fail the event log
        self._event_log.fail(error)

    # -- Inspection ---------------------------------------------------------

    @property
    def interrupted(self) -> bool:
        return self._interrupted

    @property
    def interrupts(self) -> list[InterruptPayload]:
        return list(self._interrupts)

    @property
    def event_log(self) -> EventLog[ProtocolEvent]:
        return self._event_log

    def get_latest_values(self, ns: list[str] | None = None) -> Any:
        """Return the most recent values for a namespace."""
        return self._latest_values.get(_ns_key(ns or []))

    # -- Internal -----------------------------------------------------------

    def _on_ns_discovered(self, segment: str) -> None:
        """Hook called when a new top-level namespace segment is discovered.

        The base implementation is a no-op.  :class:`AsyncStreamMux`
        overrides this to wake namespace waiters.
        """

    def register_transformer(self, transformer: StreamTransformer) -> None:
        """Register a new transformer and replay all buffered events through it.

        This is the safe way to add a late-arriving transformer after the mux
        has already started processing events.  The sequence is:

        1. Snapshot the current log length (no await → no gap possible in
           asyncio's cooperative threading model).
        2. Append the transformer so future ``push()`` calls reach it.
        3. Replay events ``[0, snapshot)`` through the transformer.
        4. If the mux is already closed, call ``finalize()`` immediately so
           the transformer's log/channel terminates cleanly.

        ``process()`` is only called for events whose namespace starts with
        any prefix — callers that need namespace filtering should do so inside
        their ``process()`` implementation, or wrap this call with their own
        filtering logic.
        """
        snapshot = len(self._event_log)
        self._transformers.append(transformer)
        for i in range(snapshot):
            transformer.process(self._event_log[i])
        if self._closed:
            if hasattr(transformer, "finalize"):
                transformer.finalize()

    def wire_channels(self, projection: Any) -> None:
        """Scan *projection* for :class:`StreamChannel` instances and wire them.

        For each ``StreamChannel`` found, registers a push callback that
        appends a :class:`ProtocolEvent` directly to the main event log
        with ``method`` set to the channel's name.

        Channel events bypass the transformer pipeline (matching the JS
        implementation).  They are visible to raw event iteration and
        remote SDK clients but not to other transformers' ``process()``.
        """
        if projection is None:
            return
        items: dict[str, Any] = {}
        if isinstance(projection, dict):
            items = projection
        elif hasattr(projection, "__dict__"):
            items = vars(projection)
        for _key, value in items.items():
            if is_stream_channel(value):
                channel: StreamChannel[Any] = value
                self._channels.append(channel)

                def _make_forwarder(ch: StreamChannel[Any]) -> Any:
                    def _forward(item: Any) -> None:
                        if self._closed:
                            return
                        # Append directly to the event log, bypassing
                        # the transformer pipeline.  This matches the JS
                        # implementation and avoids re-entrancy bugs
                        # (namespace clobbering, infinite recursion).
                        self._event_log.append(
                            ProtocolEvent(
                                type="event",
                                seq=self._next_emit_seq,
                                method=ch.channel_name,
                                params={
                                    "namespace": list(self._current_namespace),
                                    "timestamp": int(time.time() * 1000),
                                    "data": item,
                                },
                            )
                        )
                        self._next_emit_seq += 1

                    return _forward

                channel._wire(_make_forwarder(channel))


class AsyncStreamMux(StreamMux):
    """Async extension of :class:`StreamMux`.

    Adds output futures, namespace waiters, and async subscription
    endpoints (``subscribe_events``, ``subscribe_subgraphs``,
    ``get_output_future``).
    """

    def __init__(self, transformers: list[StreamTransformer] | None = None) -> None:
        super().__init__(transformers)
        # Waiters for new namespace discovery
        self._ns_waiters: list[asyncio.Future[None]] = []
        # Output promise tracking
        self._output_futures: dict[str, asyncio.Future[Any]] = {}

    # -- Producer API overrides ---------------------------------------------

    def close(self, output: Any = None) -> None:
        """Close the mux, resolving all output futures."""
        if self._closed:
            return

        # Let the base class finalize transformers, channels, and event log
        super().close(output)

        # Resolve output futures
        for ns_key, fut in self._output_futures.items():
            if not fut.done():
                value = self._latest_values.get(ns_key)
                try:
                    fut.get_loop().call_soon_threadsafe(fut.set_result, value)
                except RuntimeError:
                    pass

        # Wake namespace waiters
        self._wake_ns_waiters()

    def fail(self, error: BaseException) -> None:
        """Fail the mux, rejecting all output futures."""
        if self._closed:
            return

        # Let the base class fail transformers, channels, and event log
        super().fail(error)

        # Reject output futures
        for fut in self._output_futures.values():
            if not fut.done():
                try:
                    fut.get_loop().call_soon_threadsafe(fut.set_exception, error)
                except RuntimeError:
                    pass

        # Wake namespace waiters
        self._wake_ns_waiters()

    # -- Consumer API -------------------------------------------------------

    def subscribe_events(
        self, path: list[str] | None = None
    ) -> AsyncIterator[ProtocolEvent]:
        """Return an async iterator over events matching *path*.

        If *path* is ``None`` or empty, all events are yielded.
        Otherwise, only events whose namespace starts with *path*
        are yielded.
        """
        cursor = aiter(self._event_log)
        if not path:
            return cursor
        return _FilteredEventIterator(cursor, path)

    async def subscribe_subgraphs(
        self, path: list[str] | None = None, offset: int = 0
    ) -> AsyncIterator[str]:
        """Yield top-level namespace segments as they are discovered.

        Each yielded value is the first namespace segment of a newly
        discovered subgraph (e.g. ``"agent:0"``).
        """
        yielded: set[str] = set()
        while True:
            # Yield any newly discovered namespaces
            for ns_segment in list(self._discovered_ns):
                if ns_segment not in yielded:
                    # Filter by path prefix if specified
                    if path:
                        if not ns_segment.startswith(path[0]):
                            continue
                    yielded.add(ns_segment)
                    yield ns_segment

            if self._closed:
                return

            # Wait for new namespaces
            loop = asyncio.get_running_loop()
            fut: asyncio.Future[None] = loop.create_future()
            self._ns_waiters.append(fut)
            await fut

    def get_output_future(self, ns: list[str] | None = None) -> asyncio.Future[Any]:
        """Get or create an output future for a namespace.

        The future resolves to the latest ``values`` event data when
        the mux is closed.
        """
        ns_key = _ns_key(ns or [])
        if ns_key not in self._output_futures:
            loop = asyncio.get_running_loop()
            self._output_futures[ns_key] = loop.create_future()

            # If already closed, resolve immediately
            if self._closed:
                value = self._latest_values.get(ns_key)
                if self._error is not None:
                    self._output_futures[ns_key].set_exception(self._error)
                else:
                    self._output_futures[ns_key].set_result(value)

        return self._output_futures[ns_key]

    # -- Internal -----------------------------------------------------------

    def _on_ns_discovered(self, segment: str) -> None:
        """Wake namespace waiters when a new namespace is discovered."""
        self._wake_ns_waiters()

    def _wake_ns_waiters(self) -> None:
        for fut in self._ns_waiters:
            if not fut.done():
                try:
                    fut.get_loop().call_soon_threadsafe(fut.set_result, None)
                except RuntimeError:
                    pass
        self._ns_waiters.clear()


class _FilteredEventIterator:
    """Async iterator that filters events by namespace prefix."""

    __slots__ = ("_cursor", "_path")

    def __init__(self, cursor: AsyncIterator[ProtocolEvent], path: list[str]) -> None:
        self._cursor = cursor
        self._path = path

    def __aiter__(self) -> _FilteredEventIterator:
        return self

    async def __anext__(self) -> ProtocolEvent:
        while True:
            event = await self._cursor.__anext__()
            ns = event["params"].get("namespace", [])
            if _ns_starts_with(ns, self._path):
                return event


def _ns_key(ns: list[str] | tuple[str, ...]) -> str:
    """Convert a namespace list to a hashable key."""
    return "|".join(ns)


def _ns_starts_with(ns: list[str], prefix: list[str]) -> bool:
    """Check if *ns* starts with *prefix*."""
    if len(ns) < len(prefix):
        return False
    return ns[: len(prefix)] == prefix


__all__ = ["AsyncStreamMux", "StreamMux"]
