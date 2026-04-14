"""GraphRunStream and AsyncGraphRunStream for StreamingHandler.

These are the top-level objects returned by
``StreamingHandler.stream()`` / ``StreamingHandler.astream()``.

``AsyncGraphRunStream`` wraps an :class:`AsyncStreamMux` and exposes
``.values``, ``.messages``, ``.subgraphs``, ``.output``, and
``.messages_from()``.

``GraphRunStream`` wraps a :class:`StreamMux` and exposes the sync
equivalents: ``.values``, ``.messages``, and ``.output``.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Callable, Iterator
from typing import Any

from langgraph.stream._convert import convert_to_protocol_event
from langgraph.stream._mux import AsyncStreamMux, StreamMux
from langgraph.stream._types import InterruptPayload, ProtocolEvent, StreamTransformer
from langgraph.stream.chat_model_stream import AsyncChatModelStream, ChatModelStream
from langgraph.stream.transformers import MessagesTransformer, ValuesTransformer

# ---------------------------------------------------------------------------
# Values projection — dual async-iterable + awaitable
# ---------------------------------------------------------------------------


class _ValuesProjection:
    """Async iterable of intermediate state snapshots; awaitable for final."""

    def __init__(
        self,
        mux: AsyncStreamMux,
        values_transformer: ValuesTransformer,
        ns: list[str],
        mapper: Callable[[Any], Any] | None = None,
    ) -> None:
        self._mux = mux
        self._values_transformer = values_transformer
        self._ns = ns
        self._mapper = mapper

    async def __aiter__(self) -> AsyncIterator[Any]:
        log = self._values_transformer.values_log
        cursor = 0
        while True:
            while cursor < len(log):
                item = log[cursor]
                cursor += 1
                if item.get("namespace", []) == self._ns:
                    data = item["data"]
                    if data is not None and self._mapper is not None:
                        yield self._mapper(data)
                    else:
                        yield data
            if self._mux._closed:
                return
            self._mux._notify.clear()
            await self._mux._notify.wait()

    def __await__(self) -> Any:
        return self._await_impl().__await__()

    async def _await_impl(self) -> Any:
        value = await self._mux.get_output_future(self._ns)
        if value is not None and self._mapper is not None:
            return self._mapper(value)
        return value


# ---------------------------------------------------------------------------
# Messages projection
# ---------------------------------------------------------------------------


class _MessagesProjection:
    """Async iterable of :class:`AsyncChatModelStream` instances."""

    def __init__(self, mux: AsyncStreamMux, messages_transformer: MessagesTransformer) -> None:
        self._mux = mux
        self._transformer = messages_transformer

    async def __aiter__(self) -> AsyncIterator[AsyncChatModelStream]:
        log = self._transformer.messages_log
        cursor = 0
        while True:
            while cursor < len(log):
                yield log[cursor]
                cursor += 1
            if self._mux._closed:
                return
            self._mux._notify.clear()
            await self._mux._notify.wait()


# ---------------------------------------------------------------------------
# Subgraphs projection
# ---------------------------------------------------------------------------


class _SubgraphsProjection:
    """Async iterable yielding :class:`AsyncSubgraphRunStream` for each discovered subgraph."""

    def __init__(self, mux: AsyncStreamMux, ns: list[str]) -> None:
        self._mux = mux
        self._ns = ns

    async def __aiter__(self) -> AsyncIterator[AsyncSubgraphRunStream]:
        async for segment in self._mux.subscribe_subgraphs(self._ns):
            child_ns = self._ns + [segment]
            child_transformers: list[StreamTransformer] = [
                ValuesTransformer(),
                MessagesTransformer(
                    namespace=child_ns, stream_cls=AsyncChatModelStream
                ),
            ]
            for t in child_transformers:
                t.init()
                self._mux.register_transformer(t)

            yield AsyncSubgraphRunStream(
                mux=self._mux,
                namespace=child_ns,
                transformers=child_transformers,
            )


# ---------------------------------------------------------------------------
# AsyncGraphRunStream
# ---------------------------------------------------------------------------


class AsyncGraphRunStream:
    """The async run stream returned by ``StreamingHandler.astream()``.

    Async-iterable over all :class:`ProtocolEvent` instances.  Named
    projections provide ergonomic access to values, messages, subgraphs,
    and output.
    """

    def __init__(
        self,
        *,
        mux: AsyncStreamMux,
        namespace: list[str] | None = None,
        transformers: list[StreamTransformer],
        abort_event: asyncio.Event | None = None,
        output_mapper: Callable[[Any], Any] | None = None,
    ) -> None:
        self._mux = mux
        self._ns = namespace or []
        self._transformers = transformers
        self._abort_event = abort_event or asyncio.Event()
        self._output_mapper = output_mapper

    # -- Transformer lookup -------------------------------------------------

    def _find_transformer(self, name: str) -> StreamTransformer | None:
        for t in self._transformers:
            if getattr(t, "name", None) == name:
                return t
        return None

    # -- Raw event iteration ------------------------------------------------

    def __aiter__(self) -> AsyncIterator[ProtocolEvent]:
        return self._mux.subscribe_events(self._ns)

    # -- Named projections --------------------------------------------------

    @property
    def values(self) -> _ValuesProjection:
        """Async iterable of state snapshots; awaitable for final state."""
        t = self._find_transformer("values")
        return _ValuesProjection(self._mux, t, self._ns, self._output_mapper)

    @property
    def output(self) -> _ValuesProjection:
        """Awaitable for the final output state."""
        t = self._find_transformer("values")
        return _ValuesProjection(self._mux, t, self._ns, self._output_mapper)

    @property
    def messages(self) -> _MessagesProjection:
        """Async iterable of :class:`AsyncChatModelStream` instances."""
        t = self._find_transformer("messages")
        return _MessagesProjection(self._mux, t)

    def messages_from(self, node: str) -> _MessagesProjection:
        """Async iterable of messages from a specific node."""
        filtered = MessagesTransformer(
            namespace=self._ns,
            node_filter=node,
            stream_cls=AsyncChatModelStream,
        )
        self._mux.register_transformer(filtered)
        return _MessagesProjection(self._mux, filtered)

    @property
    def subgraphs(self) -> _SubgraphsProjection:
        """Async iterable of :class:`AsyncSubgraphRunStream` for child graphs."""
        return _SubgraphsProjection(self._mux, self._ns)

    # -- State --------------------------------------------------------------

    @property
    def interrupted(self) -> bool:
        return self._mux.interrupted

    @property
    def interrupts(self) -> list[InterruptPayload]:
        return self._mux.interrupts

    # -- Cancellation -------------------------------------------------------

    def abort(self, reason: str | None = None) -> None:
        """Signal cancellation of the run."""
        self._abort_event.set()

    @property
    def signal(self) -> asyncio.Event:
        """The underlying cancellation event."""
        return self._abort_event

    # -- Extensions ---------------------------------------------------------

    @property
    def extensions(self) -> dict[str, Any]:
        """All transformer projections."""
        result: dict[str, Any] = {}
        for t in self._transformers:
            name = getattr(t, "name", None)
            value = getattr(t, "value", None)
            if name is not None and value is not None:
                result[name] = value
        return result


# ---------------------------------------------------------------------------
# AsyncSubgraphRunStream
# ---------------------------------------------------------------------------


class AsyncSubgraphRunStream(AsyncGraphRunStream):
    """An :class:`AsyncGraphRunStream` for a child subgraph.

    Adds ``.name`` and ``.index`` parsed from the last namespace segment
    (e.g. ``"researcher:2"`` → ``name="researcher"``, ``index=2``).
    """

    @property
    def name(self) -> str:
        if self._ns:
            segment = self._ns[-1]
            return segment.split(":")[0] if ":" in segment else segment
        return ""

    @property
    def index(self) -> int:
        if self._ns:
            segment = self._ns[-1]
            if ":" in segment:
                try:
                    return int(segment.split(":")[-1])
                except ValueError:
                    pass
        return 0


# ---------------------------------------------------------------------------
# Async factory
# ---------------------------------------------------------------------------


async def create_async_graph_run_stream(
    source: AsyncIterator[tuple[tuple[str, ...], str, Any]],
    *,
    transformers: list[StreamTransformer] | None = None,
    abort_event: asyncio.Event | None = None,
    output_mapper: Callable[[Any], Any] | None = None,
) -> AsyncGraphRunStream:
    """Create an :class:`AsyncGraphRunStream` from a raw async stream source.

    1. Creates a :class:`StreamMux`
    2. Registers built-in ``ValuesTransformer`` and ``MessagesTransformer``
    3. Registers user-supplied transformers
    4. Creates the root ``AsyncGraphRunStream``
    5. Starts a background pump task that reads from *source*,
       converts each chunk to a ``ProtocolEvent``, and pushes it
       through the mux
    6. Returns the ``AsyncGraphRunStream``
    """
    abort = abort_event or asyncio.Event()

    # Built-in transformers first, then user-supplied
    all_transformers: list[StreamTransformer] = [
        ValuesTransformer(),
        MessagesTransformer(stream_cls=AsyncChatModelStream),
    ]
    all_transformers.extend(transformers or [])

    # Initialize transformers, collecting projections to wire after mux creation
    projections: list[Any] = []
    for t in all_transformers:
        projection = t.init()
        if projection is not None:
            projections.append(projection)

    mux = AsyncStreamMux(transformers=all_transformers)

    # Wire any StreamChannel instances found in transformer projections
    for projection in projections:
        mux.wire_channels(projection)

    # Create the root stream
    run_stream = AsyncGraphRunStream(
        mux=mux,
        transformers=all_transformers,
        abort_event=abort,
        output_mapper=output_mapper,
    )

    # Start the pump task
    async def pump() -> None:
        try:
            async for ns, mode, payload in source:
                if abort.is_set():
                    break
                # Extract node name embedded by StreamProtocolMessagesHandler.
                node: str | None = None
                if (
                    mode == "messages"
                    and isinstance(payload, dict)
                    and "__node__" in payload
                ):
                    payload = dict(payload)
                    node = payload.pop("__node__")
                event = convert_to_protocol_event(ns, mode, payload, node=node)
                if event is not None:
                    mux.push(event)
            mux.close()
        except Exception as exc:
            mux.fail(exc)

    asyncio.get_running_loop().create_task(pump())

    return run_stream


# ---------------------------------------------------------------------------
# GraphRunStream — returned by StreamingHandler.stream()
# ---------------------------------------------------------------------------


class _PumpDrivenLog:
    """Wraps a list so that iteration drives the sync pump.

    Used by all :class:`GraphRunStream` projections (``__iter__``,
    ``.values``, ``.messages``, ``.extensions``) so that iterating
    any projection lazily consumes the source.
    """

    __slots__ = ("_log", "_pump_one")

    def __init__(self, log: list, pump_one: Callable[[], bool]) -> None:
        self._log = log
        self._pump_one = pump_one

    def __iter__(self) -> Iterator[Any]:
        cursor = 0
        while True:
            if cursor < len(self._log):
                yield self._log[cursor]
                cursor += 1
            elif not self._pump_one():
                return

    def __len__(self) -> int:
        return len(self._log)

    def __getitem__(self, index: int) -> Any:
        return self._log[index]


class GraphRunStream:
    """Synchronous run stream returned by ``StreamingHandler.stream()``.

    All projections are blocking / sync-iterable.  Internally uses
    the same ``StreamMux`` and transformer pipeline, but without an
    async event loop.

    The source iterator is consumed lazily: each projection pulls
    events from the source on demand rather than eagerly buffering
    everything upfront.  This means callers see events as soon as
    they are produced by the underlying ``stream()`` call.
    """

    def __init__(
        self,
        *,
        mux: StreamMux,
        source: Iterator[tuple[tuple[str, ...], str, Any]],
        namespace: list[str] | None = None,
        transformers: list[StreamTransformer],
        output_mapper: Callable[[Any], Any] | None = None,
    ) -> None:
        self._mux = mux
        self._source = source
        self._source_exhausted = False
        self._ns = namespace or []
        self._transformers = transformers
        self._output_mapper = output_mapper

    # -- Transformer lookup -------------------------------------------------

    def _find_transformer(self, name: str) -> StreamTransformer | None:
        for t in self._transformers:
            if getattr(t, "name", None) == name:
                return t
        return None

    # -- Lazy pump ----------------------------------------------------------

    def _pump_one(self) -> bool:
        """Pull one item from the source, convert it, and push through the mux.

        Returns ``True`` if an item was consumed, ``False`` if the source
        is exhausted (or was already exhausted).
        """
        if self._source_exhausted:
            return False
        try:
            ns, mode, payload = next(self._source)
        except StopIteration:
            self._source_exhausted = True
            self._mux.close()
            return False
        except Exception as exc:
            self._source_exhausted = True
            self._mux.fail(exc)
            return False

        node: str | None = None
        if mode == "messages" and isinstance(payload, dict) and "__node__" in payload:
            payload = dict(payload)
            node = payload.pop("__node__")
        event = convert_to_protocol_event(ns, mode, payload, node=node)
        if event is not None:
            self._mux.push(event)
        return True

    def _pump_all(self) -> None:
        """Drain the source iterator completely."""
        while self._pump_one():
            pass

    # -- Helpers ------------------------------------------------------------

    def _map(self, value: Any) -> Any:
        if value is not None and self._output_mapper is not None:
            return self._output_mapper(value)
        return value

    # -- Raw event iteration (sync) -----------------------------------------

    def __iter__(self) -> Iterator[ProtocolEvent]:
        for event in _PumpDrivenLog(self._mux.event_log, self._pump_one):
            ns = event["params"].get("namespace", [])
            if not self._ns or ns[: len(self._ns)] == self._ns:
                yield event

    # -- Named projections (sync) -------------------------------------------

    @property
    def output(self) -> Any:
        """The final output state (blocking). Drains the source."""
        self._pump_all()
        return self._map(self._mux.get_latest_values(self._ns))

    @property
    def values(self) -> Iterator[Any]:
        """Sync iterable of intermediate state snapshots."""
        t = self._find_transformer("values")
        if t is None:
            return
        for item in _PumpDrivenLog(t.value, self._pump_one):
            if item.get("namespace", []) == self._ns:
                yield self._map(item["data"])

    @property
    def messages(self) -> Iterator[ChatModelStream]:
        """Sync iterable of :class:`ChatModelStream` instances.

        Each yielded ``ChatModelStream`` is fully populated (``done=True``)
        so that sync consumers can read ``.text``, ``.reasoning``, and
        ``.usage`` immediately.
        """
        t = self._find_transformer("messages")
        if t is None:
            return
        for msg in _PumpDrivenLog(t.value, self._pump_one):
            # Pump until this message is complete so sync consumers
            # get a fully populated ChatModelStream.
            while not msg.done:
                if not self._pump_one():
                    break
            yield msg

    # -- State --------------------------------------------------------------

    @property
    def interrupted(self) -> bool:
        return self._mux.interrupted

    @property
    def interrupts(self) -> list[InterruptPayload]:
        return self._mux.interrupts

    # -- Extensions ---------------------------------------------------------

    @property
    def extensions(self) -> dict[str, Any]:
        """All transformer projections as pump-driven iterables."""
        result: dict[str, Any] = {}
        for t in self._transformers:
            name = getattr(t, "name", None)
            value = getattr(t, "value", None)
            if name is not None and value is not None:
                if isinstance(value, list):
                    result[name] = _PumpDrivenLog(value, self._pump_one)
                else:
                    result[name] = value
        return result


def create_graph_run_stream(
    source: Iterator[tuple[tuple[str, ...], str, Any]],
    *,
    transformers: list[StreamTransformer] | None = None,
    output_mapper: Callable[[Any], Any] | None = None,
) -> GraphRunStream:
    """Create a :class:`GraphRunStream` from a sync stream source.

    The source iterator is stored on the returned stream and consumed
    lazily as projections are iterated.

    Built-in transformers (values, messages) are always registered first
    so that user-supplied transformers see events after built-in
    processing.
    """
    # Built-in transformers first, then user-supplied
    all_transformers: list[StreamTransformer] = [
        ValuesTransformer(),
        MessagesTransformer(),
    ]
    all_transformers.extend(transformers or [])

    projections: list[Any] = []
    for t in all_transformers:
        projection = t.init()
        if projection is not None:
            projections.append(projection)

    mux = StreamMux(transformers=all_transformers)

    for projection in projections:
        mux.wire_channels(projection)

    return GraphRunStream(
        mux=mux,
        source=source,
        transformers=all_transformers,
        output_mapper=output_mapper,
    )


__all__ = [
    "AsyncGraphRunStream",
    "AsyncSubgraphRunStream",
    "GraphRunStream",
    "create_async_graph_run_stream",
    "create_graph_run_stream",
]
