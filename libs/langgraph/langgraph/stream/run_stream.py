from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Awaitable, Callable, Iterator, Mapping
from types import MappingProxyType, TracebackType
from typing import TYPE_CHECKING, Any

from langchain_core._api import beta

from langgraph.stream._convert import convert_to_protocol_event
from langgraph.stream._mux import StreamMux
from langgraph.stream._types import ProtocolEvent

if TYPE_CHECKING:
    from langgraph.stream.transformers import SubgraphStatus


def _drive_until_done(pump: Callable[[], bool]) -> None:
    """Call the sync pump until it returns False."""
    while pump():
        pass


async def _adrive_until_done(pump: Callable[[], Awaitable[bool]]) -> None:
    """Call the async pump until it returns False."""
    while await pump():
        pass


@beta(message="The v3 streaming protocol on Pregel is experimental.")
class GraphRunStream:
    """Sync run stream with caller-driven pumping.

    The caller's iteration on any projection (`values`, `messages`,
    raw events, or `output`) drives the graph forward. No background
    thread is used — the caller's `for` loop is the pump.

    Projections are single-consumer — iterating `run.values` twice
    raises. Use `projection.tee(n)` if you genuinely need fan-out.

    All transformer projections live in `extensions`. Native transformer
    projections (those with `_native = True`) are also set as direct
    attributes on this instance (e.g. `run.values`, `run.messages`).

    !!! warning

        Returned by `Pregel.stream_events(version="v3")`, which is
        experimental and may change.
    """

    def __init__(
        self,
        graph_iter: Iterator[Any] | None,
        mux: StreamMux,
        *,
        wire_pump: bool = True,
    ) -> None:
        """Initialize the run stream.

        Args:
            graph_iter: Pull-based iterator over the graph's stream,
                or `None` for nested run streams whose pump is driven
                by an outer run (e.g. `SubgraphRunStream`).
            mux: The StreamMux owning projections and the main log.
            wire_pump: When True (default), bind `_pump_next` as the
                mux's pump callable. Subclasses that inherit a parent
                pump via `StreamMux._make_child` should pass False to
                preserve the parent binding.
        """
        self._graph_iter = graph_iter
        self._mux = mux
        self.extensions: Mapping[str, Any] = MappingProxyType(mux.extensions)
        self._exhausted = False
        self._latest: dict[str, Any] | None = None
        self._interrupted = False
        self._interrupts: list[Any] = []
        self._scope_list: list[str] = list(mux.scope)
        for key in mux.native_keys:
            setattr(self, key, mux.extensions[key])
        if wire_pump:
            self._wire_request_more(mux)

    def _wire_request_more(self, mux: StreamMux) -> None:
        """Wire the sync pull callback through the mux.

        Routing through `mux.bind_pump` (rather than walking
        projections directly here) lets child mini-muxes built by
        `mux._make_child(...)` inherit the same pump callable, so
        cursors on a subgraph handle's projections drive the root
        pump just like cursors on `run.values` do.
        """
        mux.bind_pump(self._pump_next)

    def _observe_event(self, event: ProtocolEvent) -> None:
        """Track values-event state for output/interrupted/interrupts."""
        if event["method"] != "values":
            return
        params = event["params"]
        if params["namespace"] != self._scope_list:
            return
        self._latest = params["data"]
        interrupts = params.get("interrupts", ())
        if interrupts:
            self._interrupted = True
            self._interrupts.extend(interrupts)

    def _pump_next(self) -> bool:
        """Pull one event from the graph and push it through the mux.

        Returns:
            True if an event was pulled, False if the graph is exhausted
            or has raised. Always False when constructed with
            `graph_iter=None` (the run is driven by an outer pump).
        """
        if self._exhausted or self._graph_iter is None:
            return False
        try:
            part = next(self._graph_iter)
            event = convert_to_protocol_event(part)
            self._observe_event(event)
            self._mux.push(event)
            return True
        except StopIteration:
            self._mux.close()
            self._exhausted = True
            return False
        except Exception as e:
            self._mux.fail(e)
            self._exhausted = True
            return False

    def abort(self) -> None:
        """Stop the run early.

        Closes the mux and marks the stream exhausted. The graph
        iterator is dropped; any in-flight nodes see the closure on
        their next yield point. Idempotent.
        """
        if self._exhausted:
            return
        self._exhausted = True
        try:
            self._mux.close()
        except Exception:
            pass

    def __enter__(self) -> GraphRunStream:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.abort()

    @property
    def output(self) -> dict[str, Any] | None:
        """Drive the run to completion and return the final state."""
        _drive_until_done(self._pump_next)
        if (err := self._mux._events._error) is not None:
            raise err
        return self._latest

    @property
    def interrupted(self) -> bool:
        """Drive the run to completion, then return whether it was
        interrupted.

        Raises:
            BaseException: If the run ended with an error.
        """
        _drive_until_done(self._pump_next)
        if (err := self._mux._events._error) is not None:
            raise err
        return self._interrupted

    @property
    def interrupts(self) -> list[Any]:
        """Drive the run to completion, then return interrupt payloads.

        Raises:
            BaseException: If the run ended with an error.
        """
        _drive_until_done(self._pump_next)
        if (err := self._mux._events._error) is not None:
            raise err
        return self._interrupts

    def __iter__(self) -> Iterator[ProtocolEvent]:
        """Subscribe to the main event log and iterate protocol events."""
        return iter(self._mux._events)

    def interleave(self, *names: str) -> Iterator[tuple[str, Any]]:
        """Iterate multiple projections in arrival order, yielding ``(name, item)``.

        Items are ordered by a monotonic push stamp assigned when each
        transformer pushes into its `StreamChannel`. This gives strict
        arrival ordering across projections, unlike round-robin.

        Args:
            *names: Projection keys to interleave. Must match keys in
                ``extensions``.

        Yields:
            ``(name, item)`` tuples in arrival order across the named
            projections.

        Each named channel is locked for the duration of iteration and
        released when the generator completes, is closed, or raises.
        Channels cannot be subscribed concurrently — use `.tee(n)` if
        you need fan-out.

        Raises:
            KeyError: If a name doesn't match a registered projection.

        Example:
            ```python
            for name, item in run.interleave("messages", "values"):
                if name == "messages":
                    print("msg:", item)
                else:
                    print("val:", item)
            ```
        """
        from langgraph.stream.stream_channel import StreamChannel

        channels: dict[str, StreamChannel[Any]] = {}
        try:
            for name in names:
                ch = self.extensions[name]
                if not isinstance(ch, StreamChannel):
                    raise TypeError(
                        f"interleave() requires StreamChannel projections, "
                        f"got {type(ch).__name__} for {name!r}"
                    )
                if ch._is_async is None:
                    raise TypeError(
                        f"StreamChannel {name!r} has not been bound yet. "
                        "Register the transformer with a StreamMux first."
                    )
                if ch._is_async:
                    raise TypeError(
                        f"StreamChannel {name!r} is bound to async mode — "
                        "sync interleave() cannot consume async channels."
                    )
                if ch._subscribed:
                    raise RuntimeError(
                        f"StreamChannel {name!r} already has a subscriber; "
                        "use .tee(n) for fan-out."
                    )
                ch._subscribed = True
                channels[name] = ch

            done: set[str] = set()

            while len(done) < len(channels):
                best: tuple[int, str] | None = None
                for name, ch in channels.items():
                    if name in done:
                        continue
                    if ch._closed and not ch._items:
                        if ch._error is not None:
                            raise ch._error
                        done.add(name)
                        continue
                    if ch._items:
                        stamp = ch._items[0][0]
                        if best is None or stamp < best[0]:
                            best = (stamp, name)

                if best is not None:
                    _stamp, item = channels[best[1]]._items.popleft()
                    yield (best[1], item)
                else:
                    pump = self._mux._pump_fn
                    if pump is None or not pump():
                        before = len(done)
                        for name, ch in channels.items():
                            if name not in done and not ch._items:
                                if ch._closed:
                                    if ch._error is not None:
                                        raise ch._error
                                    done.add(name)
                        if len(done) == before:
                            break
        finally:
            for ch in channels.values():
                ch._subscribed = False


@beta(message="The v3 streaming protocol on Pregel is experimental.")
class AsyncGraphRunStream:
    """Async run stream with caller-driven pumping.

    Async iteration on any projection drives the graph forward — there
    is no background task. Concurrent consumers share a single-flight
    pump via an `asyncio.Lock`, so each awaiting cursor contributes one
    event per acquisition. Backpressure comes from the logs: when a
    subscribed log's buffer reaches `maxlen`, `apush` awaits the
    subscriber to drain, which holds back the pump and paces the graph.

    Projections are single-consumer — a second `aiter(run.values)`
    raises. Use `projection.tee(n)` for fan-out.

    Use as an async context manager to guarantee clean shutdown on
    early exit:

    ```python
    async with await handler.astream(input) as run:
        async for msg in run.messages:
            ...
    ```

    !!! warning

        Awaited from `Pregel.astream_events(version="v3")`, which is
        experimental and may change.
    """

    def __init__(
        self,
        graph_aiter: AsyncIterator[Any] | None,
        mux: StreamMux,
        *,
        wire_pump: bool = True,
    ) -> None:
        """Initialize the async run stream.

        Args:
            graph_aiter: Async iterator over the graph's stream, or
                `None` for nested run streams whose pump is driven by
                an outer run (e.g. `AsyncSubgraphRunStream`).
            mux: The StreamMux owning projections and the main log.
            wire_pump: When True (default), bind `_apump_next` as the
                mux's async pump callable. Subclasses that inherit a
                parent pump via `StreamMux._make_child` should pass
                False to preserve the parent binding.
        """
        self._graph_aiter = graph_aiter
        self._mux = mux
        self.extensions: Mapping[str, Any] = MappingProxyType(mux.extensions)
        self._exhausted = False
        self._latest: dict[str, Any] | None = None
        self._interrupted = False
        self._interrupts: list[Any] = []
        self._scope_list: list[str] = list(mux.scope)
        self._pump_cond = asyncio.Condition()
        self._pumping = False
        for key in mux.native_keys:
            setattr(self, key, mux.extensions[key])
        if wire_pump:
            self._wire_arequest_more(mux)

    def _observe_event(self, event: ProtocolEvent) -> None:
        """Track values-event state for output/interrupted/interrupts."""
        if event["method"] != "values":
            return
        params = event["params"]
        if params["namespace"] != self._scope_list:
            return
        self._latest = params["data"]
        interrupts = params.get("interrupts", ())
        if interrupts:
            self._interrupted = True
            self._interrupts.extend(interrupts)

    def _wire_arequest_more(self, mux: StreamMux) -> None:
        """Wire the async pull callback through the mux.

        Mirrors `_wire_request_more`: routing through
        `mux.bind_apump` lets child mini-muxes inherit the pump
        callable so cursors on subgraph handles drive the root
        pump.
        """
        mux.bind_apump(self._apump_next)

    async def _apump_next(self) -> bool:
        """Drive one pump step, or wait for the active pumper to drive one.

        "Take-a-number" semantics: at most one task at a time calls
        `graph_aiter.__anext__()` (asyncio iterators can't be advanced
        concurrently). Other callers wait on a Condition that the
        active pumper notifies after each step. This lets a "passive"
        consumer — one whose projection's buffer is being filled by the
        active pumper's push — wake up as soon as its data lands,
        instead of queueing on the pump and only observing its data one
        graph event late.

        `except Exception` is intentional — `CancelledError` and other
        `BaseException` subclasses propagate, matching asyncio's
        cancellation contract.

        Returns:
            True if a pump step completed (by this task or another),
            False if the graph is exhausted.
        """
        async with self._pump_cond:
            if self._exhausted or self._graph_aiter is None:
                return False
            if self._pumping:
                # Another task is pumping; wait for its progress signal.
                await self._pump_cond.wait()
                return not self._exhausted
            self._pumping = True

        try:
            try:
                part = await self._graph_aiter.__anext__()
                event = convert_to_protocol_event(part)
                self._observe_event(event)
                await self._mux.apush(event)
                return True
            except StopAsyncIteration:
                self._exhausted = True
                await self._mux.aclose()
                return False
            except Exception as e:
                self._exhausted = True
                await self._mux.afail(e)
                return False
        finally:
            async with self._pump_cond:
                self._pumping = False
                self._pump_cond.notify_all()

    async def abort(self) -> None:
        """Stop the run early.

        Marks the stream exhausted, wakes any pump-waiters, and closes
        the mux. Any `apush` blocked on backpressure wakes and returns
        without appending. Idempotent.
        """
        async with self._pump_cond:
            if self._exhausted:
                return
            self._exhausted = True
            self._pump_cond.notify_all()
        try:
            await self._mux.aclose()
        except Exception:
            pass

    async def __aenter__(self) -> AsyncGraphRunStream:
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        await self.abort()

    async def output(self) -> dict[str, Any] | None:
        """Drive the run to completion and return the final state.

        Methods (not properties) on the async lane so `run.output`
        without `await` raises at type-check time instead of silently
        yielding a coroutine object.

        Example:
            ```python
            output = await run.output()
            ```

        Raises:
            BaseException: If the run ended with an error.
        """
        await _adrive_until_done(self._apump_next)
        if (err := self._mux._events._error) is not None:
            raise err
        return self._latest

    async def interrupted(self) -> bool:
        """Drive the run to completion and return whether it was
        interrupted.

        Raises:
            BaseException: If the run ended with an error.
        """
        await _adrive_until_done(self._apump_next)
        if (err := self._mux._events._error) is not None:
            raise err
        return self._interrupted

    async def interrupts(self) -> list[Any]:
        """Drive the run to completion and return interrupt payloads.

        Raises:
            BaseException: If the run ended with an error.
        """
        await _adrive_until_done(self._apump_next)
        if (err := self._mux._events._error) is not None:
            raise err
        return self._interrupts

    def __aiter__(self) -> AsyncIterator[ProtocolEvent]:
        """Subscribe to the main event log and iterate protocol events."""
        return self._mux._events.__aiter__()


class _SubgraphRunStreamMixin:
    """Subgraph metadata + parent-pump delegation shared by both lanes.

    Inherits from `GraphRunStream` (or `AsyncGraphRunStream`) with
    `graph_iter=None` + `wire_pump=False` — the mini-mux is driven
    by the parent's pump (inherited via `StreamMux._make_child`), and
    the handle never pulls upstream itself. Pump-driving methods
    delegate to the parent pump so `handle.output` and friends drive
    the root run.

    Subclasses set the parent pump function captured at construction
    (`_parent_pump_fn` / `_parent_apump_fn`) and override
    `_pump_next` / `_apump_next` to delegate to it.

    Status is updated in place by `SubgraphTransformer`. Iterate
    `run.subgraphs` to receive handles as subgraphs spawn, then
    drill into projections inside the loop body **before** the next
    pump cycle — same lazy-subscribe constraint as root projections.
    """

    path: tuple[str, ...]
    graph_name: str | None
    trigger_call_id: str | None
    status: SubgraphStatus
    error: str | None
    _seen_terminal: bool


class SubgraphRunStream(GraphRunStream, _SubgraphRunStreamMixin):
    """Sync handle for a discovered subgraph (extends `GraphRunStream`)."""

    def __init__(
        self,
        mux: StreamMux,
        *,
        path: tuple[str, ...],
        graph_name: str | None = None,
        trigger_call_id: str | None = None,
    ) -> None:
        # Capture the parent-inherited pump before super().__init__
        # touches anything; we delegate to it from `_pump_next`.
        self._parent_pump_fn: Callable[[], bool] | None = mux._pump_fn
        super().__init__(
            graph_iter=None,
            mux=mux,
            wire_pump=False,
        )
        self.path = path
        self.graph_name = graph_name
        self.trigger_call_id = trigger_call_id
        self.status = "started"
        self.error = None
        self._seen_terminal = False

    def _pump_next(self) -> bool:
        """Delegate to the parent's pump.

        Cursors on this handle's projections call here when their
        buffers empty. Driving the parent fans events into our
        mini-mux, transparently advancing the whole run.
        """
        if (
            self._exhausted
            or self._seen_terminal
            or self._mux._events._closed
            or self._parent_pump_fn is None
        ):
            return False
        return self._parent_pump_fn()


class AsyncSubgraphRunStream(AsyncGraphRunStream, _SubgraphRunStreamMixin):
    """Async handle for a discovered subgraph (extends `AsyncGraphRunStream`)."""

    def __init__(
        self,
        mux: StreamMux,
        *,
        path: tuple[str, ...],
        graph_name: str | None = None,
        trigger_call_id: str | None = None,
    ) -> None:
        self._parent_apump_fn: Callable[[], Awaitable[bool]] | None = mux._apump_fn
        super().__init__(
            graph_aiter=None,
            mux=mux,
            wire_pump=False,
        )
        self.path = path
        self.graph_name = graph_name
        self.trigger_call_id = trigger_call_id
        self.status = "started"
        self.error = None
        self._seen_terminal = False

    async def _apump_next(self) -> bool:
        """Delegate to the parent's async pump."""
        if (
            self._exhausted
            or self._seen_terminal
            or self._mux._events._closed
            or self._parent_apump_fn is None
        ):
            return False
        return await self._parent_apump_fn()
