from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Awaitable, Callable, Iterator, Mapping
from types import MappingProxyType, TracebackType
from typing import TYPE_CHECKING, Any

from langgraph.stream._convert import convert_to_protocol_event
from langgraph.stream._mux import StreamMux
from langgraph.stream._types import ProtocolEvent

if TYPE_CHECKING:
    from langgraph.stream.transformers import ValuesTransformer


def _drive_until_done(pump: Callable[[], bool]) -> None:
    """Call the sync pump until it returns False."""
    while pump():
        pass


async def _adrive_until_done(pump: Callable[[], Awaitable[bool]]) -> None:
    """Call the async pump until it returns False."""
    while await pump():
        pass


class BaseRunStream:
    """Shared shape for any object that wraps a `StreamMux`.

    Root (`GraphRunStream` / `AsyncGraphRunStream`) and scoped
    (`SubgraphRunStream`) streams both compose a `StreamMux`. The mux
    owns the projections — `values`, `messages`, `subgraphs`, and any
    user-registered keys — all exposed via `extensions`. Native
    projections (`_native = True`) are also bound as direct attributes
    (`run.values`, `run.messages`, …) for ergonomics.

    Raw iteration (`for event in run` / `async for event in run`) and
    the `interleave(...)` helper both live here so every subclass
    behaves consistently. Subclasses only add pump ownership, scope
    metadata, or sync/async flavor.
    """

    def __init__(self, mux: StreamMux) -> None:
        self._mux = mux
        self.extensions: Mapping[str, Any] = MappingProxyType(mux.extensions)
        for key in mux.native_keys:
            setattr(self, key, mux.extensions[key])

    def __iter__(self) -> Iterator[ProtocolEvent]:
        """Sync iteration of protocol events on this mux's main log.

        Raises at the EventLog level if the mux is async-bound.
        """
        return iter(self._mux._events)

    def __aiter__(self) -> AsyncIterator[ProtocolEvent]:
        """Async iteration of protocol events on this mux's main log.

        Raises at the EventLog level if the mux is sync-bound.
        """
        return self._mux._events.__aiter__()

    def interleave(self, *names: str) -> Iterator[tuple[str, Any]]:
        """Iterate multiple projections round-robin, yielding ``(name, item)``.

        Each turn advances one projection's cursor; when a cursor's
        buffer is empty, pulling from it drives the pump once, which
        fans out to every subscribed projection log. Projections whose
        items aren't consumed on this turn sit in their own buffers
        only until the next turn reaches them, bounding memory by the
        skew between projection rates rather than letting any single
        log grow to the full run length.

        Projections are exhausted independently; a projection that
        finishes early drops out of the rotation while others
        continue. The overall iterator ends once all named projections
        are done.

        Args:
            *names: Projection keys to interleave. Must match keys in
                `extensions`.

        Yields:
            `(name, item)` tuples in round-robin order across the named
            projections.

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
        cursors: dict[str, Iterator[Any]] = {
            name: iter(self.extensions[name]) for name in names
        }
        done: set[str] = set()
        while len(done) < len(cursors):
            for name, cursor in cursors.items():
                if name in done:
                    continue
                try:
                    item = next(cursor)
                except StopIteration:
                    done.add(name)
                    continue
                yield (name, item)


class GraphRunStream(BaseRunStream):
    """Sync run stream with caller-driven pumping.

    The caller's iteration on any projection (`values`, `messages`,
    raw events, or `output`) drives the graph forward. No background
    thread is used — the caller's `for` loop is the pump.

    Projections are single-consumer — iterating `run.values` twice
    raises. Use `projection.tee(n)` if you genuinely need fan-out.
    """

    def __init__(
        self,
        graph_iter: Iterator[Any],
        mux: StreamMux,
        values_transformer: ValuesTransformer,
    ) -> None:
        """Initialize the run stream.

        Args:
            graph_iter: Pull-based iterator over the graph's stream.
            mux: The StreamMux owning projections and the main log.
            values_transformer: The built-in values transformer
                providing `output` / `interrupted` / `interrupts`.
        """
        super().__init__(mux)
        self._graph_iter = graph_iter
        self._values_transformer = values_transformer
        self._exhausted = False
        mux.bind_pump(self._pump_next)

    def _pump_next(self) -> bool:
        """Pull one event from the graph and push it through the mux.

        Returns:
            True if an event was pulled, False if the graph is
            exhausted or has raised.
        """
        if self._exhausted:
            return False
        try:
            part = next(self._graph_iter)
        except StopIteration:
            self._mux.close()
            self._exhausted = True
            return False
        except Exception as e:
            self._mux.fail(e)
            self._exhausted = True
            return False
        self._mux.push(convert_to_protocol_event(part))
        return True

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
        err = self._values_transformer.error
        if err is not None:
            raise err
        return self._values_transformer._latest

    @property
    def interrupted(self) -> bool:
        """Drive the run to completion, then return whether it was interrupted.

        Raises:
            BaseException: If the run ended with an error.
        """
        _drive_until_done(self._pump_next)
        err = self._values_transformer.error
        if err is not None:
            raise err
        return self._values_transformer._interrupted

    @property
    def interrupts(self) -> list[Any]:
        """Drive the run to completion, then return interrupt payloads.

        Raises:
            BaseException: If the run ended with an error.
        """
        _drive_until_done(self._pump_next)
        err = self._values_transformer.error
        if err is not None:
            raise err
        return self._values_transformer._interrupts


class AsyncGraphRunStream(BaseRunStream):
    """Async run stream with caller-driven pumping.

    Async iteration on any projection drives the graph forward — there
    is no background task. Concurrent consumers share a single-flight
    pump via an `asyncio.Lock`, so each awaiting cursor contributes
    one event per acquisition. Backpressure comes from the logs: when
    a subscribed log's buffer reaches `maxlen`, `apush` awaits the
    subscriber to drain, which holds back the pump and paces the
    graph.

    Projections are single-consumer — a second `aiter(run.values)`
    raises. Use `projection.tee(n)` for fan-out.

    Use as an async context manager to guarantee clean shutdown on
    early exit:

    ```python
    async with await handler.astream(input) as run:
        async for msg in run.messages:
            ...
    ```
    """

    def __init__(
        self,
        graph_aiter: AsyncIterator[Any],
        mux: StreamMux,
        values_transformer: ValuesTransformer,
    ) -> None:
        """Initialize the async run stream.

        Args:
            graph_aiter: Async iterator over the graph's stream.
            mux: The StreamMux owning projections and the main log.
            values_transformer: The built-in values transformer
                providing `output` / `interrupted` / `interrupts`.
        """
        super().__init__(mux)
        self._graph_aiter = graph_aiter
        self._values_transformer = values_transformer
        self._exhausted = False
        self._pump_lock = asyncio.Lock()
        mux.bind_apump(self._apump_next)

    async def _apump_next(self) -> bool:
        """Pull one event from the graph and push it through the mux.

        Serialized via `self._pump_lock` so concurrent cursors each
        produce one event per acquisition rather than racing on the
        graph iterator.

        `except Exception` is intentional — `CancelledError` and other
        `BaseException` subclasses propagate, matching asyncio's
        cancellation contract.

        Returns:
            True if an event was pulled, False if the graph is
            exhausted or has raised.
        """
        async with self._pump_lock:
            if self._exhausted:
                return False
            try:
                part = await self._graph_aiter.__anext__()
            except StopAsyncIteration:
                self._exhausted = True
                await self._mux.aclose()
                return False
            except Exception as e:
                self._exhausted = True
                await self._mux.afail(e)
                return False
            await self._mux.apush(convert_to_protocol_event(part))
            return True

    async def abort(self) -> None:
        """Stop the run early.

        Closes the mux and marks the stream exhausted. Any awaiting
        cursors wake up and see the closed state; any `apush` blocked
        on backpressure wakes and returns without appending.
        Idempotent.
        """
        async with self._pump_lock:
            if self._exhausted:
                return
            self._exhausted = True
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
        if (err := self._values_transformer.error) is not None:
            raise err
        return self._values_transformer._latest

    async def interrupted(self) -> bool:
        """Drive the run to completion and return whether it was interrupted.

        Raises:
            BaseException: If the run ended with an error.
        """
        await _adrive_until_done(self._apump_next)
        if (err := self._values_transformer.error) is not None:
            raise err
        return self._values_transformer._interrupted

    async def interrupts(self) -> list[Any]:
        """Drive the run to completion and return interrupt payloads.

        Raises:
            BaseException: If the run ended with an error.
        """
        await _adrive_until_done(self._apump_next)
        if (err := self._values_transformer.error) is not None:
            raise err
        return self._values_transformer._interrupts
