from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Iterator, Mapping
from types import MappingProxyType
from typing import Any

from langgraph.stream._convert import convert_to_protocol_event
from langgraph.stream._event_log import EventLog
from langgraph.stream._mux import StreamMux
from langgraph.stream._types import ProtocolEvent
from langgraph.stream.stream_channel import StreamChannel
from langgraph.stream.transformers import ValuesTransformer


class GraphRunStream:
    """Sync run stream with caller-driven pumping.

    The caller's iteration on any projection (`values`, `messages`,
    raw events, or `output`) drives the graph forward. No background
    thread is used — this matches v1's model where the caller's `for`
    loop is the pump.

    All transformer projections live in `extensions`. Native transformer
    projections (those with `_native = True`) are also set as direct
    attributes on this instance (e.g. `run.values`, `run.messages`).

    Iterating the run stream directly yields raw `ProtocolEvent` objects
    from the mux's main event log.
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
        self._graph_iter = graph_iter
        self._mux = mux
        self.extensions: Mapping[str, Any] = MappingProxyType(mux.extensions)
        self._values_transformer = values_transformer
        self._exhausted = False
        # Native-transformer projections also show up as direct attributes.
        for key in mux.native_keys:
            setattr(self, key, mux.extensions[key])
        # Wire pull-based iteration: every sync EventLog calls _pump_next
        # when its cursor catches up to the buffer.
        self._wire_request_more(mux)

    def _wire_request_more(self, mux: StreamMux) -> None:
        """Install `_request_more` on every sync EventLog.

        Sync iteration is caller-driven, so a cursor that catches up to
        the buffer's tail needs a way to ask the graph for more events.
        """
        mux._events._request_more = self._pump_next
        for value in mux.extensions.values():
            if isinstance(value, EventLog):
                value._request_more = self._pump_next
            elif isinstance(value, StreamChannel):
                value._log._request_more = self._pump_next

    def _pump_next(self) -> bool:
        """Pull one event from the graph and push it through the mux.

        Returns:
            True if an event was pulled, False if the graph is exhausted
            or has raised.
        """
        if self._exhausted:
            return False
        try:
            part = next(self._graph_iter)
        except StopIteration:
            self._mux.close()
            self._exhausted = True
            return False
        except BaseException as e:
            self._mux.fail(e)
            self._exhausted = True
            return False
        self._mux.push(convert_to_protocol_event(part))
        return True

    def _pump_all(self) -> None:
        """Drain the graph completely."""
        while self._pump_next():
            pass

    @property
    def output(self) -> dict[str, Any] | None:
        """Block until the run completes and return the final state."""
        self._pump_all()
        err = self._values_transformer.error
        if err is not None:
            raise err
        return self._values_transformer._latest

    @property
    def interrupted(self) -> bool:
        """Block until the run completes, then return whether it was interrupted.

        Raises:
            BaseException: If the run ended with an error.
        """
        self._pump_all()
        err = self._values_transformer.error
        if err is not None:
            raise err
        return self._values_transformer._interrupted

    @property
    def interrupts(self) -> list[Any]:
        """Block until the run completes, then return interrupt payloads.

        Raises:
            BaseException: If the run ended with an error.
        """
        self._pump_all()
        err = self._values_transformer.error
        if err is not None:
            raise err
        return self._values_transformer._interrupts

    def __iter__(self) -> Iterator[ProtocolEvent]:
        """Iterate all protocol events from the mux's main event log."""
        return iter(self._mux._events)


class AsyncGraphRunStream:
    """Async run stream with transformer-driven projections.

    A background asyncio task pumps events from the graph into the
    transformer pipeline. This is the standard async pattern — the task
    runs on the same event loop and async consumers can iterate
    multiple projections concurrently.

    All transformer projections live in `extensions`. Native transformer
    projections (those with `_native = True`) are also set as direct
    attributes on this instance (e.g. `run.values`, `run.messages`).

    Async-iterating the run stream yields raw `ProtocolEvent` objects
    from the mux's main event log.
    """

    def __init__(
        self,
        mux: StreamMux,
        values_transformer: ValuesTransformer,
        pump_task: asyncio.Task[None],
    ) -> None:
        """Initialize the async run stream.

        Args:
            mux: The StreamMux owning projections and the main log.
            values_transformer: The built-in values transformer
                providing `output` / `interrupted` / `interrupts`.
            pump_task: Background task pumping graph events into the mux.
        """
        self._mux = mux
        self.extensions: Mapping[str, Any] = MappingProxyType(mux.extensions)
        self._values_transformer = values_transformer
        self._pump_task = pump_task
        # Native-transformer projections also show up as direct attributes.
        for key in mux.native_keys:
            setattr(self, key, mux.extensions[key])

    async def output(self) -> dict[str, Any] | None:
        """Wait for the run to complete and return the final state.

        Methods (not properties) on the async lane so `run.output` without
        `await` raises at type-check time instead of silently yielding a
        coroutine object that's truthy, lenless, and never awaited.

        The pump routes any Exception into `mux.afail`, which surfaces on
        `ValuesTransformer.error`. CancelledError / KeyboardInterrupt
        propagate so cancellation isn't silently dropped.

        Example:
            ```python
            output = await run.output()
            ```

        Raises:
            BaseException: If the run ended with an error.
        """
        try:
            await self._pump_task
        except Exception:
            pass
        if (err := self._values_transformer.error) is not None:
            raise err
        return self._values_transformer._latest

    async def interrupted(self) -> bool:
        """Wait for the run to complete and return whether it was interrupted.

        Example:
            ```python
            interrupted = await run.interrupted()
            ```

        Raises:
            BaseException: If the run ended with an error.
        """
        try:
            await self._pump_task
        except Exception:
            pass
        if (err := self._values_transformer.error) is not None:
            raise err
        return self._values_transformer._interrupted

    async def interrupts(self) -> list[Any]:
        """Wait for the run to complete and return interrupt payloads.

        Example:
            ```python
            interrupts = await run.interrupts()
            ```

        Raises:
            BaseException: If the run ended with an error.
        """
        try:
            await self._pump_task
        except Exception:
            pass
        if (err := self._values_transformer.error) is not None:
            raise err
        return self._values_transformer._interrupts

    def __aiter__(self) -> AsyncIterator[ProtocolEvent]:
        """Iterate all protocol events from the mux's main event log."""
        return self._mux._events.__aiter__()
