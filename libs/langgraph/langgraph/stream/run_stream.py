from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Iterator
from typing import Any

from langgraph.stream._convert import convert_to_protocol_event
from langgraph.stream._event_log import EventLog
from langgraph.stream._mux import StreamMux
from langgraph.stream._types import ProtocolEvent
from langgraph.stream.stream_channel import StreamChannel
from langgraph.stream.transformers import ValuesTransformer


class GraphRunStream:
    """Sync run stream with caller-driven pumping.

    The caller's iteration on any projection (``values``, ``messages``,
    raw events, or ``output``) drives the graph forward. No background
    thread is used — this matches v1's model where the caller's ``for``
    loop is the pump.

    All transformer projections live in ``extensions``. Native transformer
    projections (those with ``_native = True``) are also set as direct
    attributes on this instance (e.g. ``run.values``, ``run.messages``).

    Iterating the run stream directly yields raw ``ProtocolEvent`` objects
    from the mux's main event log.
    """

    def __init__(
        self,
        graph_iter: Iterator[Any],
        mux: StreamMux,
        extensions: dict[str, Any],
        values_transformer: ValuesTransformer,
    ) -> None:
        self._graph_iter = graph_iter
        self._mux = mux
        self.extensions = extensions
        self._values_transformer = values_transformer
        self._exhausted = False
        # Wire pull-based iteration: every sync EventLog calls _pump_next
        # when its cursor catches up to the buffer.
        self._wire_request_more(mux, extensions)

    def _wire_request_more(self, mux: StreamMux, extensions: dict[str, Any]) -> None:
        """Set _request_more on all sync EventLogs so iteration drives the graph."""
        mux._events._request_more = self._pump_next
        for value in extensions.values():
            if isinstance(value, EventLog):
                value._request_more = self._pump_next
            elif isinstance(value, StreamChannel):
                value._log._request_more = self._pump_next

    def _pump_next(self) -> bool:
        """Pull one event from the graph and push through the mux.

        Returns True if an event was pulled, False if the graph is exhausted.
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
        if self._values_transformer._log._error is not None:
            raise self._values_transformer._log._error
        return self._values_transformer._latest

    @property
    def interrupted(self) -> bool:
        """Block until the run completes, then return whether it was interrupted."""
        self._pump_all()
        return self._values_transformer._interrupted

    @property
    def interrupts(self) -> list[Any]:
        """Block until the run completes, then return interrupt payloads."""
        self._pump_all()
        return self._values_transformer._interrupts

    def __iter__(self) -> Iterator[ProtocolEvent]:
        """Iterate all protocol events from the mux's main event log."""
        return iter(self._mux._events)


class AsyncGraphRunStream:
    """Async run stream with transformer-driven projections.

    A background asyncio task pumps events from the graph into the
    transformer pipeline. This is the standard async pattern — the task
    runs on the same event loop and async consumers can iterate multiple
    projections concurrently.

    All transformer projections live in ``extensions``. Native transformer
    projections (those with ``_native = True``) are also set as direct
    attributes on this instance (e.g. ``run.values``, ``run.messages``).

    Async-iterating the run stream yields raw ``ProtocolEvent`` objects
    from the mux's main event log.
    """

    def __init__(
        self,
        mux: StreamMux,
        extensions: dict[str, Any],
        values_transformer: ValuesTransformer,
        pump_task: asyncio.Task[None],
    ) -> None:
        self._mux = mux
        self.extensions = extensions
        self._values_transformer = values_transformer
        self._pump_task = pump_task

    @property
    def output(self) -> Any:
        """Return an awaitable that resolves to the final state.

        Usage::

            output = await run.output
        """
        return self._get_output()

    async def _get_output(self) -> dict[str, Any] | None:
        try:
            await self._pump_task
        except BaseException:
            pass
        if self._values_transformer._log._error is not None:
            raise self._values_transformer._log._error
        return self._values_transformer._latest

    @property
    def interrupted(self) -> bool:
        """Whether the run was interrupted.

        Only meaningful after the run has completed (after consuming the
        stream or awaiting ``output``).
        """
        return self._values_transformer._interrupted

    @property
    def interrupts(self) -> list[Any]:
        """Interrupt payloads, populated when interrupted is True."""
        return self._values_transformer._interrupts

    def __aiter__(self) -> AsyncIterator[ProtocolEvent]:
        """Iterate all protocol events from the mux's main event log."""
        return self._mux._events.__aiter__()
