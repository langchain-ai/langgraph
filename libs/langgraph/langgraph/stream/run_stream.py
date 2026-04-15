from __future__ import annotations

import asyncio
import threading
from collections.abc import AsyncIterator, Iterator
from typing import Any

from langgraph.stream._mux import StreamMux
from langgraph.stream._types import ProtocolEvent
from langgraph.stream.transformers import ValuesTransformer


class GraphRunStream:
    """Sync run stream with transformer-driven projections.

    All transformer projections live in ``extensions``. Native transformer
    projections (those with ``_native = True``) are also set as direct
    attributes on this instance (e.g. ``run.values``, ``run.messages``).

    Iterating the run stream directly yields raw ``ProtocolEvent`` objects
    from the mux's main event log.
    """

    def __init__(
        self,
        mux: StreamMux,
        extensions: dict[str, Any],
        values_transformer: ValuesTransformer,
        pump_thread: threading.Thread,
    ) -> None:
        self._mux = mux
        self.extensions = extensions
        self._values_transformer = values_transformer
        self._pump_thread = pump_thread

    @property
    def output(self) -> dict[str, Any] | None:
        """Block until the run completes and return the final state."""
        self._pump_thread.join()
        if self._values_transformer._log._error is not None:
            raise self._values_transformer._log._error
        return self._values_transformer._latest

    @property
    def interrupted(self) -> bool:
        """Block until the run completes, then return whether it was interrupted."""
        self._pump_thread.join()
        return self._values_transformer._interrupted

    @property
    def interrupts(self) -> list[Any]:
        """Block until the run completes, then return interrupt payloads."""
        self._pump_thread.join()
        return self._values_transformer._interrupts

    def __iter__(self) -> Iterator[ProtocolEvent]:
        """Iterate all protocol events from the mux's main event log."""
        return iter(self._mux._events)


class AsyncGraphRunStream:
    """Async run stream with transformer-driven projections.

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
