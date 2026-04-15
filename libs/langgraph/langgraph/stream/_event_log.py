"""Replayable append-only event buffer for StreamingHandler.

``EventLog`` stores protocol events in an ordered list and supports
multiple independent async iterators, each with their own cursor
offset.  Subscribers that join mid-stream replay from a given offset
without losing earlier events.
"""

from __future__ import annotations

import asyncio
import threading
from typing import Generic, TypeVar

T = TypeVar("T")


def _resolve_future(fut: asyncio.Future[None]) -> None:
    """Set a future's result if it hasn't already completed or been cancelled.

    Runs on the event loop thread (scheduled via ``call_soon_threadsafe``)
    so that the ``done()`` check and ``set_result`` are atomic with
    respect to cancellation.
    """
    if not fut.done():
        fut.set_result(None)


class EventLog(Generic[T]):
    """Append-only event buffer with cursor-based async iteration.

    Multiple consumers can subscribe independently and each will see
    every event from their starting offset onward.
    """

    __slots__ = ("_items", "_closed", "_error", "_waiters", "_lock")

    def __init__(self) -> None:
        self._items: list[T] = []
        self._closed = False
        self._error: BaseException | None = None
        self._waiters: list[asyncio.Future[None]] = []
        self._lock = threading.Lock()

    # -- Producer API -------------------------------------------------------

    def append(self, item: T) -> None:
        """Append an event and wake all waiting consumers."""
        with self._lock:
            if self._closed:
                raise RuntimeError("EventLog is closed")
            self._items.append(item)
            self._wake_all()

    def close(self) -> None:
        """Mark the log as complete.  Iterators will end gracefully."""
        with self._lock:
            self._closed = True
            self._wake_all()

    def fail(self, error: BaseException) -> None:
        """Mark the log as failed.  Iterators will raise *error*."""
        with self._lock:
            self._error = error
            self._closed = True
            self._wake_all()

    # -- Consumer API -------------------------------------------------------

    def __aiter__(self) -> _Cursor[T]:
        """Return a fresh cursor from the beginning of the log."""
        return _Cursor(self)

    # -- Inspection ---------------------------------------------------------

    def __len__(self) -> int:
        return len(self._items)

    def __getitem__(self, index: int) -> T:
        return self._items[index]

    @property
    def closed(self) -> bool:
        return self._closed

    # -- Internal -----------------------------------------------------------

    def _wake_all(self) -> None:
        for fut in self._waiters:
            try:
                fut.get_loop().call_soon_threadsafe(_resolve_future, fut)
            except RuntimeError:
                # Loop already closed — ignore.
                pass
        self._waiters.clear()


class _Cursor(Generic[T]):
    """An independent async iterator over an :class:`EventLog`."""

    __slots__ = ("_log", "_offset")

    def __init__(self, log: EventLog[T]) -> None:
        self._log = log
        self._offset = 0

    def __aiter__(self) -> _Cursor[T]:
        return self

    async def __anext__(self) -> T:
        while True:
            with self._log._lock:
                if self._offset < len(self._log._items):
                    item = self._log._items[self._offset]
                    self._offset += 1
                    return item
                if self._log._error is not None:
                    raise self._log._error
                if self._log._closed:
                    raise StopAsyncIteration
                # Nothing available yet — register a waiter
                fut: asyncio.Future[None] = asyncio.get_running_loop().create_future()
                self._log._waiters.append(fut)
            # Wait outside the lock
            try:
                await fut
            except asyncio.CancelledError:
                with self._log._lock:
                    try:
                        self._log._waiters.remove(fut)
                    except ValueError:
                        pass  # Already removed by _wake_all
                raise


__all__ = ["EventLog"]
