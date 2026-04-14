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

    def subscribe(self, offset: int = 0) -> _Cursor[T]:
        """Create a new cursor starting at *offset*.

        If *offset* is 0 the cursor replays the entire log.  If
        *offset* equals ``len(self)`` the cursor starts from the
        current tip and only sees future events.
        """
        return _Cursor(self, offset)

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
            if not fut.done():
                try:
                    fut.get_loop().call_soon_threadsafe(fut.set_result, None)
                except RuntimeError:
                    # Loop already closed — ignore.
                    pass
        self._waiters.clear()


class _Cursor(Generic[T]):
    """An independent async iterator over an :class:`EventLog`."""

    __slots__ = ("_log", "_offset")

    def __init__(self, log: EventLog[T], offset: int) -> None:
        self._log = log
        self._offset = offset

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
                loop = asyncio.get_running_loop()
                fut: asyncio.Future[None] = loop.create_future()
                self._log._waiters.append(fut)
            # Wait outside the lock
            try:
                await fut
            except asyncio.CancelledError:
                # Remove our future so it doesn't accumulate in the list.
                try:
                    self._log._waiters.remove(fut)
                except ValueError:
                    pass  # Already removed by _wake_all
                raise


__all__ = ["EventLog"]
