from __future__ import annotations

import asyncio
import threading
from collections.abc import AsyncIterator, Callable, Iterator
from typing import Generic, TypeVar

T = TypeVar("T")


class _EventLogBase(Generic[T]):
    """Shared producer API for sync and async event logs.

    Append-only buffer that supports multiple independent consumers.
    Subclasses provide the iteration protocol (sync or async).

    Producer API (thread-safe):
        push(item)  — append an item, notify all waiting cursors
        close()     — mark the log as done
        fail(err)   — mark the log as errored
    """

    def __init__(self) -> None:
        self._items: list[T] = []
        self._closed = False
        self._error: BaseException | None = None
        self._lock = threading.Lock()

    def push(self, item: T) -> None:
        """Append *item* and wake all waiting cursors."""
        with self._lock:
            if self._closed:
                raise RuntimeError("Cannot push to a closed EventLog")
            self._items.append(item)
        self._notify()

    def close(self) -> None:
        """Mark the log as complete — open cursors will finish cleanly."""
        with self._lock:
            self._closed = True
        self._notify()

    def fail(self, err: BaseException) -> None:
        """Mark the log as errored — open cursors will raise *err*."""
        with self._lock:
            self._error = err
            self._closed = True
        self._notify()

    def _notify(self) -> None:
        """Wake waiting consumers. Overridden by subclasses."""


class EventLog(_EventLogBase[T]):
    """Sync event log with pull-based iteration.

    Each call to ``__iter__`` creates a new cursor starting from the
    beginning. When a cursor catches up to the buffer and the log is
    not yet closed, it calls ``_request_more`` to pull more data from
    the producer (typically the graph iterator via the run stream).

    If no ``_request_more`` callback is set, the cursor returns
    immediately when it reaches the end of the buffer — this is the
    behavior used in unit tests where items are pushed before iteration.

    Use ``AsyncEventLog`` for async consumers.
    """

    def __init__(self) -> None:
        super().__init__()
        self._request_more: Callable[[], bool] | None = None

    def __iter__(self) -> Iterator[T]:
        """Return a new independent sync cursor over the log."""
        return self._sync_cursor()

    def _sync_cursor(self) -> Iterator[T]:
        cursor = 0
        while True:
            if cursor < len(self._items):
                item = self._items[cursor]
                cursor += 1
                yield item
            elif self._closed:
                if self._error is not None:
                    raise self._error
                return
            elif self._request_more is not None:
                # Pull from the producer until this log gets a new item
                # or the graph is exhausted (which closes the log).
                while cursor >= len(self._items) and not self._closed:
                    if not self._request_more():
                        break
            else:
                # No producer callback and not closed — buffer is complete.
                return


class AsyncEventLog(_EventLogBase[T]):
    """Async event log with multi-cursor iteration.

    Each call to ``__aiter__`` creates a new cursor starting from the
    beginning. Cursors await ``asyncio.Future`` objects when they
    catch up to the producer.

    The producer (``push``/``close``/``fail``) is safe to call from
    any thread — async waiters are notified via
    ``loop.call_soon_threadsafe``.

    Use ``EventLog`` for sync consumers.
    """

    def __init__(self) -> None:
        super().__init__()
        self._async_waiters: list[asyncio.Future[None]] = []

    def _notify(self) -> None:
        waiters = self._async_waiters
        if not waiters:
            return
        self._async_waiters = []
        for fut in waiters:
            if not fut.done():
                try:
                    fut.get_loop().call_soon_threadsafe(fut.set_result, None)
                except RuntimeError:
                    # Event loop already closed — nothing to notify.
                    pass

    def __aiter__(self) -> AsyncIterator[T]:
        """Return a new independent async cursor over the log."""
        return self._async_cursor()

    async def _async_cursor(self) -> AsyncIterator[T]:
        cursor = 0
        while True:
            if cursor < len(self._items):
                yield self._items[cursor]
                cursor += 1
            elif self._closed:
                if self._error is not None:
                    raise self._error
                return
            else:
                loop = asyncio.get_running_loop()
                fut: asyncio.Future[None] = loop.create_future()
                self._async_waiters.append(fut)
                await fut
