from __future__ import annotations

import asyncio
from collections import deque
from collections.abc import AsyncIterator, Callable, Iterator
from typing import Generic, TypeVar

T = TypeVar("T")


class BufferOverflowError(RuntimeError):
    """Raised when an EventLog cursor falls off the back of a bounded buffer.

    Mirrors the ``restored: false`` signal from the protocol's reconnection
    story (§ 06): consumers that fall behind the retention window get an
    explicit error and can decide to rebuild from a snapshot rather than
    silently losing events.
    """


class EventLog(Generic[T]):
    """Append-only buffer that supports multiple independent consumers.

    Starts unbound — neither ``__iter__`` nor ``__aiter__`` is available
    until the ``StreamMux`` calls ``_bind(is_async)``.  After binding,
    only the matching iteration protocol works; the other raises
    ``TypeError``.

    All access is single-threaded: sync mode is caller-driven (no
    background thread), async mode runs entirely on the event loop.

    Producer API:
        push(item)  — append an item, notify all waiting cursors
        close()     — mark the log as done
        fail(err)   — mark the log as errored

    Sync iteration is pull-based: when a cursor catches up it calls
    ``_request_more`` to drive the graph forward.

    Async iteration uses a shared ``asyncio.Event`` — cursors await
    the event when they catch up, and the producer sets it on each push.

    Bounded mode
    ------------
    Pass ``maxlen=N`` to cap memory. When the buffer is full, ``push``
    drops the oldest item to make room. Cursors track an absolute
    sequence number; a cursor that falls off the back of the retention
    window raises ``BufferOverflowError`` on its next read.

    New cursors start at the current head of the buffer, not at seq 0
    — they see whatever is still retained. This matches the protocol's
    reconnection semantics (§ 06: "missed events can be replayed from
    a bounded buffer").
    """

    def __init__(self, maxlen: int | None = None) -> None:
        if maxlen is not None and maxlen <= 0:
            raise ValueError("EventLog maxlen must be a positive int or None")
        self._items: deque[T] = deque()
        self._maxlen: int | None = maxlen
        self._first_seq = 0  # absolute seq of _items[0]
        self._closed = False
        self._error: BaseException | None = None

        # Binding state — None means unbound.
        self._is_async: bool | None = None

        # Sync pull callback (set by the run stream, not by bind).
        self._request_more: Callable[[], bool] | None = None

        # Async notification (allocated on bind).
        self._event: asyncio.Event | None = None

    # ------------------------------------------------------------------
    # Binding
    # ------------------------------------------------------------------

    def _bind(self, *, is_async: bool) -> None:
        """Bind this log to sync or async mode.

        Called by the ``StreamMux`` after transformer registration.
        Must be called exactly once before any iteration.
        """
        if self._is_async is not None:
            raise RuntimeError("EventLog is already bound")
        self._is_async = is_async
        if is_async:
            self._event = asyncio.Event()

    # ------------------------------------------------------------------
    # Producer API
    # ------------------------------------------------------------------

    def push(self, item: T) -> None:
        """Append *item* and wake all waiting cursors.

        In bounded mode, evicts the oldest item first if the buffer
        is full, advancing ``_first_seq`` so cursors can detect that
        they've fallen off the back of the retention window.
        """
        if self._closed:
            raise RuntimeError("Cannot push to a closed EventLog")
        if self._maxlen is not None and len(self._items) >= self._maxlen:
            self._items.popleft()
            self._first_seq += 1
        self._items.append(item)
        self._notify()

    def close(self) -> None:
        """Mark the log as complete — open cursors will finish cleanly."""
        self._closed = True
        self._notify()

    def fail(self, err: BaseException) -> None:
        """Mark the log as errored — open cursors will raise *err*."""
        self._error = err
        self._closed = True
        self._notify()

    # ------------------------------------------------------------------
    # Notification
    # ------------------------------------------------------------------

    def _notify(self) -> None:
        """Wake async cursors waiting for data.

        No-op when sync-bound (no event exists).
        """
        if self._event is not None:
            self._event.set()

    # ------------------------------------------------------------------
    # Sync iteration (pull-based)
    # ------------------------------------------------------------------

    def __iter__(self) -> Iterator[T]:
        """Return a new independent sync cursor over the log."""
        if self._is_async is None:
            raise TypeError(
                "EventLog has not been bound yet. "
                "Register the transformer with a StreamMux first."
            )
        if self._is_async:
            raise TypeError(
                "This EventLog is bound to async mode — use 'async for' instead."
            )
        return self._sync_cursor()

    def _sync_cursor(self) -> Iterator[T]:
        # Start at the current head — if maxlen is None this is 0 (seen everything),
        # if bounded this is wherever retention currently begins.
        seq = self._first_seq
        while True:
            if seq < self._first_seq:
                raise BufferOverflowError(
                    f"Cursor fell {self._first_seq - seq} items behind the "
                    f"bounded EventLog's retention window (maxlen={self._maxlen})"
                )
            idx = seq - self._first_seq
            if idx < len(self._items):
                item = self._items[idx]
                seq += 1
                yield item
            elif self._closed:
                if self._error is not None:
                    raise self._error
                return
            elif self._request_more is not None:
                # Pull from the producer until this log gets a new item
                # or the graph is exhausted (which closes the log).
                while (seq - self._first_seq) >= len(self._items) and not self._closed:
                    if not self._request_more():
                        break
            else:
                # No producer callback and not closed — buffer is complete.
                return

    # ------------------------------------------------------------------
    # Async iteration
    # ------------------------------------------------------------------

    def __aiter__(self) -> AsyncIterator[T]:
        """Return a new independent async cursor over the log."""
        if self._is_async is None:
            raise TypeError(
                "EventLog has not been bound yet. "
                "Register the transformer with a StreamMux first."
            )
        if not self._is_async:
            raise TypeError("This EventLog is bound to sync mode — use 'for' instead.")
        return self._async_cursor()

    async def _async_cursor(self) -> AsyncIterator[T]:
        assert self._event is not None
        seq = self._first_seq
        while True:
            if seq < self._first_seq:
                raise BufferOverflowError(
                    f"Cursor fell {self._first_seq - seq} items behind the "
                    f"bounded EventLog's retention window (maxlen={self._maxlen})"
                )
            idx = seq - self._first_seq
            if idx < len(self._items):
                yield self._items[idx]
                seq += 1
            elif self._closed:
                if self._error is not None:
                    raise self._error
                return
            else:
                self._event.clear()
                if (seq - self._first_seq) >= len(self._items) and not self._closed:
                    await self._event.wait()
