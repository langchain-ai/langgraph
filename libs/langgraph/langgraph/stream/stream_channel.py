from __future__ import annotations

import asyncio
from collections import deque
from collections.abc import AsyncIterator, Awaitable, Callable, Iterator
from typing import TYPE_CHECKING, Generic, TypeVar

if TYPE_CHECKING:
    from langgraph.stream._mux import StreamMux

T = TypeVar("T")


class StreamChannel(Generic[T]):
    """Single-consumer drainable queue for streaming events, with optional
    protocol auto-forwarding.

    When constructed with a `name`, the StreamMux auto-wires every
    `push()` to also inject a `ProtocolEvent` into the main event stream
    using the channel's name as the method. When constructed without a
    name, the channel is local-only — items are only visible to
    in-process consumers that iterate the channel directly.

    Items are popped off the front as the consumer advances — there is
    no retention beyond what's currently queued. A channel accepts
    exactly one subscriber; a second `__iter__` / `__aiter__` call
    raises. Use `tee(n)` / `atee(n)` for fan-out.

    Starts unbound — neither `__iter__` nor `__aiter__` is available
    until the StreamMux calls `_bind(is_async)`. After binding, only
    the matching iteration protocol works; the other raises `TypeError`.

    Pump wiring (set by the run stream, not by `_bind`):
        - `_request_more`: sync pump callable, returns True if a new
          event was produced.
        - `_arequest_more`: async pump coroutine factory, same contract.

    Memory is bounded by caller pace: both sync and async use caller-
    driven pumps, so each cursor advance produces at most one event.

    Lazy-subscribe: `push` appends to the local buffer only when a
    subscriber has registered. Auto-forward via `_wire_fn` always fires
    regardless of subscription state.

    Lifecycle (`close` / `fail`) is managed by the mux — transformers
    don't need to close their channels manually.
    """

    def __init__(self, name: str | None = None, *, maxlen: int | None = None) -> None:
        """Initialize the channel.

        Args:
            name: Optional protocol channel name. When set, the
                StreamMux wires every `push()` to also inject a
                `ProtocolEvent` into the main event stream. Surfaced
                on the wire as `custom:<name>` for user-defined
                transformers, or as `<name>` for channels owned by a
                native transformer (`_native = True`). When `None`,
                the channel is local-only.
            maxlen: Accepted for forward compatibility; currently
                unused. The caller-driven pump bounds memory naturally
                for single-consumer use.

        Raises:
            ValueError: If `maxlen` is not a positive integer or `None`.
        """
        if maxlen is not None and maxlen <= 0:
            raise ValueError("StreamChannel maxlen must be a positive int or None")
        self.name = name
        self._items: deque[tuple[int, T]] = deque()
        self._maxlen: int | None = maxlen
        self._closed = False
        self._error: BaseException | None = None

        self._is_async: bool | None = None

        self._subscribed = False

        self._request_more: Callable[[], bool] | None = None
        self._arequest_more: Callable[[], Awaitable[bool]] | None = None

        self._wire_fn: Callable[[T], None] | None = None
        self._mux: StreamMux | None = None

    # ------------------------------------------------------------------
    # Binding
    # ------------------------------------------------------------------

    def _bind_mux(self, mux: StreamMux) -> None:
        self._mux = mux

    def _bind(self, *, is_async: bool) -> None:
        """Bind this channel to sync or async mode.

        Called by the StreamMux after transformer registration. Must be
        called exactly once before any iteration.

        Args:
            is_async: True to enable async iteration, False for sync.

        Raises:
            RuntimeError: If the channel has already been bound.
        """
        if self._is_async is not None:
            raise RuntimeError("StreamChannel is already bound")
        self._is_async = is_async

    # ------------------------------------------------------------------
    # Mux wiring (not called by transformers directly)
    # ------------------------------------------------------------------

    def _wire(self, fn: Callable[[T], None]) -> None:
        """Install the auto-forward callback (called by StreamMux)."""
        self._wire_fn = fn

    # ------------------------------------------------------------------
    # Producer API
    # ------------------------------------------------------------------

    def push(self, item: T) -> None:
        """Append an item. Auto-forwards if wired.

        The local buffer append is a no-op when no subscriber is
        registered, but auto-forwarding always fires so wired events
        reach the main event log regardless of subscription state.

        Items are stored as `(stamp, item)` tuples where stamp is a
        monotonic counter from the owning mux. Stamps are stripped by
        the default cursors; raw stamped tuples are visible on `_items`.

        Raises:
            RuntimeError: If the channel is closed (and subscribed).
        """
        if self._subscribed:
            if self._closed:
                raise RuntimeError("Cannot push to a closed StreamChannel")
            stamp = self._mux._next_push_seq() if self._mux is not None else 0
            self._items.append((stamp, item))
        if self._wire_fn is not None:
            self._wire_fn(item)

    def close(self) -> None:
        """Mark the channel as complete."""
        self._closed = True

    def fail(self, err: BaseException) -> None:
        """Mark the channel as errored.

        Args:
            err: The exception to surface to the subscriber.
        """
        self._error = err
        self._closed = True

    # ------------------------------------------------------------------
    # Sync iteration (caller-driven pump)
    # ------------------------------------------------------------------

    def __iter__(self) -> Iterator[T]:
        """Subscribe and return a sync cursor. Can be called only once.

        Raises:
            TypeError: If the channel is unbound or bound to async mode.
            RuntimeError: If the channel already has a subscriber.
        """
        if self._is_async is None:
            raise TypeError(
                "StreamChannel has not been bound yet. "
                "Register the transformer with a StreamMux first."
            )
        if self._is_async:
            raise TypeError(
                "This StreamChannel is bound to async mode — use 'async for' instead."
            )
        if self._subscribed:
            raise RuntimeError(
                "StreamChannel already has a subscriber; use .tee(n) for fan-out."
            )
        self._subscribed = True
        return self._sync_cursor()

    def _sync_cursor(self) -> Iterator[T]:
        while True:
            if self._items:
                _stamp, item = self._items.popleft()
                yield item
            elif self._closed:
                if self._error is not None:
                    raise self._error
                return
            elif self._request_more is not None:
                if not self._request_more():
                    if not self._items and not self._closed:
                        return
            else:
                return

    # ------------------------------------------------------------------
    # Async iteration (caller-driven pump)
    # ------------------------------------------------------------------

    def __aiter__(self) -> AsyncIterator[T]:
        """Subscribe and return an async cursor. Can be called only once.

        Raises:
            TypeError: If the channel is unbound or bound to sync mode.
            RuntimeError: If the channel already has a subscriber.
        """
        if self._is_async is None:
            raise TypeError(
                "StreamChannel has not been bound yet. "
                "Register the transformer with a StreamMux first."
            )
        if not self._is_async:
            raise TypeError(
                "This StreamChannel is bound to sync mode — use 'for' instead."
            )
        if self._subscribed:
            raise RuntimeError(
                "StreamChannel already has a subscriber; use .atee(n) for fan-out."
            )
        self._subscribed = True
        return self._async_cursor()

    async def _async_cursor(self) -> AsyncIterator[T]:
        while True:
            if self._items:
                _stamp, item = self._items.popleft()
                yield item
            elif self._closed:
                if self._error is not None:
                    raise self._error
                return
            elif self._arequest_more is not None:
                if not await self._arequest_more():
                    if not self._items and not self._closed:
                        return
            else:
                return

    # ------------------------------------------------------------------
    # Fan-out via tee
    # ------------------------------------------------------------------

    def tee(self, n: int = 2) -> tuple[Iterator[T], ...]:
        """Subscribe and return `n` independent sync iterators.

        Each branch has its own buffer; items pulled from the
        underlying cursor are copied into every branch. Branches are
        naturally bounded by caller pace since the sync pump is
        caller-driven.

        Args:
            n: Number of branches to create. Must be >= 1.

        Returns:
            A tuple of `n` iterators over the same underlying stream.

        Raises:
            TypeError: If the channel is unbound or bound to async mode.
            RuntimeError: If the channel already has a subscriber.
            ValueError: If `n` < 1.
        """
        if n < 1:
            raise ValueError("tee() requires n >= 1")
        source = self.__iter__()
        buffers: list[deque[T]] = [deque() for _ in range(n)]
        exhausted = [False]

        def branch(i: int) -> Iterator[T]:
            buf = buffers[i]
            while True:
                if buf:
                    yield buf.popleft()
                elif exhausted[0]:
                    return
                else:
                    try:
                        item = next(source)
                    except StopIteration:
                        exhausted[0] = True
                        return
                    for b in buffers:
                        b.append(item)

        return tuple(branch(i) for i in range(n))

    def atee(self, n: int = 2) -> tuple[AsyncIterator[T], ...]:
        """Subscribe and return `n` independent async iterators.

        Caller-driven fan-out: each branch's `__anext__` either pops
        from its own buffer or, under a shared `asyncio.Lock`, pulls
        one item from the underlying cursor and distributes it to
        every branch's buffer.

        Args:
            n: Number of branches to create. Must be >= 1.

        Returns:
            A tuple of `n` async iterators over the same underlying
            stream.

        Raises:
            TypeError: If the channel is unbound or bound to sync mode.
            RuntimeError: If the channel already has a subscriber.
            ValueError: If `n` < 1.
        """
        if n < 1:
            raise ValueError("atee() requires n >= 1")
        source = self.__aiter__()
        buffers: list[deque[T]] = [deque() for _ in range(n)]
        exhausted = [False]
        error: list[BaseException | None] = [None]
        lock = asyncio.Lock()

        async def branch(i: int) -> AsyncIterator[T]:
            buf = buffers[i]
            while True:
                if buf:
                    yield buf.popleft()
                    continue
                if exhausted[0]:
                    if error[0] is not None:
                        raise error[0]
                    return
                async with lock:
                    if buf or exhausted[0]:
                        continue
                    try:
                        item = await source.__anext__()
                    except StopAsyncIteration:
                        exhausted[0] = True
                        continue
                    except Exception as e:
                        error[0] = e
                        exhausted[0] = True
                        continue
                    for b in buffers:
                        b.append(item)

        return tuple(branch(i) for i in range(n))
