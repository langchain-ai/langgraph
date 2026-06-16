"""Unbounded async-iterable append-only log with per-iterator cursors.

Direct port of `libs/sdk/src/client/stream/multi-cursor-buffer.ts`. Each
`async for` loop gets its own cursor starting at position 0, so late
consumers still see all previously buffered items. Lifetime is bounded by
the owning projection / handle; there is no eviction policy.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterable, AsyncIterator
from typing import Generic, TypeVar

T = TypeVar("T")


class MultiCursorBuffer(AsyncIterable[T], Generic[T]):
    def __init__(self) -> None:
        self._items: list[T] = []
        self._wakeups: set[asyncio.Future[None]] = set()
        self._closed = False

    def push(self, item: T) -> None:
        # Post-close pushes are accepted: cursors already terminated miss the item,
        # but new cursors started later see the full log including it. Matches JS.
        self._items.append(item)
        self._wake_all()

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._wake_all()

    def __len__(self) -> int:
        return len(self._items)

    def __aiter__(self) -> AsyncIterator[T]:
        return _Cursor(self)

    def _wake_all(self) -> None:
        for fut in self._wakeups:
            if not fut.done():
                fut.set_result(None)
        self._wakeups.clear()


class _Cursor(Generic[T]):
    def __init__(self, buffer: MultiCursorBuffer[T]) -> None:
        self._buffer = buffer
        self._idx = 0

    def __aiter__(self) -> _Cursor[T]:
        return self

    async def __anext__(self) -> T:
        while True:
            if self._idx < len(self._buffer._items):
                item = self._buffer._items[self._idx]
                self._idx += 1
                return item
            if self._buffer._closed:
                raise StopAsyncIteration
            loop = asyncio.get_running_loop()
            fut: asyncio.Future[None] = loop.create_future()
            self._buffer._wakeups.add(fut)
            try:
                await fut
            finally:
                self._buffer._wakeups.discard(fut)
