"""Per-message streaming objects for StreamingHandler.

``ChatModelStream`` is the synchronous variant returned by
``GraphRunStream.messages``.  Properties (``.text``, ``.reasoning``,
``.usage``) return final accumulated values.

``AsyncChatModelStream`` is the asynchronous variant returned by
``AsyncGraphRunStream.messages``.  Projections are dual
async-iterable + awaitable (e.g. ``async for delta in msg.text``
or ``full = await msg.text``).
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable, Generator, Iterator
from typing import Any

from langgraph.stream._types import UsageInfo


# ---------------------------------------------------------------------------
# Sync dual projection — iterable of deltas, str() for accumulated text
# ---------------------------------------------------------------------------


class _SyncDualProjection:
    """Pump-driven sync iterable of string deltas.

    Iterating yields incremental text fragments as the pump delivers
    new ``content-block-delta`` events.  Calling ``str()`` drains the
    pump and returns the full accumulated string.

    This is the sync counterpart of :class:`_DualProjection` (the async
    variant used by ``AsyncChatModelStream``).
    """

    __slots__ = ("_stream", "_attr", "_pump_one")

    def __init__(
        self,
        stream: ChatModelStream,
        attr: str,
        pump_one: Callable[[], bool],
    ) -> None:
        self._stream = stream
        self._attr = attr
        self._pump_one = pump_one

    def __iter__(self) -> Iterator[str]:
        prev_len = 0
        while True:
            cur = getattr(self._stream, self._attr)
            if len(cur) > prev_len:
                yield cur[prev_len:]
                prev_len = len(cur)
            if self._stream._done:
                return
            if not self._pump_one():
                # Source exhausted — yield any remaining
                cur = getattr(self._stream, self._attr)
                if len(cur) > prev_len:
                    yield cur[prev_len:]
                return

    def __str__(self) -> str:
        while not self._stream._done:
            if not self._pump_one():
                break
        return getattr(self._stream, self._attr)

    def __repr__(self) -> str:
        return repr(getattr(self._stream, self._attr))

    def __bool__(self) -> bool:
        return bool(getattr(self._stream, self._attr))


# ---------------------------------------------------------------------------
# Sync variant
# ---------------------------------------------------------------------------


class ChatModelStream:
    """Synchronous per-message object for a single LLM response.

    Created by :class:`~langgraph.stream.transformers.MessagesTransformer`
    and yielded by ``GraphRunStream.messages``.  By the time the sync
    iterator yields a ``ChatModelStream``, the message lifecycle is
    complete and all properties contain their final values.

    Projections:

    - ``.text`` — accumulated text content (``str``)
    - ``.reasoning`` — accumulated reasoning content (``str``)
    - ``.usage`` — :class:`UsageInfo` or ``None``
    - ``.namespace`` / ``.node`` — provenance metadata
    """

    def __init__(
        self,
        *,
        namespace: list[str] | None = None,
        node: str | None = None,
        message_id: str | None = None,
    ) -> None:
        self._namespace = namespace or []
        self._node = node
        self._message_id = message_id

        # Accumulated state
        self._text_acc = ""
        self._reasoning_acc = ""
        self._usage_value: UsageInfo | None = None
        self._done = False

        # Optional pump for sync streaming (set via _bind_pump)
        self._pump_one: Callable[[], bool] | None = None

    # -- Pump binding (called by GraphRunStream) ---------------------------

    def _bind_pump(self, pump_one: Callable[[], bool]) -> None:
        """Bind a pump function for sync token-by-token streaming.

        When bound, ``.text`` and ``.reasoning`` return
        :class:`_SyncDualProjection` instances that drive the pump and
        yield deltas as the LLM produces tokens.
        """
        self._pump_one = pump_one

    # -- Public projections ------------------------------------------------

    @property
    def text(self) -> str | _SyncDualProjection:
        """Text content.

        When a pump is bound (sync streaming), returns a
        :class:`_SyncDualProjection` — iterable of deltas,
        ``str()`` for the full accumulated text.  Otherwise returns
        the accumulated text string directly.
        """
        if self._pump_one is not None and not self._done:
            return _SyncDualProjection(self, "_text_acc", self._pump_one)
        return self._text_acc

    @property
    def reasoning(self) -> str | _SyncDualProjection:
        """Reasoning content.

        Same dual behavior as :attr:`text`.
        """
        if self._pump_one is not None and not self._done:
            return _SyncDualProjection(self, "_reasoning_acc", self._pump_one)
        return self._reasoning_acc

    @property
    def usage(self) -> UsageInfo | None:
        """Usage info, available after the message finishes."""
        if self._pump_one is not None and not self._done:
            while not self._done:
                if not self._pump_one():
                    break
        return self._usage_value

    @property
    def namespace(self) -> list[str]:
        return self._namespace

    @property
    def node(self) -> str | None:
        return self._node

    @property
    def message_id(self) -> str | None:
        return self._message_id

    @property
    def done(self) -> bool:
        return self._done

    # -- Internal API (called by MessagesTransformer) ----------------------

    def _push_content_block_delta(self, data: dict[str, Any]) -> None:
        """Process a ``content-block-delta`` event."""
        block = data.get("content_block", {})
        btype = block.get("type", "")

        if btype == "text":
            delta_text = block.get("text", "")
            if delta_text:
                self._text_acc += delta_text
        elif btype == "reasoning":
            delta_r = block.get("reasoning", "")
            if delta_r:
                self._reasoning_acc += delta_r

    def _push_content_block_finish(self, data: dict[str, Any]) -> None:
        """Process a ``content-block-finish`` event."""
        block = data.get("content_block", {})
        btype = block.get("type", "")

        if btype == "text":
            full_text = block.get("text", "")
            if full_text and full_text != self._text_acc:
                self._text_acc = full_text
        elif btype == "reasoning":
            full_r = block.get("reasoning", "")
            if full_r and full_r != self._reasoning_acc:
                self._reasoning_acc = full_r

    def _finish(self, data: dict[str, Any]) -> None:
        """Process a ``message-finish`` event."""
        self._done = True
        self._usage_value = data.get("usage")

    def _fail(self, error: BaseException) -> None:
        """Process a ``message-error`` event."""
        self._done = True


# ---------------------------------------------------------------------------
# Async dual-projection helpers
# ---------------------------------------------------------------------------


class _DualProjection:
    """Async iterable of deltas that is also awaitable for the final value.

    When iterated, yields delta values (e.g. text fragments) as they arrive.
    When awaited, returns the accumulated final value (e.g. full text string).
    """

    def __init__(self) -> None:
        self._deltas: list[Any] = []
        self._done = False
        self._error: BaseException | None = None
        self._waiters: list[asyncio.Future[None]] = []
        self._final_value: Any = None
        self._final_set = False

    # -- Producer API (called by AsyncChatModelStream) ---------------------

    def _push(self, delta: Any) -> None:
        """Add a new delta value."""
        self._deltas.append(delta)
        self._wake()

    def _finish(self, accumulated: Any) -> None:
        """Set the final accumulated value and mark as done."""
        self._final_value = accumulated
        self._final_set = True
        self._done = True
        self._wake()

    def _fail(self, error: BaseException) -> None:
        self._error = error
        self._done = True
        self._wake()

    def _wake(self) -> None:
        for fut in self._waiters:
            if not fut.done():
                try:
                    fut.get_loop().call_soon_threadsafe(fut.set_result, None)
                except RuntimeError:
                    pass
        self._waiters.clear()

    # -- Async iterable (yields deltas) ------------------------------------

    def __aiter__(self) -> _DualProjectionIterator:
        return _DualProjectionIterator(self)

    # -- Awaitable (returns final value) -----------------------------------

    def __await__(self) -> Generator[Any, None, Any]:
        return self._await_impl().__await__()

    async def _await_impl(self) -> Any:
        while not self._final_set:
            if self._error is not None:
                raise self._error
            loop = asyncio.get_running_loop()
            fut: asyncio.Future[None] = loop.create_future()
            self._waiters.append(fut)
            await fut
        if self._error is not None:
            raise self._error
        return self._final_value


class _DualProjectionIterator:
    """Async iterator over a :class:`_DualProjection`'s deltas."""

    __slots__ = ("_proj", "_offset")

    def __init__(self, proj: _DualProjection) -> None:
        self._proj = proj
        self._offset = 0

    def __aiter__(self) -> _DualProjectionIterator:
        return self

    async def __anext__(self) -> Any:
        while True:
            if self._offset < len(self._proj._deltas):
                item = self._proj._deltas[self._offset]
                self._offset += 1
                return item
            if self._proj._error is not None:
                raise self._proj._error
            if self._proj._done:
                raise StopAsyncIteration
            loop = asyncio.get_running_loop()
            fut: asyncio.Future[None] = loop.create_future()
            self._proj._waiters.append(fut)
            await fut


# ---------------------------------------------------------------------------
# Async variant
# ---------------------------------------------------------------------------


class AsyncChatModelStream(ChatModelStream):
    """Asynchronous per-message streaming object for a single LLM response.

    Created by :class:`~langgraph.stream.transformers.MessagesTransformer`
    and yielded by ``AsyncGraphRunStream.messages``.  Content-block events
    are fed into this object until ``message-finish``.

    Projections:

    - ``.text`` — async iterable of text deltas; awaitable for full text
    - ``.reasoning`` — async iterable of reasoning deltas; awaitable for
      full reasoning text
    - ``.usage`` — awaitable for :class:`UsageInfo`
    - ``.namespace`` / ``.node`` — provenance metadata
    """

    def __init__(
        self,
        *,
        namespace: list[str] | None = None,
        node: str | None = None,
        message_id: str | None = None,
    ) -> None:
        super().__init__(namespace=namespace, node=node, message_id=message_id)
        self._text_proj = _DualProjection()
        self._reasoning_proj = _DualProjection()
        self._usage_proj = _DualProjection()

    # -- Public projections (override sync properties) ---------------------

    @property
    def text(self) -> _DualProjection:
        """Text content — async iterable of deltas, awaitable for full text."""
        return self._text_proj

    @property
    def reasoning(self) -> _DualProjection:
        """Reasoning content — async iterable of deltas, awaitable for full text."""
        return self._reasoning_proj

    @property
    def usage(self) -> _DualProjection:
        """Usage info — awaitable for :class:`UsageInfo`."""
        return self._usage_proj

    # -- Internal API (extend base to also drive projections) --------------

    def _push_content_block_delta(self, data: dict[str, Any]) -> None:
        """Process a ``content-block-delta`` event."""
        super()._push_content_block_delta(data)
        block = data.get("content_block", {})
        btype = block.get("type", "")

        if btype == "text":
            delta_text = block.get("text", "")
            if delta_text:
                self._text_proj._push(delta_text)
        elif btype == "reasoning":
            delta_r = block.get("reasoning", "")
            if delta_r:
                self._reasoning_proj._push(delta_r)

    def _finish(self, data: dict[str, Any]) -> None:
        """Process a ``message-finish`` event."""
        super()._finish(data)
        self._text_proj._finish(self._text_acc)
        self._reasoning_proj._finish(self._reasoning_acc)
        self._usage_proj._finish(self._usage_value)

    def _fail(self, error: BaseException) -> None:
        """Process a ``message-error`` event."""
        super()._fail(error)
        self._text_proj._fail(error)
        self._reasoning_proj._fail(error)
        self._usage_proj._fail(error)


__all__ = ["AsyncChatModelStream", "ChatModelStream", "_SyncDualProjection"]
