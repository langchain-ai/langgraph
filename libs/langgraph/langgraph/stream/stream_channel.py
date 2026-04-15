from __future__ import annotations

from collections.abc import AsyncIterator, Callable, Iterator
from typing import Generic, TypeVar

from langgraph.stream._event_log import EventLog

T = TypeVar("T")


class StreamChannel(Generic[T]):
    """A named projection channel with optional protocol auto-forwarding.

    Wraps an `EventLog[T]` and declares a protocol channel name. When the
    `StreamMux` detects a `StreamChannel` in a transformer's ``init()``
    return value, it automatically wires every ``push()`` to inject a
    `ProtocolEvent` into the main event stream using the channel's name
    as the ``method``.

    In-process consumers iterate the channel directly (``for item in ch``
    or ``async for item in ch``). Remote SDK clients subscribe via
    ``session.subscribe("custom:<channelName>")``.

    Lifecycle (``_close`` / ``_fail``) is managed by the mux — transformers
    using only StreamChannels don't need ``finalize`` / ``fail`` hooks.
    """

    def __init__(self, name: str) -> None:
        self.name = name
        self._log: EventLog[T] = EventLog()
        self._wire_fn: Callable[[T], None] | None = None

    def push(self, item: T) -> None:
        """Append *item* to the log and auto-forward if wired."""
        self._log.push(item)
        if self._wire_fn is not None:
            self._wire_fn(item)

    # ------------------------------------------------------------------
    # Mux lifecycle hooks (not called by transformers directly)
    # ------------------------------------------------------------------

    def _wire(self, fn: Callable[[T], None]) -> None:
        """Install the auto-forward callback (called by StreamMux)."""
        self._wire_fn = fn

    def _close(self) -> None:
        """Close the underlying log (called by StreamMux on run end)."""
        self._log.close()

    def _fail(self, err: BaseException) -> None:
        """Fail the underlying log (called by StreamMux on run error)."""
        self._log.fail(err)

    # ------------------------------------------------------------------
    # Iteration — delegates to the inner EventLog (multi-cursor)
    # ------------------------------------------------------------------

    def __iter__(self) -> Iterator[T]:
        return iter(self._log)

    def __aiter__(self) -> AsyncIterator[T]:
        return self._log.__aiter__()
