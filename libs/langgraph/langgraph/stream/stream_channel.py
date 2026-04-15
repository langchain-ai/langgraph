"""StreamChannel — typed push-based channel for StreamTransformer projections.

A ``StreamChannel`` wraps an :class:`EventLog` and declares a protocol
channel name.  When the :class:`StreamMux` detects a ``StreamChannel``
in a transformer's ``init()`` return, it wires every ``push()`` call to
inject a :class:`ProtocolEvent` into the main event stream using the
channel's name as the ``method``.

In-process consumers iterate the channel directly (it is an async
iterable).  Remote SDK clients subscribe via
``session.subscribe("custom:<channelName>")``.

"""

from __future__ import annotations

from collections.abc import AsyncIterator, Callable
from typing import Any, Generic, TypeVar

from langgraph.stream._event_log import EventLog

T = TypeVar("T")


class StreamChannel(Generic[T]):
    """A typed push-based channel that integrates with the mux.

    Transformer authors create a ``StreamChannel`` in ``init()`` and
    call ``push()`` inside ``process()`` to emit domain objects.  The
    mux auto-wires pushes to protocol events and auto-closes/fails the
    channel on run completion.
    """

    __slots__ = ("channel_name", "_log", "_on_push")

    def __init__(self, name: str) -> None:
        self.channel_name = name
        self._log: EventLog[T] = EventLog()
        self._on_push: Callable[[Any], None] | None = None

    def push(self, item: T) -> None:
        """Push an item to the channel.

        If the mux has wired this channel, the push also injects a
        protocol event into the main event stream.
        """
        self._log.append(item)
        if self._on_push is not None:
            self._on_push(item)

    # -- Async iteration (in-process consumption) ---------------------------

    def __aiter__(self) -> AsyncIterator[T]:
        return aiter(self._log)

    # -- Internal (called by the mux) ---------------------------------------

    def _wire(self, fn: Callable[[Any], None]) -> None:
        """Wire a callback invoked on every ``push()``.  Called by the mux."""
        self._on_push = fn

    def _close(self) -> None:
        """Close the underlying log.  Called by the mux on normal completion."""
        self._log.close()

    def _fail(self, err: BaseException) -> None:
        """Fail the underlying log.  Called by the mux on failure."""
        self._log.fail(err)


def is_stream_channel(value: object) -> bool:
    """Check if *value* is a :class:`StreamChannel` instance."""
    return isinstance(value, StreamChannel)


__all__ = ["StreamChannel", "is_stream_channel"]
