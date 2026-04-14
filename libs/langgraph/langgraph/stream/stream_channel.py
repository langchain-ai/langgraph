"""StreamChannel — typed push-based channel for StreamTransformer projections.

A ``StreamChannel`` wraps a list and declares a protocol channel name.
When the :class:`StreamMux` detects a ``StreamChannel`` in a transformer's
``init()`` return, it wires every ``push()`` call to inject a
:class:`ProtocolEvent` into the main event stream using the channel's
name as the ``method``.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, Generic, TypeVar

T = TypeVar("T")


class StreamChannel(Generic[T]):
    """A typed push-based channel that integrates with the mux.

    Transformer authors create a ``StreamChannel`` in ``init()`` and
    call ``push()`` inside ``process()`` to emit domain objects.  The
    mux auto-wires pushes to protocol events.
    """

    __slots__ = ("channel_name", "_items", "_on_push")

    def __init__(self, name: str) -> None:
        self.channel_name = name
        self._items: list[T] = []
        self._on_push: Callable[[Any], None] | None = None

    def push(self, item: T) -> None:
        """Push an item to the channel."""
        self._items.append(item)
        if self._on_push is not None:
            self._on_push(item)

    def _wire(self, fn: Callable[[Any], None]) -> None:
        """Wire a callback invoked on every ``push()``.  Called by the mux."""
        self._on_push = fn


def is_stream_channel(value: object) -> bool:
    """Check if *value* is a :class:`StreamChannel` instance."""
    return isinstance(value, StreamChannel)


__all__ = ["StreamChannel", "is_stream_channel"]
