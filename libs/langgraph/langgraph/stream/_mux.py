from __future__ import annotations

import time
from collections.abc import Callable
from typing import Any

from langgraph.stream._event_log import EventLog
from langgraph.stream._types import ProtocolEvent, StreamTransformer
from langgraph.stream.stream_channel import StreamChannel


class StreamMux:
    """Central event dispatcher for the streaming infrastructure.

    Owns the main event log and routes events through a transformer
    pipeline. StreamChannels discovered in transformer projections are
    auto-wired so that every ``push()`` also injects a ``ProtocolEvent``
    into the main log.

    Pass ``is_async=True`` when the mux will be consumed via async
    iteration (``handler.astream()``). All ``EventLog`` and
    ``StreamChannel`` instances discovered during ``register()`` are
    automatically bound to the matching mode.
    """

    def __init__(self, *, is_async: bool = False) -> None:
        self._is_async = is_async
        self._events: EventLog[ProtocolEvent] = EventLog()
        self._events._bind(is_async=is_async)
        self._transformers: list[StreamTransformer] = []
        self._channels: list[StreamChannel[Any]] = []
        self._seq = 0

    def register(self, transformer: StreamTransformer) -> dict[str, Any]:
        """Register a transformer and return its projection dict.

        Calls ``transformer.init()``, stores the transformer for event
        processing, binds any ``EventLog`` or ``StreamChannel`` instances
        in the projection, and returns the projection.
        """
        projection = transformer.init()
        if not isinstance(projection, dict):
            raise TypeError(
                f"StreamTransformer.init() must return a dict, "
                f"got {type(projection).__name__}"
            )
        self._transformers.append(transformer)
        self._bind_and_wire(projection)
        return projection

    def push(self, event: ProtocolEvent) -> None:
        """Route *event* through all transformers, then append to the main log.

        Each transformer's ``process()`` is called in registration order.
        If any transformer returns ``False``, the event is suppressed
        from the main log (but transformers that already saw it keep
        their side-effects).

        Seq is assigned right before an event enters the main log, not
        before the transformer pipeline runs. This ensures that events
        auto-forwarded from StreamChannels during ``process()`` get
        earlier seq numbers than the original event, preserving
        monotonic ordering in the log.
        """
        keep = True
        for transformer in self._transformers:
            if not transformer.process(event):
                keep = False
        if keep:
            self._seq += 1
            event["seq"] = self._seq
            self._events.push(event)

    def close(self) -> None:
        """Finalize all transformers, close all channels and the main log.

        If any transformer's ``finalize()`` raises, the remaining
        transformers, channels, and the main log are still closed.
        The first error is re-raised after cleanup completes.
        """
        first_error: BaseException | None = None
        for transformer in self._transformers:
            try:
                transformer.finalize()
            except BaseException as e:
                if first_error is None:
                    first_error = e
        for ch in self._channels:
            ch._close()
        self._events.close()
        if first_error is not None:
            raise first_error

    def fail(self, err: BaseException) -> None:
        """Fail all transformers, channels, and the main log.

        If any transformer's ``fail()`` raises, the remaining
        transformers, channels, and the main log are still failed.
        """
        for transformer in self._transformers:
            try:
                transformer.fail(err)
            except BaseException:
                pass
        for ch in self._channels:
            ch._fail(err)
        self._events.fail(err)

    # ------------------------------------------------------------------
    # Binding and StreamChannel auto-wiring
    # ------------------------------------------------------------------

    def _bind_and_wire(self, projection: dict[str, Any]) -> None:
        """Bind and wire EventLog / StreamChannel instances in *projection*."""
        for value in projection.values():
            if isinstance(value, StreamChannel):
                value._bind(is_async=self._is_async)
                self._channels.append(value)
                channel_name = value.name

                def _make_forward(name: str) -> Callable[[Any], None]:
                    def _forward(item: Any) -> None:
                        self._forward(name, item)

                    return _forward

                value._wire(_make_forward(channel_name))
            elif isinstance(value, EventLog):
                value._bind(is_async=self._is_async)

    def _forward(self, channel_name: str, item: Any) -> None:
        """Inject a ProtocolEvent for a StreamChannel push.

        Forwarded events bypass the transformer pipeline to avoid
        infinite recursion (a transformer that pushes to a channel
        during ``process()`` would re-trigger itself). These events
        are visible in the main event log but are not passed through
        transformers' ``process()`` methods.
        """
        self._seq += 1
        event: ProtocolEvent = {
            "type": "event",
            "seq": self._seq,
            "method": f"custom:{channel_name}",
            "params": {
                "namespace": [],
                "timestamp": int(time.time() * 1000),
                "data": item,
            },
        }
        self._events.push(event)
