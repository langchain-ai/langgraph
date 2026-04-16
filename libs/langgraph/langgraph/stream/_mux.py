from __future__ import annotations

import asyncio
import time
from collections.abc import Callable
from typing import Any

from langgraph.stream._event_log import EventLog
from langgraph.stream._types import (
    ProtocolEvent,
    StreamTransformer,
    transformer_requires_async,
)
from langgraph.stream.stream_channel import StreamChannel


class StreamMux:
    """Central event dispatcher for the streaming infrastructure.

    Owns the main event log and routes events through a transformer
    pipeline. StreamChannels discovered in transformer projections are
    auto-wired so that every `push()` also injects a `ProtocolEvent`
    into the main log.

    Pass `is_async=True` when the mux will be consumed via async
    iteration (`handler.astream()`). All EventLog and StreamChannel
    instances discovered during registration are automatically bound
    to the matching mode.

    Attributes:
        extensions: Merged projection dict across all registered
            transformers. Treat as read-only — mutations won't be
            reflected back in individual transformers' state.
        native_keys: Projection keys contributed by transformers with
            `_native = True`.
    """

    def __init__(
        self,
        transformers: list[StreamTransformer] | None = None,
        *,
        is_async: bool = False,
        max_events: int | None = None,
    ) -> None:
        """Initialize the mux and register transformers in order.

        Transformers are fixed at construction time — there is no
        post-init `register()`. Each transformer's `init()` is called,
        projections are merged into `extensions`, `_native` keys are
        recorded in `native_keys`, and any EventLog / StreamChannel
        instances are bound and wired.

        Args:
            transformers: Transformers to register, in dispatch order.
                `None` or empty gives a mux with no projections.
            is_async: True for async dispatch (`apush` / `aclose` /
                `afail`), False for the sync path.
            max_events: Default capacity for every EventLog and
                StreamChannel the mux binds, including the main event
                log. Logs constructed with an explicit `maxlen` keep
                their own setting — the mux only fills in unset
                defaults. `None` leaves the logs unbounded.

        Raises:
            RuntimeError: If any transformer requires an async run but
                the mux is in sync mode.
            TypeError: If a transformer's `init()` doesn't return a dict.
            ValueError: If transformers' projection keys collide.
        """
        self._is_async = is_async
        self._default_maxlen = max_events
        self._events: EventLog[ProtocolEvent] = EventLog(maxlen=max_events)
        self._events._bind(is_async=is_async)
        self._transformers: list[StreamTransformer] = []
        self._channels: list[StreamChannel[Any]] = []
        self._logs: list[EventLog[Any]] = []
        self._seq = 0

        self.extensions: dict[str, Any] = {}
        self.native_keys: set[str] = set()

        for transformer in transformers or ():
            self._register(transformer)

    def _register(self, transformer: StreamTransformer) -> None:
        """Register a single transformer.

        Calls `transformer.init()`, stores the transformer for event
        processing, binds any EventLog or StreamChannel instances in
        the projection, and merges the projection into `extensions`.
        """
        if transformer_requires_async(transformer) and not self._is_async:
            raise RuntimeError(
                f"{type(transformer).__name__} requires an async run — "
                "it overrides aprocess/afinalize/afail or sets "
                "requires_async=True. Use astream(), not stream()."
            )
        projection = transformer.init()
        if not isinstance(projection, dict):
            raise TypeError(
                f"StreamTransformer.init() must return a dict, "
                f"got {type(projection).__name__}"
            )
        conflicts = set(projection) & set(self.extensions)
        if conflicts:
            raise ValueError(
                f"Transformer {type(transformer).__name__} returned "
                f"projection keys that conflict with already-registered "
                f"keys: {conflicts}"
            )
        self._transformers.append(transformer)
        self._bind_and_wire(projection)
        self.extensions.update(projection)
        if getattr(transformer, "_native", False):
            self.native_keys.update(projection.keys())

    def push(self, event: ProtocolEvent) -> None:
        """Route an event through all transformers, then append to the main log.

        Each transformer's `process()` is called in registration order.
        If any transformer returns False, the event is suppressed from
        the main log, but transformers that already saw it keep their
        side effects.

        Seq is assigned right before an event enters the main log, not
        before the transformer pipeline runs. This ensures that events
        auto-forwarded from StreamChannels during `process()` get
        earlier seq numbers than the original event, preserving
        monotonic ordering in the log.

        Args:
            event: The protocol event to dispatch.
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
        """Finalize all transformers, close all projections and the main log.

        EventLogs and StreamChannels discovered in transformer
        projections are auto-closed after `finalize()` runs —
        transformers don't need to close them manually. If any
        transformer's `finalize()` raises, the remaining transformers,
        projections, and the main log are still closed; the first error
        is re-raised after cleanup completes.

        Raises:
            BaseException: The first error raised by a transformer's
                `finalize()`, re-raised after cleanup finishes.
        """
        first_error: BaseException | None = None
        for transformer in self._transformers:
            try:
                transformer.finalize()
            except BaseException as e:
                if first_error is None:
                    first_error = e
        for log in self._logs:
            if not log._closed:
                log.close()
        for ch in self._channels:
            if not ch._log._closed:
                ch._close()
        self._events.close()
        if first_error is not None:
            raise first_error

    def fail(self, err: BaseException) -> None:
        """Fail all transformers, projections, and the main log.

        EventLogs and StreamChannels discovered in transformer
        projections are auto-failed — transformers don't need to fail
        them manually. If any transformer's `fail()` raises, the
        remaining transformers, projections, and the main log are still
        failed.

        Args:
            err: The exception that ended the run.
        """
        for transformer in self._transformers:
            try:
                transformer.fail(err)
            except BaseException:
                pass
        for log in self._logs:
            if not log._closed:
                log.fail(err)
        for ch in self._channels:
            if not ch._log._closed:
                ch._fail(err)
        self._events.fail(err)

    # ------------------------------------------------------------------
    # Async dispatch
    # ------------------------------------------------------------------

    async def apush(self, event: ProtocolEvent) -> None:
        """Dispatch an event on the async lane.

        Awaits each transformer's `aprocess` in registration order
        before appending to the main log. A slow `aprocess` serializes
        the pipeline by design — that's the guarantee that lets a later
        transformer (or a synchronous consumer) see the result of the
        async work. For decoupled work, use `schedule()` from inside
        `process` / `aprocess` instead.

        Args:
            event: The protocol event to dispatch.
        """
        keep = True
        for transformer in self._transformers:
            if not await transformer.aprocess(event):
                keep = False
        if keep:
            self._seq += 1
            event["seq"] = self._seq
            self._events.push(event)

    async def aclose(self) -> None:
        """Finalize on the async lane.

        Awaits every task started via `StreamTransformer.schedule()`
        across all transformers, then calls `afinalize()` on each,
        then auto-closes logs, channels, and the main event log.

        If any scheduled task raised under `on_error="raise"`, or any
        transformer's `afinalize` raises, the exception propagates.
        The caller (the pump) handles it by routing into `afail`.

        Raises:
            BaseException: The first scheduled-task or `afinalize`
                error, re-raised after cleanup.
        """
        pending = self._collect_scheduled_tasks()
        if pending:
            results = await asyncio.gather(*pending, return_exceptions=True)
            first_err = next(
                (
                    r
                    for r in results
                    if isinstance(r, BaseException)
                    and not isinstance(r, asyncio.CancelledError)
                ),
                None,
            )
            if first_err is not None:
                raise first_err

        first_error: BaseException | None = None
        for transformer in self._transformers:
            try:
                await transformer.afinalize()
            except BaseException as e:
                if first_error is None:
                    first_error = e
        for log in self._logs:
            if not log._closed:
                log.close()
        for ch in self._channels:
            if not ch._log._closed:
                ch._close()
        self._events.close()
        if first_error is not None:
            raise first_error

    async def afail(self, err: BaseException) -> None:
        """Fail on the async lane.

        Cancels every scheduled task across all transformers, awaits
        them to completion, then runs each transformer's `afail` hook
        and auto-fails logs, channels, and the main event log.

        Args:
            err: The exception that ended the run.
        """
        pending = self._collect_scheduled_tasks()
        for task in pending:
            task.cancel()
        if pending:
            await asyncio.gather(*pending, return_exceptions=True)

        for transformer in self._transformers:
            try:
                await transformer.afail(err)
            except BaseException:
                pass
        for log in self._logs:
            if not log._closed:
                log.fail(err)
        for ch in self._channels:
            if not ch._log._closed:
                ch._fail(err)
        if not self._events._closed:
            self._events.fail(err)

    def _collect_scheduled_tasks(self) -> list[asyncio.Task[Any]]:
        """Return a snapshot of in-flight tasks scheduled via transformers."""
        return [
            task
            for transformer in self._transformers
            for task in getattr(transformer, "_stream_scheduled_tasks", ())
            if not task.done()
        ]

    # ------------------------------------------------------------------
    # Binding and StreamChannel auto-wiring
    # ------------------------------------------------------------------

    def _bind_and_wire(self, projection: dict[str, Any]) -> None:
        """Bind and wire EventLog / StreamChannel instances in a projection."""
        for value in projection.values():
            if isinstance(value, StreamChannel):
                self._apply_default_maxlen(value._log)
                value._bind(is_async=self._is_async)
                self._channels.append(value)
                channel_name = value.name

                def _make_forward(name: str) -> Callable[[Any], None]:
                    def _forward(item: Any) -> None:
                        self._forward(name, item)

                    return _forward

                value._wire(_make_forward(channel_name))
            elif isinstance(value, EventLog):
                self._apply_default_maxlen(value)
                value._bind(is_async=self._is_async)
                self._logs.append(value)

    def _apply_default_maxlen(self, log: EventLog[Any]) -> None:
        """Fill in the mux's default maxlen when the log hasn't set its own."""
        if log._maxlen is None and self._default_maxlen is not None:
            log._maxlen = self._default_maxlen

    def _forward(self, channel_name: str, item: Any) -> None:
        """Inject a ProtocolEvent for a StreamChannel push.

        Forwarded events bypass the transformer pipeline to avoid
        infinite recursion (a transformer that pushes to a channel
        during `process()` would re-trigger itself). These events are
        visible in the main event log but are not passed through
        transformers' `process()` methods.
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
