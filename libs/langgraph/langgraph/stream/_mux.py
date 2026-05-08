from __future__ import annotations

import asyncio
import time
from collections.abc import Awaitable, Callable
from typing import Any

from langgraph.stream._types import (
    ProtocolEvent,
    StreamTransformer,
    transformer_requires_async,
)
from langgraph.stream.stream_channel import StreamChannel

TransformerFactory = Callable[["tuple[str, ...]"], StreamTransformer]
"""Factory that builds a scoped transformer for a mux.

Called once per `StreamMux` with the mux's scope (typically `()` for
the root). Standard transformer classes accept a single positional
scope argument, so the class itself is a valid factory. User
transformers can close over their config:
`lambda scope: MyTransformer(scope, foo=...)`.
"""


class StreamMux:
    """Central event dispatcher for the streaming infrastructure.

    Owns the main event log and routes events through a transformer
    pipeline. StreamChannels with a name discovered in transformer
    projections are auto-wired so that every `push()` also injects a
    `ProtocolEvent` into the main log. StreamChannels without a name
    are local-only.

    Pass `is_async=True` when the mux will be consumed via async
    iteration (`handler.astream()`). All StreamChannel instances
    discovered during registration are automatically bound to the
    matching mode.

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
        factories: list[TransformerFactory] | None = None,
        scope: tuple[str, ...] = (),
        _assign_seq: bool = True,
    ) -> None:
        """Initialize the mux and register transformers in order.

        Callers pass either `transformers` (pre-built instances) or
        `factories` (callables producing fresh instances per mux). Each
        transformer's `init()` is called, projections are merged into
        `extensions`, `_native` keys are recorded in `native_keys`, and
        any StreamChannel instances are bound and (if named) wired.

        Args:
            transformers: Already-built transformer instances. Registered
                only on this mux — they are NOT cloned into child
                mini-muxes built by `_make_child`. Use `factories` for
                transformers that should propagate to nested scopes.
            is_async: True for async dispatch (`apush` / `aclose` /
                `afail`), False for the sync path.
            factories: One-argument callables `(scope) -> StreamTransformer`.
                Called once with this mux's `scope` here, and cloned
                again per child scope by `_make_child` so each
                sub-mux gets fresh instances.
            scope: The namespace the mux operates within. The root mux
                is `()`.
            _assign_seq: Internal flag for child muxes. Root muxes assign
                monotonic `seq` numbers when appending to their main event
                log; child muxes share forwarded event objects and must not
                mutate their envelopes.

        Raises:
            RuntimeError: If any transformer requires an async run but
                the mux is in sync mode.
            TypeError: If a transformer's `init()` doesn't return a dict.
            ValueError: If transformers' projection keys collide.
        """
        self.is_async = is_async
        self.scope: tuple[str, ...] = scope
        self._assign_seq = _assign_seq
        self._events: StreamChannel[ProtocolEvent] = StreamChannel()
        self._events._bind(is_async=is_async)
        self._events._bind_mux(self)
        self._transformers: list[StreamTransformer] = []
        self._channels: list[StreamChannel[Any]] = []
        self._seq = 0
        self._push_seq = 0

        self.extensions: dict[str, Any] = {}
        self.native_keys: set[str] = set()
        self._projection_owners: dict[str, str] = {}
        self._transformer_by_key: dict[str, StreamTransformer] = {}

        # Stored only when constructed from factories — used by
        # `_make_child` to clone the transformer pipeline at a deeper
        # scope. Pre-built transformers can't be cloned, so a mux
        # built with `transformers=` rejects child construction.
        self._factories: list[TransformerFactory] | None = (
            list(factories) if factories is not None else None
        )
        self._pump_fn: Callable[[], bool] | None = None
        self._apump_fn: Callable[[], Awaitable[bool]] | None = None

        # Factories run first (they propagate to child mini-muxes
        # via `_make_child`), then any pre-built `transformers=`
        # instances are registered as root-only — they aren't cloned
        # for child scopes.
        if factories is not None:
            for factory in factories:
                self._register(factory(scope))
        for transformer in transformers or ():
            self._register(transformer)

    def transformer_by_key(self, key: str) -> StreamTransformer | None:
        """Return the transformer that contributed `key` to the projection."""
        return self._transformer_by_key.get(key)

    def _next_push_seq(self) -> int:
        self._push_seq += 1
        return self._push_seq

    # ------------------------------------------------------------------
    # Pump wiring + mini-mux nesting
    # ------------------------------------------------------------------

    def bind_pump(self, fn: Callable[[], bool]) -> None:
        """Wire the sync pull callback onto every projection in this mux.

        Records the pump on the mux so child mini-muxes built by
        `_make_child` can inherit it. Propagates to:
        - the main event log (`self._events`)
        - every projection StreamChannel in `extensions`
        - any registered transformer that exposes `_bind_pump` (e.g.
          `MessagesTransformer` so `ChatModelStream` instances drive the
          shared pump from their cursors)
        """
        self._pump_fn = fn
        self._events._request_more = fn
        for ch in self._channels:
            ch._request_more = fn
        for transformer in self._transformers:
            bind = getattr(transformer, "_bind_pump", None)
            if bind is not None:
                bind(fn)

    def bind_apump(self, fn: Callable[[], Awaitable[bool]]) -> None:
        """Async counterpart to `bind_pump`."""
        self._apump_fn = fn
        self._events._arequest_more = fn
        for ch in self._channels:
            ch._arequest_more = fn
        for transformer in self._transformers:
            abind = getattr(transformer, "_bind_apump", None)
            if abind is not None:
                abind(fn)

    def _make_child(self, scope: tuple[str, ...]) -> StreamMux:
        """Build a mini-mux with the same factories scoped to `scope`.

        Used by `SubgraphTransformer` to attach a fresh transformer
        pipeline to each discovered subgraph handle. The child mux
        inherits the current pump bindings (so cursors on its
        projection logs drive the root pump), carries the same factory
        list forward to any grandchild subgraphs, and does not assign
        `seq` numbers so forwarded events can be shared without
        mutating their envelope.

        Raises:
            RuntimeError: If the mux was not constructed with
                `factories=`. Mini-muxes require factories so each scope
                gets its own fresh transformer instances.
        """
        if self._factories is None:
            raise RuntimeError(
                "StreamMux._make_child requires the mux to be constructed "
                "with `factories=`; pre-built transformers can't be "
                "cloned to a new scope."
            )
        child = StreamMux(
            factories=self._factories,
            is_async=self.is_async,
            scope=scope,
            _assign_seq=False,
        )
        if self._pump_fn is not None:
            child.bind_pump(self._pump_fn)
        if self._apump_fn is not None:
            child.bind_apump(self._apump_fn)
        return child

    def _register(self, transformer: StreamTransformer) -> None:
        """Register a single transformer.

        Calls `transformer.init()`, stores the transformer for event
        processing, binds any StreamChannel instances in the projection,
        and merges the projection into `extensions`.
        """
        if transformer_requires_async(transformer) and not self.is_async:
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
            attributions = ", ".join(
                f"{key!r} (owned by {self._projection_owners[key]})"
                for key in sorted(conflicts)
            )
            raise ValueError(
                f"Transformer {type(transformer).__name__} returned "
                f"projection keys that conflict with already-registered "
                f"keys: {attributions}"
            )
        is_native = bool(getattr(transformer, "_native", False))
        self._transformers.append(transformer)
        self._bind_and_wire(projection, native=is_native)
        self.extensions.update(projection)
        owner_name = type(transformer).__name__
        for key in projection:
            self._projection_owners[key] = owner_name
            self._transformer_by_key[key] = transformer
        if is_native:
            self.native_keys.update(projection.keys())
        transformer._on_register(self)

    def push(self, event: ProtocolEvent) -> None:
        """Route an event through all transformers, then append to the main log.

        Each transformer's `process()` is called in registration order.
        If any transformer returns False, the event is suppressed from
        the main log, but transformers that already saw it keep their
        side effects.

        On the root mux, `seq` is assigned right before an event enters
        the main log, not before the transformer pipeline runs. This
        ensures that events auto-forwarded from StreamChannels during
        `process()` get earlier seq numbers than the original event,
        preserving monotonic ordering in the root log. Child muxes do
        not assign `seq`, so subgraph forwarding can share event objects
        without mutating their envelopes.

        Args:
            event: The protocol event to dispatch.
        """
        keep = True
        for transformer in self._transformers:
            if not transformer.process(event):
                keep = False
        if keep:
            if self._assign_seq:
                self._seq += 1
                event["seq"] = self._seq
            self._events.push(event)

    def close(self) -> None:
        """Finalize all transformers, close all projections and the main log.

        StreamChannels discovered in transformer projections are
        auto-closed after `finalize()` runs — transformers don't need
        to close them manually. If any transformer's `finalize()` raises,
        the remaining transformers, projections, and the main log are
        still closed; the first error is re-raised after cleanup
        completes.

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
        for ch in self._channels:
            if not ch._closed:
                ch.close()
        self._events.close()
        if first_error is not None:
            raise first_error

    def fail(self, err: BaseException) -> None:
        """Fail all transformers, projections, and the main log.

        StreamChannels discovered in transformer projections are
        auto-failed — transformers don't need to fail them manually.
        If any transformer's `fail()` raises, the remaining
        transformers, projections, and the main log are still failed.

        Args:
            err: The exception that ended the run.
        """
        for transformer in self._transformers:
            try:
                transformer.fail(err)
            except BaseException:
                pass
        for ch in self._channels:
            if not ch._closed:
                ch.fail(err)
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

        The main log append is a non-blocking `push` — matching v1's
        `put_nowait` shape. The root mux assigns `seq`; child muxes do
        not, so forwarded subgraph events can be shared without copying.
        Memory is bounded by caller pace via the caller-driven pump; see
        `StreamChannel` for the full tradeoff story.

        Args:
            event: The protocol event to dispatch.
        """
        keep = True
        for transformer in self._transformers:
            if not await transformer.aprocess(event):
                keep = False
        if keep:
            if self._assign_seq:
                self._seq += 1
                event["seq"] = self._seq
            self._events.push(event)

    async def aclose(self) -> None:
        """Finalize on the async lane.

        Awaits every task started via `StreamTransformer.schedule()`
        across all transformers, then calls `afinalize()` on each,
        then auto-closes channels and the main event log.

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
        for ch in self._channels:
            if not ch._closed:
                ch.close()
        self._events.close()
        if first_error is not None:
            raise first_error

    async def afail(self, err: BaseException) -> None:
        """Fail on the async lane.

        Cancels every scheduled task across all transformers, awaits
        them to completion, then runs each transformer's `afail` hook
        and auto-fails channels and the main event log.

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
        for ch in self._channels:
            if not ch._closed:
                ch.fail(err)
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

    def _bind_and_wire(
        self, projection: dict[str, Any], *, native: bool = False
    ) -> None:
        """Bind and optionally wire StreamChannel instances in a projection.

        All StreamChannels are bound and tracked. Channels with a name
        are additionally wired for protocol auto-forwarding.

        Args:
            projection: The projection dict returned by a transformer's
                `init()`.
            native: True when the owning transformer is `_native`.
                Named channels owned by a native transformer use the
                channel name directly as the protocol method;
                user-defined channels are prefixed with `custom:`.
        """
        for value in projection.values():
            if isinstance(value, StreamChannel):
                value._bind(is_async=self.is_async)
                value._bind_mux(self)
                self._channels.append(value)
                if value.name is not None:
                    method = value.name if native else f"custom:{value.name}"

                    def _make_forward(method_name: str) -> Callable[[Any], None]:
                        def _forward(item: Any) -> None:
                            self._forward(method_name, item)

                        return _forward

                    value._wire(_make_forward(method))

    def _forward(self, method: str, item: Any) -> None:
        """Inject a ProtocolEvent for a StreamChannel push.

        Forwarded events bypass the transformer pipeline to avoid
        infinite recursion (a transformer that pushes to a channel
        during `process()` would re-trigger itself). These events are
        visible in this mux's main event log but are not passed through
        transformers' `process()` methods. Only the root mux assigns
        `seq` to forwarded channel events.

        Args:
            method: The full protocol method (already with or without
                the `custom:` prefix; resolved by `_bind_and_wire`).
            item: The payload pushed onto the channel.
        """
        event: ProtocolEvent = {
            "type": "event",
            "method": method,
            "params": {
                "namespace": [],
                "timestamp": int(time.time() * 1000),
                "data": item,
            },
        }
        if self._assign_seq:
            self._seq += 1
            event["seq"] = self._seq
        self._events.push(event)
