"""Tests for `StreamTransformer.before_builtins` lane ordering.

`before_builtins = True` transformers are registered ahead of the
rest, preserving relative order within each lane. This lets
content-mutating transformers run before built-ins like
`MessagesTransformer` that eagerly snapshot text fields into their
projections.
"""

from __future__ import annotations

import time
from typing import Any, ClassVar

from langgraph.stream._mux import StreamMux
from langgraph.stream._types import StreamTransformer
from langgraph.stream.stream_channel import StreamChannel
from langgraph.stream.transformers import (
    LifecycleTransformer,
    MessagesTransformer,
    TasksTransformer,
)

TS = int(time.time() * 1000)


def _messages_event(namespace: list[str], data: Any) -> dict[str, Any]:
    return {
        "type": "event",
        "method": "messages",
        "params": {"namespace": namespace, "timestamp": TS, "data": data},
    }


class _Tap(StreamTransformer):
    """Records the order it observed each event."""

    required_stream_modes: ClassVar[tuple[str, ...]] = ()

    def __init__(self, scope: tuple[str, ...] = (), *, label: str = "tap") -> None:
        super().__init__(scope)
        self.label = label
        self.log: list[str] = []
        self._channel: StreamChannel[str] = StreamChannel()

    def init(self) -> dict[str, Any]:
        return {f"tap_{self.label}": self._channel}

    def process(self, event: dict[str, Any]) -> bool:
        self.log.append(self.label)
        return True


class _PreTap(_Tap):
    before_builtins: ClassVar[bool] = True


class _TextRedactor(StreamTransformer):
    """Mutates `text-delta` events in place to a fixed redacted string."""

    before_builtins: ClassVar[bool] = True
    required_stream_modes: ClassVar[tuple[str, ...]] = ("messages",)

    def __init__(self, scope: tuple[str, ...] = ()) -> None:
        super().__init__(scope)
        self._channel: StreamChannel[str] = StreamChannel()

    def init(self) -> dict[str, Any]:
        return {"redactor": self._channel}

    def process(self, event: dict[str, Any]) -> bool:
        if event.get("method") != "messages":
            return True
        payload, _meta = event["params"]["data"]
        if isinstance(payload, dict) and payload.get("event") == "content-block-delta":
            delta = payload.get("delta") or {}
            if delta.get("type") == "text-delta":
                delta["text"] = "[REDACTED]"
        return True


def test_before_builtins_factories_run_before_others() -> None:
    """A `before_builtins=True` factory is registered ahead of the rest."""

    seen: list[type[StreamTransformer]] = []

    class _PostTap(_Tap):
        def __init__(self, scope: tuple[str, ...] = ()) -> None:
            super().__init__(scope, label="post")

        def process(self, event: dict[str, Any]) -> bool:
            seen.append(_PostTap)
            return True

    class _EagerTap(_Tap):
        before_builtins: ClassVar[bool] = True

        def __init__(self, scope: tuple[str, ...] = ()) -> None:
            super().__init__(scope, label="eager")

        def process(self, event: dict[str, Any]) -> bool:
            seen.append(_EagerTap)
            return True

    mux = StreamMux(
        factories=[_PostTap, _EagerTap],
        scope=(),
        is_async=False,
    )
    # `_EagerTap` was supplied second but should be registered first.
    types_in_order = [type(t) for t in mux._transformers]
    assert types_in_order.index(_EagerTap) < types_in_order.index(_PostTap)

    mux.push(
        _messages_event([], ({"event": "message-start", "role": "ai", "id": "m1"}, {}))
    )
    # And it ran first when the event was dispatched.
    assert seen == [_EagerTap, _PostTap]


def test_within_lane_order_preserved() -> None:
    """Within each lane, the supplied order is the registration order."""

    class _A(_PreTap):
        pass

    class _B(_PreTap):
        pass

    class _C(_Tap):
        pass

    class _D(_Tap):
        pass

    mux = StreamMux(
        factories=[
            lambda scope: _C(scope, label="c"),
            lambda scope: _A(scope, label="a"),
            lambda scope: _D(scope, label="d"),
            lambda scope: _B(scope, label="b"),
        ],
        scope=(),
        is_async=False,
    )
    order = [t.label for t in mux._transformers]  # type: ignore[attr-defined]
    # Pre lane (a, b) ahead of default lane (c, d). Within each, supplied order kept.
    assert order == ["a", "b", "c", "d"]


def test_redactor_runs_before_messages_transformer() -> None:
    """Content mutated by a pre-lane transformer reaches `MessagesTransformer`."""

    # Order supplied: built-ins first (as in pregel/main.py), then the
    # opt-in pre-lane redactor. Partitioning should still register the
    # redactor first.
    mux = StreamMux(
        factories=[MessagesTransformer, _TextRedactor],
        scope=(),
        is_async=False,
    )
    types_in_order = [type(t) for t in mux._transformers]
    assert types_in_order.index(_TextRedactor) < types_in_order.index(
        MessagesTransformer
    )


def test_lifecycle_unaffected_by_pre_lane_observer() -> None:
    """An observer-only pre-lane transformer doesn't break lifecycle bookkeeping."""

    class _NoopPreObserver(StreamTransformer):
        before_builtins: ClassVar[bool] = True
        required_stream_modes: ClassVar[tuple[str, ...]] = ("tasks",)

        def __init__(self, scope: tuple[str, ...] = ()) -> None:
            super().__init__(scope)
            self._channel: StreamChannel[str] = StreamChannel()
            self.seen: list[tuple[str, ...]] = []

        def init(self) -> dict[str, Any]:
            return {"noop_observer": self._channel}

        def process(self, event: dict[str, Any]) -> bool:
            if event.get("method") == "tasks":
                self.seen.append(tuple(event["params"]["namespace"]))
            return True

    mux = StreamMux(
        factories=[LifecycleTransformer, TasksTransformer, _NoopPreObserver],
        scope=(),
        is_async=False,
    )
    types_in_order = [type(t) for t in mux._transformers]
    assert types_in_order.index(_NoopPreObserver) < types_in_order.index(
        LifecycleTransformer
    )

    observer = next(t for t in mux._transformers if isinstance(t, _NoopPreObserver))
    lifecycle = next(
        t for t in mux._transformers if isinstance(t, LifecycleTransformer)
    )

    # Push a synthetic `tasks` event that lifecycle would normally track.
    mux.push(
        {
            "type": "event",
            "method": "tasks",
            "params": {
                "namespace": ["child:abc"],
                "timestamp": TS,
                "data": {"name": "child"},
            },
        }
    )

    # Pre-lane observer saw the event, AND lifecycle's bookkeeping still
    # registered the new namespace (the observer didn't mutate anything).
    assert observer.seen == [("child:abc",)]
    assert ("child:abc",) in lifecycle._seen  # type: ignore[attr-defined]


def test_default_is_false() -> None:
    """`StreamTransformer.before_builtins` defaults to False."""

    assert StreamTransformer.before_builtins is False
    assert MessagesTransformer.before_builtins is False
    assert LifecycleTransformer.before_builtins is False


def test_pre_lane_mutation_lands_in_messages_projection() -> None:
    """End-to-end: text mutated by a pre-lane transformer is what
    `MessagesTransformer` snapshots into its `ChatModelStream` projection.

    Without `before_builtins`, the redactor would run after
    MessagesTransformer's eager extraction and the projection would
    contain the raw, un-redacted text.
    """

    mux = StreamMux(
        factories=[MessagesTransformer, _TextRedactor],
        scope=(),
        is_async=False,
    )
    messages_transformer = next(
        t for t in mux._transformers if isinstance(t, MessagesTransformer)
    )
    # Unblock both the mux's main log and the messages projection log so
    # synthetic pushes are accepted without a real consumer attached.
    mux._events._subscribed = True
    messages_transformer._log._subscribed = True

    meta = {"langgraph_node": "model", "run_id": "run-1"}

    # message-start → MessagesTransformer creates a ChatModelStream.
    mux.push(
        _messages_event(
            [],
            ({"event": "message-start", "role": "ai", "id": "msg-1"}, meta),
        )
    )

    # content-block-delta carrying the secret. The redactor (pre-lane)
    # mutates `delta.text` BEFORE MessagesTransformer snapshots it.
    mux.push(
        _messages_event(
            [],
            (
                {
                    "event": "content-block-delta",
                    "index": 0,
                    "delta": {"type": "text-delta", "text": "secret@example.com"},
                },
                meta,
            ),
        )
    )

    # Capture the still-open stream before message-finish removes it from
    # MessagesTransformer's `_by_run` dict.
    chat_stream = messages_transformer._by_run["run-1"]  # type: ignore[attr-defined]

    # message-finish → closes the stream.
    mux.push(
        _messages_event(
            [],
            ({"event": "message-finish"}, meta),
        )
    )

    # The redactor mutated `delta.text` to "[REDACTED]" before
    # MessagesTransformer snapshotted the string into the text
    # accumulator. Without `before_builtins`, the accumulator would hold
    # the raw "secret@example.com".
    assert chat_stream._text_acc == "[REDACTED]", (  # type: ignore[attr-defined]
        f"expected redacted text in projection, got {chat_stream._text_acc!r}"
    )
