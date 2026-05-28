"""Per-channel event → items state machines.

Used both by the projection iterators (`_ValuesProjection`,
`_MessagesProjection`, `_ToolCallsProjection`, `_SubgraphsProjection`) on
`AsyncThreadStream` / `SyncThreadStream`, and by `interleave_projections`,
which drives multiple decoders from one shared subscription.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from typing import Any, Protocol


def _event_namespace(params_field: Any) -> list[str]:
    if not isinstance(params_field, dict):
        return []
    namespace = params_field.get("namespace") or []
    return list(namespace) if isinstance(namespace, list) else []


def _message_event_id(data: dict[str, Any]) -> str | None:
    message_id = data.get("id") or data.get("message_id")
    return str(message_id) if message_id is not None else None


def _message_route_key(data: dict[str, Any], fallback: str | None = None) -> str:
    """Return the routing key for a message-channel event in `active`.

    Keys on `message_id` when available so concurrent messages that share the
    same `run_id` (two AI turns in one agent step) route to independent streams
    rather than colliding on a shared `run:<run_id>` slot.
    """
    message_id = _message_event_id(data)
    if message_id is not None:
        return f"message:{message_id}"
    if fallback is not None:
        return f"message:{fallback}"
    return "__single__"


class Decoder(Protocol):
    def feed(self, event: dict[str, Any]) -> Iterable[Any]: ...


class ValuesDecoder:
    """Yields snapshot dicts from `values` method events.

    Mirrors the per-event body of `_ValuesProjection._values_iter` in
    `langgraph_sdk/_async/stream.py` (the REST-state seeding stays at the
    projection layer; it is a one-shot pre-stream fetch, not part of the
    event state machine).
    """

    def feed(self, event: dict[str, Any]) -> Iterable[Any]:
        if event.get("method") == "values":
            params = event.get("params") or {}
            data = params.get("data")
            if data is not None:
                yield data


class MessagesDecoder:
    """Yields one chat-model stream per `message-start` event.

    Subsequent events route to the matching stream via `stream.dispatch(data)`.
    Mirrors the per-event body of `_MessagesProjection._messages_iter`
    (`_async/stream.py:404-458`). The subscription open/close and the
    `_root_messages_inbox` drain branch stay at the projection layer.

    Args:
        namespace: Events whose namespace differs are ignored (scope filter).
        stream_factory: Keyword-only `(namespace, node, message_id) -> stream`.
            Sync binds `ChatModelStream`; async binds `AsyncChatModelStream`.
    """

    def __init__(
        self,
        namespace: list[str],
        stream_factory: Callable[..., Any],
    ):
        self._namespace = list(namespace)
        self._stream_factory = stream_factory
        self._active: dict[str, Any] = {}  # route_key -> stream

    def feed(self, event: dict[str, Any]) -> Iterable[Any]:
        if event.get("method") != "messages":
            return
        params = event.get("params") or {}
        if _event_namespace(params) != self._namespace:
            return
        data = params.get("data")
        if not isinstance(data, dict):
            return
        if data.get("event") == "message-start":
            message_id = _message_event_id(data)
            key = _message_route_key(data, fallback=message_id)
            metadata = (
                data.get("metadata") if isinstance(data.get("metadata"), dict) else {}
            )
            stream = self._stream_factory(
                namespace=list(self._namespace),
                node=metadata.get("langgraph_node") if metadata else None,
                message_id=message_id,
            )
            self._active[key] = stream
            stream.dispatch(data)
            yield stream
        else:
            key = _message_route_key(data)
            stream = self._active.get(key)
            if stream is None and key == "__single__" and len(self._active) == 1:
                stream = next(iter(self._active.values()))
            if stream is None:
                return
            stream.dispatch(data)
            if data.get("event") in ("message-finish", "error"):
                for route_key, candidate in list(self._active.items()):
                    if candidate is stream:
                        del self._active[route_key]


class ToolCallsDecoder:
    """Yields one tool-call handle per `tool-started` event.

    Mirrors the per-event body of `_ToolCallsProjection._tool_calls_iter`
    (`_async/stream.py:1168-1217`). The thread register/unregister and the
    terminal-error-on-close finally stay at the projection / wrapper layer.

    Args:
        namespace: Events whose namespace differs are ignored.
        handle_factory: Keyword-only `(tool_call_id, name, input, namespace) -> handle`.
    """

    def __init__(self, namespace: list[str], handle_factory: Callable[..., Any]):
        self._namespace = list(namespace)
        self._handle_factory = handle_factory
        self._active: dict[str, Any] = {}

    def feed(self, event: dict[str, Any]) -> Iterable[Any]:
        if event.get("method") != "tools":
            return
        params = event.get("params") or {}
        if _event_namespace(params) != self._namespace:
            return
        data = params.get("data")
        if not isinstance(data, dict):
            return
        tool_call_id = data.get("tool_call_id")
        if not isinstance(tool_call_id, str):
            return
        event_type = data.get("event")
        if event_type == "tool-started":
            name = data.get("tool_name")
            handle = self._handle_factory(
                tool_call_id=tool_call_id,
                name=name if isinstance(name, str) else "",
                input=data.get("input"),
                namespace=list(self._namespace),
            )
            self._active[tool_call_id] = handle
            yield handle
        elif event_type == "tool-output-delta":
            handle = self._active.get(tool_call_id)
            delta = data.get("delta")
            if handle is not None and isinstance(delta, str):
                handle._push_delta(delta)
        elif event_type == "tool-finished":
            handle = self._active.pop(tool_call_id, None)
            if handle is not None:
                handle._finish(data.get("output"))
        elif event_type == "tool-error":
            handle = self._active.pop(tool_call_id, None)
            if handle is not None:
                message = data.get("message")
                handle._fail(
                    RuntimeError(str(message) if message else "Tool call errored")
                )
