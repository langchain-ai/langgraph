"""Per-channel event → items state machines.

Used both by the projection iterators (`_ValuesProjection`,
`_MessagesProjection`, `_ToolCallsProjection`, `_SubgraphsProjection`) on
`AsyncThreadStream` / `SyncThreadStream`, and by `interleave_projections`,
which drives multiple decoders from one shared subscription.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from typing import Any, Literal, Protocol

#: Channel names the public ``interleave_projections`` API accepts as built-ins.
SUPPORTED_INTERLEAVE_CHANNELS = (
    "values",
    "messages",
    "tool_calls",
    "subgraphs",
    "updates",
    "checkpoints",
    "tasks",
)

#: Channel names that ``infer_channel`` recognizes as first-class protocol
#: methods but that ``interleave_projections`` has no decoder for. Routing them
#: to the extension/``custom:`` fallback would subscribe to a channel that never
#: matches and silently yield nothing, so they are rejected up front (fail
#: closed). ``lifecycle`` is control-plane (drives run output/interrupt); ``tools``
#: is the wire alias for the public ``tool_calls`` channel.
RESERVED_INTERLEAVE_CHANNELS = frozenset({"lifecycle", "tools", "input"})


def validate_interleave_channels(channels: list[str]) -> None:
    """Reject reserved protocol channel names before they hit the fallback.

    Genuine extension names pass through untouched; only names that
    ``infer_channel`` treats as built-in methods without an interleave decoder
    are rejected, so a typo'd or unsupported protocol channel surfaces an error
    instead of an empty stream.
    """
    for ch in channels:
        if ch in RESERVED_INTERLEAVE_CHANNELS:
            hint = ' (use "tool_calls")' if ch == "tools" else ""
            raise ValueError(
                f"{ch!r} is not a valid interleave_projections channel{hint}. "
                f"Supported channels: {', '.join(SUPPORTED_INTERLEAVE_CHANNELS)}, "
                "or an extension name."
            )


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


SubgraphStatus = Literal["started", "completed", "failed", "interrupted"]


def _parse_namespace_segment(segment: str) -> tuple[str, str | None]:
    name, sep, task_id = segment.partition(":")
    return name, task_id if sep else None


def _terminal_from_tasks_result(
    data: dict[str, Any],
) -> tuple[SubgraphStatus, str | None]:
    if data.get("interrupts"):
        return "interrupted", None
    error = data.get("error")
    if error:
        return "failed", str(error)
    return "completed", None


def _is_direct_child(namespace: list[str], scope: tuple[str, ...]) -> bool:
    return len(namespace) == len(scope) + 1 and tuple(namespace[: len(scope)]) == scope


class Decoder(Protocol):
    def feed(self, event: Mapping[str, Any]) -> Iterable[Any]: ...


class DataDecoder:
    """Yields `params.data` from events of a single `method`.

    Covers the channels whose projection is just "emit the payload": `values`,
    `updates`, `checkpoints`, `tasks` — the SDK analog of local's
    `Values`/`Updates`/`Checkpoints`/`TasksTransformer`, all of which push
    `params["data"]` unchanged. The REST-state seeding for `values` stays at
    the projection layer; it is a one-shot pre-stream fetch, not part of the
    event state machine.

    Args:
        method: The protocol `method` this decoder consumes.
        namespace: When not `None`, events whose namespace differs are ignored
            (scope filter, mirroring the local transformers' `namespace != scope`
            check). `None` consumes every namespace — the historical `values`
            projection behavior, where subscription scoping is handled upstream.
    """

    def __init__(self, method: str, namespace: list[str] | None = None):
        self._method = method
        self._namespace = list(namespace) if namespace is not None else None

    def feed(self, event: Mapping[str, Any]) -> Iterable[Any]:
        if event.get("method") != self._method:
            return
        params = event.get("params") or {}
        if self._namespace is not None and _event_namespace(params) != self._namespace:
            return
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

    def feed(self, event: Mapping[str, Any]) -> Iterable[Any]:
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

    def feed(self, event: Mapping[str, Any]) -> Iterable[Any]:
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


class SubgraphsDecoder:
    """Discovers child subgraph handles and fans out events to active ones.

    Mirrors the per-event body of `_SubgraphsProjection._subgraphs_iter`
    (`_async/stream.py:963-1041`) plus `_apply_tasks_result`. Root-inbox
    forwarding and terminal-status-on-close stay at the projection / wrapper
    layer.

    Args:
        scope: Tuple-form namespace of this decoder's parent. `()` for root.
        handle_factory: Keyword-only `(path, graph_name, trigger_call_id) -> handle`.
    """

    def __init__(self, scope: tuple[str, ...], handle_factory: Callable[..., Any]):
        self._scope = scope
        self._handle_factory = handle_factory
        self._active: dict[tuple[str, ...], Any] = {}
        self._seen: set[tuple[str, ...]] = set()

    def feed(self, event: Mapping[str, Any]) -> Iterable[Any]:
        params = event.get("params") or {}
        namespace = _event_namespace(params)
        data = params.get("data")
        if not isinstance(data, dict):
            return
        method = event.get("method")

        # 1. Fanout: first active child whose path prefixes this namespace.
        ns_tuple = tuple(namespace)
        for child_path, child_handle in self._active.items():
            child_len = len(child_path)
            if len(ns_tuple) >= child_len and ns_tuple[:child_len] == child_path:
                child_handle._push_event(event)
                break

        # 2 + 3. Discovery / status from tasks; discovery from lifecycle.
        if method == "tasks":
            if "result" in data:
                self._apply_tasks_result(namespace, data)
            elif _is_direct_child(namespace, self._scope):
                yield from self._discover(namespace)
        elif (
            method == "lifecycle"
            and data.get("event") == "started"
            and _is_direct_child(namespace, self._scope)
        ):
            yield from self._discover(namespace)

    def _discover(self, namespace: list[str]) -> Iterable[Any]:
        path = tuple(namespace)
        if path in self._seen:
            return
        self._seen.add(path)
        graph_name, trigger_call_id = _parse_namespace_segment(path[-1])
        handle = self._handle_factory(
            path=path,
            graph_name=graph_name or None,
            trigger_call_id=trigger_call_id,
        )
        self._active[path] = handle
        yield handle

    def _apply_tasks_result(self, namespace: list[str], data: dict[str, Any]) -> None:
        result_id = data.get("id")
        if not result_id:
            return
        parent_path = tuple(namespace)
        for child_path, handle in list(self._active.items()):
            if child_path[:-1] != parent_path:
                continue
            if handle.trigger_call_id != result_id:
                continue
            status, error = _terminal_from_tasks_result(data)
            handle._finish(status, error)
            del self._active[child_path]


class ExtensionsDecoder:
    """Yields `params.data` from one named custom channel.

    Mirrors `_ExtensionProjection._iter` (`_async/stream.py:1278-1299`), with
    an added name filter so it can share one subscription in interleave.

    Args:
        name: The extension name. Only `custom` events whose `data["name"]`
            matches are consumed.
    """

    def __init__(self, name: str):
        if not name:
            raise ValueError("extension name must be non-empty.")
        self._name = name

    def feed(self, event: Mapping[str, Any]) -> Iterable[Any]:
        if event.get("method") != "custom":
            return
        params = event.get("params") or {}
        data = params.get("data")
        if not isinstance(data, dict):
            return
        if data.get("name") != self._name:
            return
        yield data
