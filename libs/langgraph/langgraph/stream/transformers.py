"""Built-in stream transformers for StreamingHandler.

``ValuesTransformer`` extracts ``values`` events and maintains the latest
state per namespace.  ``MessagesTransformer`` groups ``messages`` events
into :class:`ChatModelStream` instances.
"""

from __future__ import annotations

from typing import Any

from langgraph.stream._event_log import EventLog
from langgraph.stream._types import ProtocolEvent
from langgraph.stream.chat_model_stream import ChatModelStream

# Type alias for the stream class constructor signature
_StreamCls = type[ChatModelStream]


class ValuesTransformer:
    """Extracts ``values`` events and populates a values event log.

    Maintains the latest state per namespace and provides a separate
    event log that :class:`AsyncGraphRunStream` / :class:`GraphRunStream` uses for ``.values``
    iteration.

    Implements the :class:`StreamTransformer` protocol.
    """

    name = "values"

    def __init__(self) -> None:
        self._values_log: EventLog[dict[str, Any]] = EventLog()
        self._latest: dict[str, Any] = {}

    @property
    def value(self) -> EventLog[dict[str, Any]]:
        return self._values_log

    @property
    def values_log(self) -> EventLog[dict[str, Any]]:
        return self._values_log

    def get_latest(self, ns_key: str = "") -> Any:
        return self._latest.get(ns_key)

    def init(self) -> Any:
        return None

    def process(self, event: ProtocolEvent) -> bool:
        if event["method"] != "values":
            return True

        ns = event["params"].get("namespace", [])
        data = event["params"]["data"]
        ns_key = "|".join(ns) if ns else ""
        self._latest[ns_key] = data

        # Append to the values log for iteration
        self._values_log.append({"namespace": ns, "data": data})
        return True

    def finalize(self) -> None:
        self._values_log.close()

    def fail(self, err: BaseException) -> None:
        self._values_log.fail(err)


class MessagesTransformer:
    """Groups ``messages`` events into :class:`ChatModelStream` instances.

    One ``ChatModelStream`` is created per ``message-start`` event.
    Content-block events are routed to the active stream until
    ``message-finish`` or ``message-error`` closes it.

    Implements the :class:`StreamTransformer` protocol.
    """

    name = "messages"

    def __init__(
        self,
        *,
        namespace: list[str] | None = None,
        node_filter: str | None = None,
        stream_cls: _StreamCls | None = None,
    ) -> None:
        self._namespace = namespace
        self._node_filter = node_filter
        self._stream_cls: _StreamCls = stream_cls or ChatModelStream

        # Message log for .messages iteration
        self._messages_log: EventLog[ChatModelStream] = EventLog()

        # Current active stream per namespace key
        self._active: dict[str, ChatModelStream] = {}

    @property
    def value(self) -> EventLog[ChatModelStream]:
        return self._messages_log

    @property
    def messages_log(self) -> EventLog[ChatModelStream]:
        return self._messages_log

    def init(self) -> Any:
        return None

    def process(self, event: ProtocolEvent) -> bool:
        if event["method"] != "messages":
            return True

        ns = event["params"].get("namespace", [])
        node = event["params"].get("node")
        data = event["params"]["data"]

        # Apply namespace filter
        if self._namespace is not None:
            if ns[: len(self._namespace)] != self._namespace:
                return True

        # Apply node filter
        if self._node_filter is not None and node != self._node_filter:
            return True

        ns_key = "|".join(ns) if ns else ""
        event_type = data.get("event") if isinstance(data, dict) else None

        if event_type == "message-start":
            stream = self._stream_cls(
                namespace=ns,
                node=node,
                message_id=data.get("message_id"),
            )
            self._active[ns_key] = stream
            self._messages_log.append(stream)

        elif event_type in ("content-block-delta", "content-block-start"):
            active = self._active.get(ns_key)
            if active is not None and event_type == "content-block-delta":
                active._push_content_block_delta(data)

        elif event_type == "content-block-finish":
            active = self._active.get(ns_key)
            if active is not None:
                active._push_content_block_finish(data)

        elif event_type == "message-finish":
            active = self._active.pop(ns_key, None)
            if active is not None:
                active._finish(data)

        elif event_type == "error":
            active = self._active.pop(ns_key, None)
            if active is not None:
                msg = data.get("message", "Unknown error")
                active._fail(RuntimeError(msg))

        return True

    def finalize(self) -> None:
        # Close any remaining active streams
        for stream in self._active.values():
            stream._finish({"reason": "stop"})
        self._active.clear()
        self._messages_log.close()

    def fail(self, err: BaseException) -> None:
        for stream in self._active.values():
            stream._fail(err)
        self._active.clear()
        self._messages_log.fail(err)


__all__ = [
    "MessagesTransformer",
    "ValuesTransformer",
]
