from __future__ import annotations

from typing import Any

from langgraph.stream._event_log import EventLog
from langgraph.stream._types import ProtocolEvent, StreamTransformer


class ValuesTransformer(StreamTransformer):
    """Captures values events and projects them into an iterable of state snapshots.

    Native transformer — projection keys are exposed as direct attributes
    on the run stream (e.g. ``run.values``).
    """

    _native = True

    def __init__(self) -> None:
        self._log: EventLog[dict[str, Any]] = EventLog()
        self._latest: dict[str, Any] | None = None
        self._interrupted = False
        self._interrupts: list[Any] = []

    def init(self) -> dict[str, Any]:
        return {"values": self._log}

    def process(self, event: ProtocolEvent) -> bool:
        if event["method"] != "values":
            return True
        params = event["params"]
        # Only capture root namespace events
        if params["namespace"]:
            return True
        self._latest = params["data"]
        self._log.push(params["data"])
        interrupts = params.get("interrupts", ())
        if interrupts:
            self._interrupted = True
            self._interrupts.extend(interrupts)
        return True


class MessagesTransformer(StreamTransformer):
    """Captures messages events and passes through raw (chunk, metadata) tuples.

    This is the same shape as today's ``stream_mode="messages"`` output.
    A follow-on PR will replace this with a richer transformer that
    produces ChatModelStream objects using the protocol handler.

    Native transformer — projection keys are exposed as direct attributes
    on the run stream (e.g. ``run.messages``).
    """

    _native = True

    def __init__(self) -> None:
        self._log: EventLog[tuple[Any, dict[str, Any]]] = EventLog()

    def init(self) -> dict[str, Any]:
        return {"messages": self._log}

    def process(self, event: ProtocolEvent) -> bool:
        if event["method"] != "messages":
            return True
        params = event["params"]
        # Only capture root namespace events
        if params["namespace"]:
            return True
        self._log.push(params["data"])
        return True
