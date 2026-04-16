from __future__ import annotations

from typing import Any

from langgraph.stream._event_log import EventLog
from langgraph.stream._types import ProtocolEvent, StreamTransformer


class ValuesTransformer(StreamTransformer):
    """Capture values events as an iterable of state snapshots.

    Native transformer — projection keys are exposed as direct
    attributes on the run stream (e.g. `run.values`).

    Only root-namespace values events are captured; subgraph state
    snapshots are ignored.
    """

    _native = True

    def __init__(self) -> None:
        self._log: EventLog[dict[str, Any]] = EventLog()
        self._latest: dict[str, Any] | None = None
        self._interrupted = False
        self._interrupts: list[Any] = []

    def init(self) -> dict[str, Any]:
        return {"values": self._log}

    @property
    def error(self) -> BaseException | None:
        """The error that ended the run, or `None` if it succeeded.

        Set by the mux when it auto-fails the projection log.
        """
        return self._log._error

    def process(self, event: ProtocolEvent) -> bool:
        if event["method"] != "values":
            return True
        params = event["params"]
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
    """Pass through raw (chunk, metadata) tuples from messages events.

    This is the same shape as today's `stream_mode="messages"` output.
    A follow-on PR will replace this with a richer transformer that
    produces ChatModelStream objects using the protocol handler.

    Only root-namespace messages events are captured; tokens emitted
    from subgraphs are dropped from the `messages` projection. Consumers
    that need subgraph tokens should iterate the raw event stream or
    register a custom transformer.

    Native transformer — projection keys are exposed as direct
    attributes on the run stream (e.g. `run.messages`).
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
        if params["namespace"]:
            return True
        self._log.push(params["data"])
        return True
