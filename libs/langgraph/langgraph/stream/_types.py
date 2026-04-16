from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Literal

from typing_extensions import NotRequired, TypedDict


class _ProtocolEventParams(TypedDict):
    """Parameters for a protocol event."""

    namespace: list[str]
    timestamp: int
    data: Any
    interrupts: NotRequired[tuple[Any, ...]]


class ProtocolEvent(TypedDict):
    """A protocol event emitted by the streaming infrastructure.

    Wraps a raw stream part (values, messages, custom, etc.) in a uniform
    envelope with a monotonic sequence number assigned by the StreamMux.
    """

    type: Literal["event"]
    eventId: NotRequired[str]
    seq: NotRequired[int]
    method: str  # StreamMode value: "values", "messages", "custom", etc.
    params: _ProtocolEventParams


class StreamTransformer(ABC):
    """Extension point for custom stream projections.

    Transformers observe protocol events flowing through the StreamMux and
    build typed derived projections (EventLogs, StreamChannels, promises, etc.).

    Set `_native = True` on a transformer to have its projection keys
    exposed as direct attributes on the run stream (in addition to
    appearing in `run.extensions`).

    Subclasses must implement `init` and `process`. The `finalize` and
    `fail` hooks are optional — the default implementations are no-ops.
    EventLog and StreamChannel instances in the projection dict are
    auto-closed/failed by the mux, so most transformers don't need
    ``finalize`` or ``fail`` at all.
    """

    @abstractmethod
    def init(self) -> dict[str, Any]:
        """Return the projection dict.

        Keys become entries in `run.extensions`. If the transformer has
        `_native = True`, keys are also set as direct attributes on the
        run stream.

        StreamChannel instances in the return value are automatically
        wired by the StreamMux for protocol event auto-forwarding.
        """
        ...

    @abstractmethod
    def process(self, event: ProtocolEvent) -> bool:
        """Process a protocol event.

        Called for every event before it is appended to the main event log.
        Return False to suppress the event from the main log.
        """
        ...

    def finalize(self) -> None:
        """Called when the run ends normally.

        Override to close EventLogs, resolve promises, or perform other
        teardown. StreamChannel instances are auto-closed by the mux.
        """

    def fail(self, err: BaseException) -> None:
        """Called when the run ends with an error.

        Override to fail EventLogs, reject promises, or perform other
        teardown. StreamChannel instances are auto-failed by the mux.
        """
