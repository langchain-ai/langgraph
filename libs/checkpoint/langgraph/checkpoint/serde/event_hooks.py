from __future__ import annotations

import logging
from collections.abc import Callable
from threading import Lock
from typing import TypedDict

from typing_extensions import NotRequired

logger = logging.getLogger(__name__)


class SerdeEvent(TypedDict):
    """Event emitted during serialization or deserialization.

    Used by the allowlist system to notify listeners when types are
    encountered during msgpack deserialization.
    """

    kind: str
    """The event kind, e.g. `"msgpack_blocked"` or `"msgpack_unregistered_allowed"`."""
    module: str
    """The module path of the type that triggered the event."""
    name: str
    """The class name of the type that triggered the event."""
    method: NotRequired[str]
    """The method name, if the event involves a method invocation."""


SerdeEventListener = Callable[[SerdeEvent], None]

_listeners: list[SerdeEventListener] = []
_listeners_lock = Lock()


def register_serde_event_listener(listener: SerdeEventListener) -> Callable[[], None]:
    """Register a listener for serde allowlist events."""
    with _listeners_lock:
        _listeners.append(listener)

    def unregister() -> None:
        with _listeners_lock:
            try:
                _listeners.remove(listener)
            except ValueError:
                pass

    return unregister


def emit_serde_event(event: SerdeEvent) -> None:
    """Emit a serde event to all listeners.

    Listener failures are isolated and logged.
    """
    with _listeners_lock:
        listeners = tuple(_listeners)
    for listener in listeners:
        try:
            listener(event)
        except Exception:
            logger.warning("Serde listener failed", exc_info=True)
