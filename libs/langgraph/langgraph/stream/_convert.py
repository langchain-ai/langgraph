from __future__ import annotations

from typing import Any

from langgraph.stream._types import ProtocolEvent, _ProtocolEventParams


def convert_to_protocol_event(part: dict[str, Any]) -> ProtocolEvent:
    """Convert a v2 StreamPart dict to a ProtocolEvent.

    Expects a dict with keys ``type``, ``ns``, ``data``, and optionally
    ``interrupts`` (present on values events).
    """
    params: _ProtocolEventParams = {
        "namespace": list(part["ns"]),
        "data": part["data"],
    }
    if "interrupts" in part:
        params["interrupts"] = part["interrupts"]
    return {
        "type": "event",
        "method": part["type"],
        "params": params,
    }
