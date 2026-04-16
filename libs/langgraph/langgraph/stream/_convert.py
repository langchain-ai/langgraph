from __future__ import annotations

import time
from typing import Any

from langgraph.stream._types import ProtocolEvent, _ProtocolEventParams


def convert_to_protocol_event(part: dict[str, Any]) -> ProtocolEvent:
    """Convert a v2 StreamPart dict to a ProtocolEvent.

    Args:
        part: A stream part dict with keys `type`, `ns`, `data`, and
            optionally `interrupts` (present on values events).

    Returns:
        The equivalent ProtocolEvent.
    """
    params: _ProtocolEventParams = {
        "namespace": list(part["ns"]),
        "timestamp": int(time.time() * 1000),
        "data": part["data"],
    }
    if "interrupts" in part:
        params["interrupts"] = part["interrupts"]
    return {
        "type": "event",
        "method": part["type"],
        "params": params,
    }
