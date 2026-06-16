from __future__ import annotations

import time
from typing import Any, cast

from langgraph.stream._types import ProtocolEvent, _ProtocolEventParams
from langgraph.types import StreamPart


def convert_to_protocol_event(part: StreamPart) -> ProtocolEvent:
    """Convert a v2 StreamPart to a ProtocolEvent.

    Args:
        part: A stream part with keys `type`, `ns`, `data`, and
            optionally `interrupts` (present on values events).

    Returns:
        The equivalent ProtocolEvent.
    """
    part_dict = cast(dict[str, Any], part)
    params: _ProtocolEventParams = {
        "namespace": list(part_dict["ns"]),
        "timestamp": int(time.time() * 1000),
        "data": part_dict["data"],
    }
    if "interrupts" in part_dict:
        params["interrupts"] = part_dict["interrupts"]
    return {
        "type": "event",
        "method": part_dict["type"],
        "params": params,
    }
