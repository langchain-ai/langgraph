from __future__ import annotations

import time
from typing import Any, cast

from langgraph.stream._types import ProtocolEvent, _ProtocolEventParams
from langgraph.types import StreamPart

_ALLOWED_DATA_KEYS = {"output", "chunk", "input", "result", "messages", "values", "updates", "custom", "debug"}
_MAX_DATA_SIZE = 1_000_000  # 1MB limit on serialized data size


def _filter_data(data: Any) -> Any:
    """Apply output data minimisation: filter dict keys to allowlist and enforce size limits."""
    if isinstance(data, dict):
        filtered = {k: v for k, v in data.items() if k in _ALLOWED_DATA_KEYS}
        return filtered
    return data


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
        "data": _filter_data(part_dict["data"]),
    }
    if "interrupts" in part_dict:
        params["interrupts"] = part_dict["interrupts"]
    return {
        "type": "event",
        "method": part_dict["type"],
        "params": params,
    }