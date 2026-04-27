from __future__ import annotations

import time
from typing import Any, cast

from langgraph.stream._types import ProtocolEvent, _ProtocolEventParams
from langgraph.types import StreamPart


def _is_v2_messages_payload(data: Any) -> bool:
    return isinstance(data, dict) and isinstance(data.get("event"), str)


def _normalize_messages_data(data: dict[str, Any]) -> dict[str, Any]:
    """Normalize Python Core message fields to the protocol wire shape."""
    normalized = {**data}
    if (
        normalized["event"] == "message-start"
        and "id" not in normalized
        and isinstance(normalized.get("message_id"), str)
    ):
        normalized["id"] = normalized["message_id"]
    if (
        normalized["event"]
        in ("content-block-start", "content-block-delta", "content-block-finish")
        and "content" not in normalized
        and isinstance(normalized.get("content_block"), dict)
    ):
        normalized["content"] = normalized["content_block"]
    normalized.pop("message_id", None)
    normalized.pop("content_block", None)
    return normalized


def convert_to_protocol_event(part: StreamPart) -> ProtocolEvent:
    """Convert a v2 StreamPart to a ProtocolEvent.

    Args:
        part: A stream part with keys `type`, `ns`, `data`, and
            optionally `interrupts` (present on values events).

    Returns:
        The equivalent ProtocolEvent.
    """
    part_dict = cast(dict[str, Any], part)
    data = part_dict["data"]
    params: _ProtocolEventParams = {
        "namespace": list(part_dict["ns"]),
        "timestamp": int(time.time() * 1000),
        "data": data,
    }
    if (
        part_dict["type"] == "messages"
        and isinstance(data, tuple)
        and len(data) == 2
        and _is_v2_messages_payload(data[0])
        and isinstance(data[1], dict)
    ):
        payload, metadata = data
        params["data"] = _normalize_messages_data(payload)
        if isinstance(metadata.get("langgraph_node"), str):
            params["node"] = metadata["langgraph_node"]
        if isinstance(metadata.get("run_id"), str):
            params["run_id"] = metadata["run_id"]
    if "interrupts" in part_dict:
        params["interrupts"] = part_dict["interrupts"]
    return {
        "type": "event",
        "method": part_dict["type"],
        "params": params,
    }
