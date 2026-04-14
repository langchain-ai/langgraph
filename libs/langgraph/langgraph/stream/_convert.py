"""Convert raw ``StreamChunk`` tuples to ``ProtocolEvent`` envelopes.

Each ``StreamMode`` is mapped to a ``ProtocolEvent`` whose ``method``
field matches the mode name and whose ``params.data`` wraps the
original payload.
"""

from __future__ import annotations

from typing import Any

from langgraph.stream._types import ProtocolEvent, _ProtocolEventParams
from langgraph.types import StreamMode

#: All stream modes requested by ``StreamingHandler`` when calling the
#: underlying ``stream()`` / ``astream()``.
STREAM_V2_MODES: list[StreamMode] = [
    "values",
    "updates",
    "messages",
    "custom",
    "checkpoints",
    "tasks",
    "debug",
]

_SUPPORTED_MODES: set[str] = set(STREAM_V2_MODES)


def convert_to_protocol_event(
    ns: tuple[str, ...],
    mode: str,
    payload: Any,
    *,
    node: str | None = None,
) -> ProtocolEvent | None:
    """Convert a ``StreamChunk`` to a ``ProtocolEvent``.

    Returns ``None`` for unsupported or unknown modes.

    The ``seq`` field is left as ``0`` here; the :class:`StreamMux` is
    the sole seq assigner and overwrites it inside ``push()``.

    Parameters
    ----------
    ns:
        Namespace tuple from the ``StreamChunk``.
    mode:
        Stream mode string (``"values"``, ``"updates"``, etc.).
    payload:
        The raw payload from the stream.
    node:
        Optional node name for provenance.
    """
    if mode not in _SUPPORTED_MODES:
        return None

    params: _ProtocolEventParams = {
        "namespace": list(ns),
        "data": payload,
    }
    if node is not None:
        params["node"] = node

    return ProtocolEvent(
        type="event",
        method=mode,
        params=params,
    )


__all__ = ["STREAM_V2_MODES", "convert_to_protocol_event"]
