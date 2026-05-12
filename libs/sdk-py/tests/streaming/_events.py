"""Builders for protocol `Event` payloads used in tests.

Mirrors `libs/sdk/src/client/stream/test/event-builders.ts` from the JS SDK.
"""

from __future__ import annotations

from typing import Any


def _base(seq: int, method: str, namespace: list[str], data: Any) -> dict[str, Any]:
    return {
        "method": method,
        "params": {
            "namespace": namespace,
            "data": data,
        },
        "seq": seq,
        "id": f"evt-{seq}",
    }


def lifecycle_event(
    seq: int = 0, namespace: list[str] | None = None, **data: Any
) -> dict[str, Any]:
    return _base(seq, "lifecycle", namespace or [], data or {"phase": "started"})


def values_event(
    seq: int = 0, namespace: list[str] | None = None, **data: Any
) -> dict[str, Any]:
    return _base(seq, "values", namespace or [], data or {"values": {}})


def custom_event(
    seq: int = 0, name: str = "ext", namespace: list[str] | None = None, **data: Any
) -> dict[str, Any]:
    payload = {"name": name, **data} if name else dict(data)
    return _base(seq, "custom", namespace or [], payload)


def input_requested_event(
    seq: int = 0, namespace: list[str] | None = None
) -> dict[str, Any]:
    return _base(seq, "input.requested", namespace or [], {"interrupt_id": "i-1"})
