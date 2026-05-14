"""Builders for protocol `Event` payloads used in tests.

Mirrors `libs/sdk/src/client/stream/test/event-builders.ts` from the JS SDK.
"""

from __future__ import annotations

from typing import Any


def _base(seq: int, method: str, namespace: list[str], data: Any) -> dict[str, Any]:
    return {
        "type": "event",
        "method": method,
        "params": {
            "namespace": namespace,
            "data": data,
        },
        "seq": seq,
        "event_id": f"evt-{seq}",
    }


def lifecycle_event(
    seq: int = 0, namespace: list[str] | None = None, **data: Any
) -> dict[str, Any]:
    return _base(seq, "lifecycle", namespace or [], data or {"phase": "started"})


def lifecycle_started_event(
    seq: int = 0, namespace: list[str] | None = None
) -> dict[str, Any]:
    """Lifecycle event with `phase="started"`."""
    return _base(seq, "lifecycle", namespace or [], {"phase": "started"})


def lifecycle_completed_event(
    seq: int = 0, namespace: list[str] | None = None
) -> dict[str, Any]:
    """Lifecycle event with `phase="completed"`."""
    return _base(seq, "lifecycle", namespace or [], {"phase": "completed"})


def lifecycle_errored_event(
    seq: int = 0,
    namespace: list[str] | None = None,
    error: str = "run errored",
) -> dict[str, Any]:
    """Lifecycle event with `phase="errored"` and an error message."""
    return _base(
        seq, "lifecycle", namespace or [], {"phase": "errored", "error": error}
    )


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
