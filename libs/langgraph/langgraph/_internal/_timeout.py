from __future__ import annotations

from datetime import timedelta
from typing import Literal

_SYNC_TIMEOUT_PREFIX = (
    "Node timeouts are only supported for async nodes because sync Python "
    "execution cannot be safely cancelled in-process."
)


def coerce_timeout(value: float | timedelta | None) -> float | None:
    """Normalize a timeout to positive seconds, or None if unset."""
    if value is None:
        return None
    seconds = value.total_seconds() if isinstance(value, timedelta) else float(value)
    if seconds <= 0:
        raise ValueError("timeout must be greater than 0")
    return seconds


def sync_timeout_unsupported(
    name: str, *, kind: Literal["Node", "Task"] = "Node"
) -> ValueError:
    """Build the canonical error for using `timeout` with a sync target."""
    return ValueError(f"{_SYNC_TIMEOUT_PREFIX} {kind} {name!r} is sync.")
