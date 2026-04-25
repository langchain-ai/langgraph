from __future__ import annotations

from datetime import timedelta
from typing import Literal

_SYNC_IDLE_TIMEOUT_PREFIX = (
    "Node idle timeouts are only supported for async nodes because sync Python "
    "execution cannot be safely cancelled in-process."
)


def coerce_idle_timeout(value: float | timedelta | None) -> float | None:
    """Normalize an idle timeout to positive seconds, or None if unset."""
    if value is None:
        return None
    seconds = value.total_seconds() if isinstance(value, timedelta) else float(value)
    if seconds <= 0:
        raise ValueError("idle_timeout must be greater than 0")
    return seconds


def sync_idle_timeout_unsupported(
    name: str, *, kind: Literal["Node", "Task"] = "Node"
) -> ValueError:
    """Build the canonical error for using `idle_timeout` with a sync target."""
    return ValueError(f"{_SYNC_IDLE_TIMEOUT_PREFIX} {kind} {name!r} is sync.")
