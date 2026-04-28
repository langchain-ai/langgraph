from __future__ import annotations

from datetime import timedelta
from typing import Literal

from langgraph.types import TimeoutPolicy

_SYNC_TIMEOUT_PREFIX = (
    "Node timeouts are only supported for async nodes because sync Python "
    "execution cannot be safely cancelled in-process."
)


def _coerce_timeout_seconds(
    value: float | timedelta | None, *, field: str
) -> float | None:
    if value is None:
        return None
    seconds = value.total_seconds() if isinstance(value, timedelta) else float(value)
    if seconds <= 0:
        raise ValueError(f"{field} must be greater than 0")
    return seconds


def coerce_timeout_policy(
    value: float | timedelta | TimeoutPolicy | None,
) -> TimeoutPolicy | None:
    """Normalize a timeout value to positive-second policy fields."""
    if value is not None and not isinstance(value, TimeoutPolicy):
        value = TimeoutPolicy(run_timeout=value)
    if value is None:
        return None
    if value.refresh_on not in ("auto", "heartbeat"):
        raise ValueError("refresh_on must be 'auto' or 'heartbeat'")
    run_timeout = _coerce_timeout_seconds(value.run_timeout, field="run_timeout")
    idle_timeout_s = _coerce_timeout_seconds(value.idle_timeout, field="idle_timeout")
    if run_timeout is None and idle_timeout_s is None:
        return None
    return TimeoutPolicy(
        run_timeout=run_timeout,
        idle_timeout=idle_timeout_s,
        refresh_on=value.refresh_on,
    )


def sync_timeout_unsupported(
    name: str, *, kind: Literal["Node", "Task"] = "Node"
) -> ValueError:
    """Build the canonical error for using `timeout` with a sync target."""
    return ValueError(f"{_SYNC_TIMEOUT_PREFIX} {kind} {name!r} is sync.")
