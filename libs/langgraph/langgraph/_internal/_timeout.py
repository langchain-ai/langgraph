from __future__ import annotations

from datetime import timedelta
from typing import Literal

from langgraph.types import TimeoutPolicy

_SYNC_TIMEOUT_PREFIX = (
    "Node timeouts are only supported for async nodes because sync Python "
    "execution cannot be safely cancelled in-process."
)


def coerce_timeout_policy(
    value: float | timedelta | TimeoutPolicy | None,
) -> TimeoutPolicy | None:
    """Normalize a timeout value to positive-second policy fields."""
    return TimeoutPolicy.coerce(value)


def sync_timeout_unsupported(
    name: str, *, kind: Literal["Node", "Task"] = "Node"
) -> ValueError:
    """Build the canonical error for using `timeout` with a sync target."""
    return ValueError(f"{_SYNC_TIMEOUT_PREFIX} {kind} {name!r} is sync.")
