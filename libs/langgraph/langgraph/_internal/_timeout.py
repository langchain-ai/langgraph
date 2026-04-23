from __future__ import annotations

from datetime import timedelta


def validate_timeout(value: float | timedelta | None) -> float | timedelta | None:
    if value is None:
        return None
    seconds = value.total_seconds() if isinstance(value, timedelta) else float(value)
    if seconds <= 0:
        raise ValueError("timeout must be greater than 0")
    return value


def timeout_seconds(value: float | timedelta | None) -> float | None:
    if value is None:
        return None
    seconds = value.total_seconds() if isinstance(value, timedelta) else float(value)
    if seconds <= 0:
        raise ValueError("timeout must be greater than 0")
    return seconds
