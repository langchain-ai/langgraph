from __future__ import annotations

from datetime import timedelta


def _coerce_timeout_seconds(value: float | timedelta | None) -> float | None:
    if value is None:
        return None
    return value.total_seconds() if isinstance(value, timedelta) else float(value)


def validate_timeout(value: float | timedelta | None) -> float | timedelta | None:
    timeout = _coerce_timeout_seconds(value)
    if timeout is not None and timeout <= 0:
        raise ValueError("timeout must be greater than 0")
    return value


def timeout_seconds(value: float | timedelta | None) -> float | None:
    validate_timeout(value)
    return _coerce_timeout_seconds(value)
