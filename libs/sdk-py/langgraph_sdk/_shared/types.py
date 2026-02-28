"""Type aliases and constants."""

from __future__ import annotations

TimeoutTypes = (
    None
    | float
    | tuple[float | None, float | None]
    | tuple[float | None, float | None, float | None, float | None]
)
