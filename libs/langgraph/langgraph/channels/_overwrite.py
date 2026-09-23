from __future__ import annotations

from typing import Any

from langgraph._internal._constants import OVERWRITE
from langgraph.types import Overwrite


def get_overwrite(value: Any) -> tuple[bool, Any]:
    """Inspect a channel write and return ``(is_overwrite, overwrite_value)``.

    Recognizes the typed dataclass, its sentinel-keyed dict form, and the
    dataclass-erased dict form that can appear after JSON serialization.
    """
    if isinstance(value, Overwrite):
        return True, value.value
    if isinstance(value, dict):
        if len(value) == 1 and OVERWRITE in value:
            return True, value[OVERWRITE]
        if value.get("type") == OVERWRITE and "value" in value:
            return True, value["value"]
    return False, None
