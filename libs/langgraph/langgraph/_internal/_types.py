from __future__ import annotations

from typing import Any

from langgraph._internal._constants import (
    CONFIG_KEY_SCRATCHPAD,
    CONFIG_KEY_SEND,
    RESUME,
)


def _get_resume_value(conf: dict[str, Any]) -> tuple[bool, Any | None]:
    """Return resume value if present; mutates scratchpad and emits RESUME writes."""
    # track interrupt index
    scratchpad = conf[CONFIG_KEY_SCRATCHPAD]
    idx = scratchpad.interrupt_counter()
    # find previous resume values
    if scratchpad.resume:
        if idx < len(scratchpad.resume):
            conf[CONFIG_KEY_SEND]([(RESUME, scratchpad.resume)])
            return True, scratchpad.resume[idx]
    # find current resume value
    v = scratchpad.get_null_resume(True)
    if v is not None:
        if len(scratchpad.resume) != idx:
            raise RuntimeError(
                f"Resume index mismatch: expected {idx}, got {len(scratchpad.resume)}"
            )
        scratchpad.resume.append(v)
        conf[CONFIG_KEY_SEND]([(RESUME, scratchpad.resume)])
        return True, v
    return False, None
