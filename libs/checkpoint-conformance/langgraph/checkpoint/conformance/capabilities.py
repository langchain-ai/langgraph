"""Capability detection for checkpointer implementations."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING

from langgraph.checkpoint.base import BaseCheckpointSaver

if TYPE_CHECKING:
    pass


class Capability(str, Enum):
    """Capabilities that a checkpointer may support."""

    PUT = "put"
    PUT_WRITES = "put_writes"
    GET_TUPLE = "get_tuple"
    LIST = "list"
    DELETE_THREAD = "delete_thread"
    DELETE_FOR_RUNS = "delete_for_runs"
    COPY_THREAD = "copy_thread"
    PRUNE = "prune"


# Capabilities that every checkpointer must support.
BASE_CAPABILITIES = frozenset(
    {
        Capability.PUT,
        Capability.PUT_WRITES,
        Capability.GET_TUPLE,
        Capability.LIST,
        Capability.DELETE_THREAD,
    }
)

# Capabilities that are optional extensions.
EXTENDED_CAPABILITIES = frozenset(
    {
        Capability.DELETE_FOR_RUNS,
        Capability.COPY_THREAD,
        Capability.PRUNE,
    }
)

ALL_CAPABILITIES = BASE_CAPABILITIES | EXTENDED_CAPABILITIES

# Maps capability to the async method name on BaseCheckpointSaver (or subclass).
_CAPABILITY_METHOD_MAP: dict[Capability, str] = {
    Capability.PUT: "aput",
    Capability.PUT_WRITES: "aput_writes",
    Capability.GET_TUPLE: "aget_tuple",
    Capability.LIST: "alist",
    Capability.DELETE_THREAD: "adelete_thread",
    Capability.DELETE_FOR_RUNS: "adelete_for_runs",
    Capability.COPY_THREAD: "acopy_thread",
    Capability.PRUNE: "aprune",
}


@dataclass(frozen=True)
class DetectedCapabilities:
    """Result of capability detection for a checkpointer type."""

    detected: frozenset[Capability]
    missing: frozenset[Capability]

    @classmethod
    def from_instance(cls, saver: BaseCheckpointSaver) -> DetectedCapabilities:
        """Detect capabilities from a checkpointer instance."""
        inner_type = type(saver)
        detected: set[Capability] = set()

        for cap, method_name in _CAPABILITY_METHOD_MAP.items():
            if _is_overridden(inner_type, method_name):
                detected.add(cap)

        detected_fs = frozenset(detected)
        return cls(
            detected=detected_fs,
            missing=ALL_CAPABILITIES - detected_fs,
        )


def _is_overridden(inner_type: type, method: str) -> bool:
    """Check if *method* on *inner_type* differs from the base class default."""
    base = getattr(BaseCheckpointSaver, method, None)
    impl = getattr(inner_type, method, None)
    if base is None or impl is None:
        return impl is not None
    return impl is not base
