from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from langgraph.types import Interrupt


@dataclass(slots=True, frozen=True)
class GraphLifecycleEvent:
    """Lifecycle transition captured during graph execution.

    Events are queued in `PregelLoop` and drained later by the runtime when
    dispatching graph-level callbacks.
    """

    kind: Literal["resume", "interrupt"]
    """Lifecycle transition kind."""

    status: str
    """Loop status when the event was captured."""

    checkpoint_id: str
    """Checkpoint id associated with this transition."""

    checkpoint_ns: tuple[str, ...]
    """Checkpoint namespace path for the current graph/subgraph."""

    is_nested: bool
    """Whether the event originated from a nested graph invocation."""

    interrupts: tuple[Interrupt, ...] = ()
    """Interrupt payloads (populated for `kind="interrupt"`)."""
