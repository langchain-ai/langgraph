"""Boundary-level progress events for service capabilities (not internal node xray)."""

from __future__ import annotations

from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Generic, TypeVar

from langgraph.capability.service import ServiceRunResult, ServiceRunStatus

OutputT = TypeVar("OutputT")


class ProgressPhase(str, Enum):
    """Coarse phases suitable for black-box service observability."""

    ACCEPTED = "accepted"
    RUNNING = "running"
    STREAMING = "streaming"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    INTERRUPTED = "interrupted"
    TIMED_OUT = "timed_out"


@dataclass(frozen=True)
class CapabilityProgressEvent:
    """One boundary progress event (safe to expose across org/service boundaries)."""

    capability_id: str
    version: str
    run_id: str
    phase: ProgressPhase
    message: str = ""
    percent: float | None = None
    """Optional 0-100 progress hint; providers may omit."""

    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "type": "capability_progress",
            "capability_id": self.capability_id,
            "version": self.version,
            "run_id": self.run_id,
            "phase": self.phase.value,
            "message": self.message,
            "percent": self.percent,
            "metadata": self.metadata,
        }


def progress_events_from_run_result(
    result: ServiceRunResult[Any],
    /,
    *,
    include_accepted: bool = True,
) -> list[CapabilityProgressEvent]:
    """Map a completed service run into a minimal progress timeline."""
    events: list[CapabilityProgressEvent] = []
    base = dict(
        capability_id=result.capability_id,
        version=result.version,
        run_id=result.run_id,
        metadata=dict(result.metadata),
    )
    if include_accepted:
        events.append(
            CapabilityProgressEvent(**base, phase=ProgressPhase.ACCEPTED, percent=0.0)
        )
        events.append(
            CapabilityProgressEvent(**base, phase=ProgressPhase.RUNNING, percent=None)
        )

    status_map = {
        ServiceRunStatus.SUCCEEDED: ProgressPhase.SUCCEEDED,
        ServiceRunStatus.FAILED: ProgressPhase.FAILED,
        ServiceRunStatus.INTERRUPTED: ProgressPhase.INTERRUPTED,
        ServiceRunStatus.TIMED_OUT: ProgressPhase.TIMED_OUT,
        ServiceRunStatus.RUNNING: ProgressPhase.RUNNING,
        ServiceRunStatus.QUEUED: ProgressPhase.ACCEPTED,
    }
    phase = status_map.get(result.status, ProgressPhase.FAILED)
    percent = 100.0 if phase is ProgressPhase.SUCCEEDED else None
    events.append(
        CapabilityProgressEvent(
            **base,
            phase=phase,
            message=result.error_message or "",
            percent=percent,
        )
    )
    return events


def iter_progress_dicts(result: ServiceRunResult[Any]) -> Iterator[dict[str, Any]]:
    for ev in progress_events_from_run_result(result):
        yield ev.to_dict()


ProgressCallback = Callable[[CapabilityProgressEvent], None]


def emit_run_progress(
    result: ServiceRunResult[OutputT],
    callback: ProgressCallback,
    /,
) -> ServiceRunResult[OutputT]:
    """Invoke a callback for each synthesized progress event; returns the same result."""
    for ev in progress_events_from_run_result(result):
        callback(ev)
    return result
