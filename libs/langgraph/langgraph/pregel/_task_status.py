"""Read the status of each task from the writes recorded for a superstep.

While a superstep is open, the checkpointer keeps a log of writes for each
task in that step. Entries are added as tasks run and are only discarded when
the whole superstep finishes and a new checkpoint is saved. When a task runs
again, for example after being resumed, its earlier entries stay in the log.

This module is the single place that turns that log into task status. Code that
needs to know whether a task finished, which interrupts it raised, which of them
are still waiting for an answer, or which output it produced must use
`read_task_statuses` instead of inspecting the writes directly.

The log uses two kinds of writes:

- Control writes describe what happened to a task: `INTERRUPT` (the task asked
  a question), `RESUME` (answers the task has received), `ERROR`, and
  `ERROR_SOURCE_NODE`. `INTERRUPT`, `RESUME` and `ERROR` each have a fixed slot
  per task (`WRITES_IDX_MAP`), so a newer write of the same kind can replace an
  older one.
- Every other write is output: channel writes, `RETURN` for functional tasks,
  and the `NO_WRITES` marker.

The rules are:

1. When a task that ran finishes successfully, `PregelRunner.commit` records at
   least one output write, adding `NO_WRITES` if the task produced no other
   output.
2. A task that pauses at an interrupt records only control writes.
3. A task is therefore treated as finished if and only if it has an output
   write.
4. Because `INTERRUPT` is stored in a fixed slot, its recorded value is the most
   recent question the task asked. That question is waiting for an answer only
   while the task is unfinished.

A `RESUME` write never means a task is finished: it can hold the answer to an
earlier question while the task waits on a later one.

What these rules cannot see:

- A task whose result came from the cache does not go through
  `PregelRunner.commit`, so nothing is recorded for it. It reads as not
  finished.
- A task that fails can record partial output writes along with its error. It
  reads as finished, which is how the executor has always treated it.
- Writes recorded before rule 1 existed may describe a finished task with no
  output using only control writes. Those tasks read as unfinished. The
  executor treated them the same way before and runs them again. Their last
  interrupt now counts as pending, so they appear in `next` and a resume
  without an interrupt id raises if another interrupt is also pending.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Any

from langgraph.checkpoint.base import PendingWrite

from langgraph._internal._constants import (
    ERROR,
    ERROR_SOURCE_NODE,
    INTERRUPT,
    NULL_TASK_ID,
    RESUME,
)
from langgraph.types import Interrupt

__all__ = ("CONTROL_WRITES", "TaskStatus", "read_task_statuses")

CONTROL_WRITES = frozenset((ERROR, ERROR_SOURCE_NODE, INTERRUPT, RESUME))
"""Channels that describe what happened to a task rather than what it produced."""


@dataclass(frozen=True, slots=True)
class TaskStatus:
    """The status of one task, read from the writes recorded for its superstep."""

    output: tuple[tuple[str, Any], ...] = ()
    """Output writes in recorded order. Empty if the task has not finished."""

    interrupts: tuple[Interrupt, ...] = ()
    """The most recent interrupts the task raised, whether or not they were answered."""

    error: BaseException | None = None
    """The recorded error, if any."""

    @property
    def finished(self) -> bool:
        """Whether the task ran to completion."""
        return bool(self.output)

    @property
    def pending_interrupts(self) -> tuple[Interrupt, ...]:
        """Interrupts waiting for an answer. Always empty for a finished task."""
        return () if self.finished else self.interrupts


def read_task_statuses(
    pending_writes: Iterable[PendingWrite],
) -> dict[str, TaskStatus]:
    """Return the status of every task that has recorded writes, keyed by task id.

    Writes from `NULL_TASK_ID` are input to the superstep, not task activity, so
    they are not included.
    """
    output: dict[str, list[tuple[str, Any]]] = {}
    interrupts: dict[str, list[Interrupt]] = {}
    errors: dict[str, BaseException] = {}
    for task_id, channel, value in pending_writes:
        if task_id == NULL_TASK_ID:
            continue
        output.setdefault(task_id, [])
        if channel == INTERRUPT:
            interrupts.setdefault(task_id, []).extend(
                value if isinstance(value, Sequence) else [value]
            )
        elif channel == ERROR:
            errors.setdefault(task_id, value)
        elif channel not in CONTROL_WRITES:
            output[task_id].append((channel, value))
    return {
        task_id: TaskStatus(
            output=tuple(task_output),
            interrupts=tuple(interrupts.get(task_id, ())),
            error=errors.get(task_id),
        )
        for task_id, task_output in output.items()
    }
