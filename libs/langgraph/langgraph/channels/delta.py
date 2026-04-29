from __future__ import annotations

import copy as _copy
from collections.abc import Callable, Sequence
from typing import Any, Generic

from langgraph.checkpoint.base import DELTA_SENTINEL, PendingWrite
from langgraph.checkpoint.serde.types import _DeltaSnapshot
from typing_extensions import Self

from langgraph._internal._typing import MISSING
from langgraph.channels.base import BaseChannel, Value
from langgraph.channels.binop import _get_overwrite, _operators_equal
from langgraph.errors import (
    EmptyChannelError,
    ErrorCode,
    InvalidUpdateError,
    create_error_message,
)

__all__ = ("DeltaChannel",)


def _empty(typ: Any) -> Any:
    try:
        return typ()
    except Exception:
        return []


class DeltaChannel(Generic[Value], BaseChannel[Any, Any, Any]):
    """Fold-reducer channel with configurable snapshot cadence.

    `snapshot_frequency=None` (default): pure delta — stores only
    `DELTA_SENTINEL` in checkpoint blobs; reads replay all ancestor writes.

    `snapshot_frequency=N`: pregel's `create_checkpoint` writes a full
    `_DeltaSnapshot` blob every N steps (eagerly, even if the channel had
    no write that step). Reads walk at most N ancestor checkpoints before
    hitting the snapshot, bounding replay depth to N regardless of thread
    length.

    Parameters:
        operator: Binary reducer `(Value, Value) -> Value`.
        snapshot_frequency: Every Nth pregel step writes a snapshot blob.
            `None` (default) = pure delta, never snapshot.
    """

    __slots__ = ("value", "operator", "snapshot_frequency")
    value: Value | Any

    def __init__(
        self,
        operator: Callable[[Any, Any], Any],
        *,
        snapshot_frequency: int | None = None,
    ) -> None:
        super().__init__(list)
        self.operator = operator
        self.snapshot_frequency = snapshot_frequency
        self.value: Any = MISSING

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, DeltaChannel):
            return False
        if self.snapshot_frequency != other.snapshot_frequency:
            return False
        return _operators_equal(self.operator, other.operator)

    @property
    def ValueType(self) -> Any:
        return self.typ

    @property
    def UpdateType(self) -> Any:
        return self.typ

    def is_snapshot_step(self, step: int) -> bool:
        """True if pregel should write a snapshot blob at this step."""
        return (
            self.snapshot_frequency is not None
            and step > 0
            and step % self.snapshot_frequency == 0
        )

    def _clone_empty(self) -> Self:
        new = self.__class__(self.operator, snapshot_frequency=self.snapshot_frequency)
        new.typ = self.typ  # typ may differ from list when set via Annotated injection
        new.key = self.key  # key is injected externally by the graph builder
        return new

    def copy(self) -> Self:
        new = self._clone_empty()
        new.value = self.value if self.value is MISSING else _copy.copy(self.value)
        return new

    def _apply_write(self, value: Any, write: Any) -> Any:
        is_overwrite, overwrite_value = _get_overwrite(write)
        if is_overwrite:
            return (
                _copy.copy(overwrite_value)
                if overwrite_value is not None
                else _empty(self.typ)
            )
        base = _empty(self.typ) if value is MISSING else value
        return self.operator(base, write)

    def from_checkpoint(self, checkpoint: Any) -> Self:
        """Initialize from a stored blob or sentinel.

        Blob types (dispatched via serde ext code, not dict key inspection):
          * `DELTA_SENTINEL` / `MISSING`: start empty; caller replays writes.
          * `_DeltaSnapshot(value)`: restore value directly from snapshot.
          * plain value (migration from old BinOp blobs): use directly.
        """
        new = self._clone_empty()
        if checkpoint is MISSING or checkpoint is DELTA_SENTINEL:
            new.value = _empty(new.typ)
        elif isinstance(checkpoint, _DeltaSnapshot):
            new.value = checkpoint.value
        else:
            new.value = checkpoint
        return new

    def replay_writes(self, writes: Sequence[PendingWrite]) -> None:
        """Fold ancestor writes oldest→newest into current value."""
        for _, _, value in writes:
            self.value = self._apply_write(self.value, value)

    def update(self, values: Sequence[Any]) -> bool:
        if not values:
            return False
        overwrite_idx: int | None = None
        for i, v in enumerate(values):
            is_ow, _ = _get_overwrite(v)
            if is_ow:
                if overwrite_idx is not None:
                    msg = create_error_message(
                        message="Can receive only one Overwrite value per super-step.",
                        error_code=ErrorCode.INVALID_CONCURRENT_GRAPH_UPDATE,
                    )
                    raise InvalidUpdateError(msg)
                overwrite_idx = i
        if overwrite_idx is not None:
            self.value = self._apply_write(self.value, values[overwrite_idx])
            return True
        for value in values:
            self.value = self._apply_write(self.value, value)
        return True

    def get(self) -> Any:
        if self.value is MISSING:
            raise EmptyChannelError()
        return self.value

    def is_available(self) -> bool:
        return self.value is not MISSING

    def checkpoint(self) -> Any:
        """Return stored representation: always `DELTA_SENTINEL`.

        Snapshot decisions are made by `create_checkpoint` in pregel (which
        has the step number) via `is_snapshot_step`. `checkpoint()` is only
        called for non-snapshot steps or when no checkpointer is available.
        """
        if self.value is MISSING:
            return MISSING
        return DELTA_SENTINEL
