from __future__ import annotations

import copy as _copy
import math
from collections.abc import Callable, Sequence
from typing import Any, Generic

from langgraph.checkpoint.base import DELTA_SENTINEL, PendingWrite
from typing_extensions import Self

from langgraph._internal._constants import OVERWRITE
from langgraph._internal._typing import MISSING
from langgraph.channels.base import BaseChannel, Value
from langgraph.errors import (
    EmptyChannelError,
    ErrorCode,
    InvalidUpdateError,
    create_error_message,
)
from langgraph.types import Overwrite

__all__ = ("DeltaChannel",)


def _empty(typ: Any) -> Any:
    try:
        return typ()
    except Exception:
        return []


def _get_overwrite(value: Any) -> tuple[bool, Any]:
    if isinstance(value, Overwrite):
        return True, value.value
    if isinstance(value, dict) and set(value.keys()) == {OVERWRITE}:
        return True, value[OVERWRITE]
    return False, None


class DeltaChannel(Generic[Value], BaseChannel[Any, Any, Any]):
    """Fold-reducer channel with configurable snapshot cadence.

    `snapshot_frequency=math.inf` (default): pure delta mode — stores only
    `DELTA_SENTINEL` in checkpoint blobs; reads replay all ancestor writes
    through `operator`. O(N) storage, O(N) read replay depth.

    `snapshot_frequency=N` (integer > 1): writes a full snapshot blob every
    N steps; on non-snapshot steps stores `DELTA_SENTINEL`. Reads walk at
    most N ancestors before finding the snapshot, bounding replay depth to N.
    Storage is O(N²/N) = O(N·avg_msg_size) — still linear per step,
    but snapshots grow with accumulated messages so total is O(N²/N).

    Parameters:
        operator: Binary reducer `(Value, Value) -> Value` applied pairwise.
            Must be associative when `snapshot_frequency > 1` since replayed
            writes fold through the same operator as live writes.
        snapshot_frequency: Every Nth step writes a full snapshot blob.
            Default `math.inf` (pure delta, never snapshot).
    """

    __slots__ = ("value", "operator", "snapshot_frequency", "_write_count")
    value: Value | Any

    def __init__(
        self,
        operator: Callable[[Any, Any], Any],
        *,
        snapshot_frequency: int | float = math.inf,
    ) -> None:
        super().__init__(list)
        self.operator = operator
        self.snapshot_frequency = snapshot_frequency
        self.value: Any = []
        self._write_count: int = 0

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, DeltaChannel):
            return False
        if self.snapshot_frequency != other.snapshot_frequency:
            return False
        if (
            self.operator.__name__ != "<lambda>"
            and other.operator.__name__ != "<lambda>"
        ):
            return self.operator is other.operator
        return True

    @property
    def ValueType(self) -> Any:
        return self.typ

    @property
    def UpdateType(self) -> Any:
        return self.typ

    def _clone_empty(self) -> Self:
        new = self.__class__.__new__(self.__class__)
        new.typ = self.typ
        new.key = self.key
        new.operator = self.operator
        new.snapshot_frequency = self.snapshot_frequency
        new._write_count = 0
        new.value = MISSING
        return new

    def copy(self) -> Self:
        new = self._clone_empty()
        new._write_count = self._write_count
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

        Blob formats:
          * `DELTA_SENTINEL` / `MISSING`: start empty; caller will replay writes.
          * `{"__delta_v__": value, "__delta_wc__": n}`: snapshot blob — restore
            value and write count so future writes trigger the next snapshot at
            the right cadence.
          * plain value (migration from old DeltaChannel blobs): restore value,
            reset write count to 0.
        """
        new = self._clone_empty()
        if checkpoint is MISSING or checkpoint is DELTA_SENTINEL:
            new.value = _empty(self.typ)
            new._write_count = 0
        elif isinstance(checkpoint, dict) and "__delta_v__" in checkpoint:
            new.value = checkpoint["__delta_v__"]
            new._write_count = checkpoint["__delta_wc__"]
        else:
            new.value = checkpoint
            new._write_count = 0
        return new

    def replay_writes(self, writes: Sequence[PendingWrite]) -> None:
        """Fold ancestor writes oldest→newest into current value.

        Also increments `_write_count` per replayed write so the snapshot
        cadence stays correct across invocations. The count resumes from
        wherever the seed's `__delta_wc__` left off (set by `from_checkpoint`).
        """
        for _, _, value in writes:
            self.value = self._apply_write(self.value, value)
            self._write_count += 1

    def update(self, values: Sequence[Any]) -> bool:
        if not values:
            return False
        seen_overwrite = False
        for value in values:
            is_overwrite, _ = _get_overwrite(value)
            if is_overwrite:
                if seen_overwrite:
                    msg = create_error_message(
                        message="Can receive only one Overwrite value per super-step.",
                        error_code=ErrorCode.INVALID_CONCURRENT_GRAPH_UPDATE,
                    )
                    raise InvalidUpdateError(msg)
                seen_overwrite = True
            elif seen_overwrite:
                continue
            self.value = self._apply_write(self.value, value)
        self._write_count += 1
        return True

    def get(self) -> Any:
        if self.value is MISSING:
            raise EmptyChannelError()
        return self.value

    def is_available(self) -> bool:
        return self.value is not MISSING

    def checkpoint(self) -> Any:
        """Return stored representation.

        Pure delta mode (`snapshot_frequency=math.inf`) always returns
        `DELTA_SENTINEL`. For finite `snapshot_frequency`, returns a snapshot
        blob `{"__delta_v__": value, "__delta_wc__": n}` every
        `snapshot_frequency` writes, with the current write count embedded so
        `from_checkpoint` can restore the counter and maintain correct cadence
        across invocations. All other writes return `DELTA_SENTINEL`.
        """
        if self.value is MISSING:
            return MISSING
        if self.snapshot_frequency == math.inf:
            return DELTA_SENTINEL
        if self._write_count > 0 and self._write_count % self.snapshot_frequency == 0:
            return {"__delta_v__": self.value, "__delta_wc__": self._write_count}
        return DELTA_SENTINEL
