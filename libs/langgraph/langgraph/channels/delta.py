from __future__ import annotations

import copy as _copy
from collections.abc import Callable, Sequence
from typing import Any, Generic

from langgraph.checkpoint.base import DELTA_SENTINEL, DeltaChannelWrites
from typing_extensions import Self

from langgraph._internal._typing import MISSING
from langgraph.channels.base import BaseChannel, Value
from langgraph.channels.binop import _get_overwrite
from langgraph.errors import EmptyChannelError
from langgraph.types import Overwrite

__all__ = ("DeltaChannel",)


def _empty(typ: Any) -> Any:
    try:
        return typ()
    except Exception:
        return []


class DeltaChannel(Generic[Value], BaseChannel[Any, Any, Any]):
    """A channel that stores only a sentinel in checkpoints; per-step writes are
    stored in checkpoint_writes and replayed through the operator at load time.

    Use with append-style reducers (e.g. `add_messages`) on long-running threads
    to eliminate O(N²) blob growth — storage is O(N) using the writes table that
    every checkpointer already maintains.

    ``snapshot_every`` bounds reconstruction cost. When set, every N effective
    writes an `Overwrite(full_value)` is injected into `checkpoint_writes`; on
    reload the saver scans writes newest→oldest and stops at the first
    `Overwrite`, so replay work is bounded regardless of thread age. Any
    user-written `Overwrite` on the channel provides the same benefit for free.

    Usage::

        class State(TypedDict):
            messages: Annotated[list[AnyMessage], DeltaChannel(add_messages)]
            # With periodic snapshotting for long threads:
            messages: Annotated[
                list[AnyMessage],
                DeltaChannel(add_messages, snapshot_every=100),
            ]
    """

    __slots__ = (
        "value",
        "operator",
        "snapshot_every",
        "_writes_since_snapshot",
    )

    def __init__(
        self,
        operator: Callable[[Any, Any], Any],
        *,
        snapshot_every: int | None = None,
    ) -> None:
        super().__init__(list)
        self.operator = operator
        self.value: Any = []
        self.snapshot_every = snapshot_every
        self._writes_since_snapshot = 0

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, DeltaChannel):
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

    def copy(self) -> Self:
        new: DeltaChannel[Value] = DeltaChannel(
            self.operator, snapshot_every=self.snapshot_every
        )
        new.typ = self.typ
        new.key = self.key
        new.value = self.value if self.value is MISSING else _copy.copy(self.value)
        new._writes_since_snapshot = self._writes_since_snapshot
        return new

    def _apply_write(self, value: Any, write: Any, counter: int) -> tuple[Any, int]:
        """Apply one write to `value`; return (new_value, new_counter).

        An `Overwrite` resets the counter to 0; any other write increments it.
        Centralizes the Overwrite/reducer branching used by both `update` (live
        super-step) and `from_checkpoint` (ancestor replay).
        """
        is_overwrite, overwrite_value = _get_overwrite(write)
        if is_overwrite:
            new_value = (
                _copy.copy(overwrite_value)
                if overwrite_value is not None
                else _empty(self.typ)
            )
            return new_value, 0
        base = _empty(self.typ) if value is MISSING else value
        return self.operator(base, write), counter + 1

    def from_checkpoint(self, checkpoint: Any) -> Self:
        new: DeltaChannel[Value] = DeltaChannel(
            self.operator, snapshot_every=self.snapshot_every
        )
        new.typ = self.typ
        new.key = self.key
        if checkpoint is MISSING:
            new.value = _empty(new.typ)
            new._writes_since_snapshot = 0
        elif isinstance(checkpoint, DeltaChannelWrites):
            # Saver reconstructed per-step writes; replay through the operator.
            # Counter tracks writes since the last Overwrite so snapshot cadence
            # stays accurate across reloads.
            value: Any = _empty(new.typ)
            counter = 0
            for write in checkpoint.writes:
                value, counter = new._apply_write(value, write, counter)
            new.value = value
            new._writes_since_snapshot = counter
        else:
            # Backward compat: a pre-DeltaChannel thread stored the accumulated
            # value directly. Trust it as-is.
            new.value = checkpoint
            new._writes_since_snapshot = 0
        return new

    def update(self, values: Sequence[Any]) -> bool:
        if not values:
            return False
        seen_overwrite = False
        for value in values:
            is_overwrite, _ = _get_overwrite(value)
            if is_overwrite:
                if seen_overwrite:
                    from langgraph.errors import (
                        ErrorCode,
                        InvalidUpdateError,
                        create_error_message,
                    )

                    msg = create_error_message(
                        message="Can receive only one Overwrite value per super-step.",
                        error_code=ErrorCode.INVALID_CONCURRENT_GRAPH_UPDATE,
                    )
                    raise InvalidUpdateError(msg)
                seen_overwrite = True
            elif seen_overwrite:
                # Post-Overwrite writes within the same super-step are dropped.
                continue
            self.value, self._writes_since_snapshot = self._apply_write(
                self.value, value, self._writes_since_snapshot
            )
        return True

    def get(self) -> Any:
        if self.value is MISSING:
            raise EmptyChannelError()
        return self.value

    def is_available(self) -> bool:
        return self.value is not MISSING

    def checkpoint(self) -> Any:
        return DELTA_SENTINEL

    def should_snapshot(self) -> bool:
        """True if enough writes have accumulated to justify a snapshot.

        Pregel checks this after a checkpoint is saved; if true, it injects
        `snapshot_write()` into `checkpoint_writes`.
        """
        return (
            self.snapshot_every is not None
            and self._writes_since_snapshot >= self.snapshot_every
        )

    def snapshot_write(self) -> Overwrite:
        """Return the write to persist and reset the counter.

        The write is an `Overwrite(current_value)`; on replay the operator's
        `Overwrite` handling resets state to this value, and the saver's
        ancestor walk can stop here.
        """
        self._writes_since_snapshot = 0
        return Overwrite(_copy.copy(self.value))
