from __future__ import annotations

import collections.abc
import copy as _copy
import math
from collections.abc import Callable, Sequence
from typing import Any, Generic

from langgraph.checkpoint.base import DELTA_SENTINEL, PendingWrite
from typing_extensions import NotRequired, Required, Self

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

__all__ = ("AggregateChannel",)


def _strip_extras(t: Any) -> Any:
    """Strips Annotated, Required, and NotRequired wrappers."""
    if hasattr(t, "__origin__"):
        return _strip_extras(t.__origin__)
    if hasattr(t, "__origin__") and t.__origin__ in (Required, NotRequired):
        return _strip_extras(t.__args__[0])
    return t


def _concrete(typ: Any) -> Any:
    """Replace abstract collection types from `typing`/`collections.abc` with
    their instantiable counterparts."""
    typ = _strip_extras(typ)
    if typ in (collections.abc.Sequence, collections.abc.MutableSequence):
        return list
    if typ in (collections.abc.Set, collections.abc.MutableSet):
        return set
    if typ in (collections.abc.Mapping, collections.abc.MutableMapping):
        return dict
    return typ


def _empty(typ: Any) -> Any:
    try:
        return typ()
    except Exception:
        return []


def _get_overwrite(value: Any) -> tuple[bool, Any]:
    """Return (is_overwrite, overwrite_value) for an incoming write."""
    if isinstance(value, Overwrite):
        return True, value.value
    if isinstance(value, dict) and set(value.keys()) == {OVERWRITE}:
        return True, value[OVERWRITE]
    return False, None


class AggregateChannel(Generic[Value], BaseChannel[Value, Value, Any]):
    """Fold-reducer channel with configurable snapshot cadence.

    `snapshot_frequency=1` (default) writes a full blob every step — same
    storage behavior as the classic `BinaryOperatorAggregate`.

    `snapshot_frequency=N` (integer > 1) writes a sentinel on non-snapshot
    steps; the value is reconstructed at read time by folding ancestor
    writes through `operator`. On every Nth step a full blob is written,
    bounding replay depth to N.

    `snapshot_frequency=math.inf` never writes a blob — pure delta storage.
    Reconstruction replays every write from thread start.

    Parameters:
        operator: Binary reducer `(Value, Value) -> Value` applied pairwise
            to accumulate writes. Must be associative for correctness under
            `snapshot_frequency != 1` where the fold order across ancestor
            replay vs live writes differs from the classic single-fold path.
            Most practical reducers (`operator.add`, `add_messages`) satisfy
            this.
        snapshot_frequency: Every Nth step writes a full snapshot blob.
            Default 1 (snapshot always). `math.inf` for pure-delta mode.
            Reading at step M with `snapshot_frequency=N` walks at most
            `M % N` ancestor writes — bounded replay regardless of thread
            depth.
        typ: Value type. When used as an `Annotated[T, AggregateChannel(...)]`
            state field, the type is inferred from `T` and this kwarg is
            unused. Explicit kwarg is the escape hatch for imperative
            graph construction.

    Experimental under `snapshot_frequency > 1`: the sentinel+replay path
    is the same mechanism that the (now removed) `DeltaChannel` used; the
    cadence knob is new and should be validated on real workloads before
    being relied on in production.
    """

    __slots__ = ("value", "operator", "snapshot_frequency", "_typ_provided")

    def __init__(
        self,
        operator: Callable[[Value, Value], Value],
        *,
        snapshot_frequency: int | float = 1,
        typ: type[Value] | None = None,
    ) -> None:
        self._typ_provided = typ is not None
        concrete_typ = _concrete(typ) if typ is not None else list
        super().__init__(concrete_typ)
        self.operator = operator
        self.snapshot_frequency = snapshot_frequency
        try:
            self.value = concrete_typ()
        except Exception:
            self.value = MISSING

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, AggregateChannel):
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
        """Create a blank clone preserving all attributes, bypassing __init__.

        Subclasses (e.g. BinaryOperatorAggregate) have different __init__
        signatures; going through __init__ from copy/from_checkpoint would
        pass kwargs those subclasses don't accept. Bypassing avoids that.
        """
        new = self.__class__.__new__(self.__class__)
        new.typ = self.typ
        new.key = self.key
        new.operator = self.operator
        new.snapshot_frequency = self.snapshot_frequency
        new._typ_provided = self._typ_provided
        new.value = MISSING
        return new

    def copy(self) -> Self:
        new = self._clone_empty()
        new.value = self.value if self.value is MISSING else _copy.copy(self.value)
        return new

    def is_snapshot_step(self, step: int) -> bool:
        """Return True if a full blob should be written at this step.

        `snapshot_frequency=1` → always. `snapshot_frequency=math.inf` →
        never. Otherwise, `step % snapshot_frequency == 0`.
        """
        if self.snapshot_frequency == 1:
            return True
        if self.snapshot_frequency == math.inf:
            return False
        return step % self.snapshot_frequency == 0

    def _apply_write(self, value: Any, write: Any) -> Any:
        """Apply one write and return the new value. Handles Overwrite."""
        is_overwrite, overwrite_value = _get_overwrite(write)
        if is_overwrite:
            return (
                _copy.copy(overwrite_value)
                if overwrite_value is not None
                else _empty(self.typ)
            )
        if value is MISSING:
            return write
        return self.operator(value, write)

    def from_checkpoint(self, checkpoint: Any) -> Self:
        """Initialize from a stored blob, sentinel, or MISSING.

        If the stored value is a full blob, use it as-is. If it is
        `DELTA_SENTINEL` or `MISSING`, start empty — the caller (pregel)
        is responsible for replaying writes via `replay_writes` when
        applicable.
        """
        new = self._clone_empty()
        if checkpoint is MISSING or checkpoint is DELTA_SENTINEL:
            new.value = _empty(self.typ)
        else:
            new.value = checkpoint
        return new

    def replay_writes(self, writes: Sequence[PendingWrite]) -> None:
        """Fold a sequence of PendingWrite tuples into the current value.

        Called by pregel after `from_checkpoint(seed)` to replay per-step
        deltas from on-path ancestors through the operator. Writes are
        oldest→newest. Overwrite markers reset the reducer state at that
        point. `task_id` and `channel` fields are ignored — the caller
        has already filtered to this channel.
        """
        for _, _, value in writes:
            self.value = self._apply_write(self.value, value)

    def update(self, values: Sequence[Value]) -> bool:
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
        return True

    def get(self) -> Value:
        if self.value is MISSING:
            raise EmptyChannelError()
        return self.value

    def is_available(self) -> bool:
        return self.value is not MISSING

    def checkpoint(self) -> Any:
        """Return the serializable representation of current state.

        For `snapshot_frequency=math.inf` (pure delta), always returns
        `DELTA_SENTINEL` — the value lives in `checkpoint_writes` and is
        reconstructed by replay, never materialized as a blob.

        For integer `snapshot_frequency`, returns the full value. Pregel's
        `create_checkpoint` is responsible for consulting
        `is_snapshot_step(step)` to decide whether to actually store the
        full value or write `DELTA_SENTINEL` for that step.
        """
        if self.value is MISSING:
            return MISSING
        if self.snapshot_frequency == math.inf:
            return DELTA_SENTINEL
        return self.value
