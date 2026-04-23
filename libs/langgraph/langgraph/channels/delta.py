from __future__ import annotations

import copy as _copy
from collections.abc import Callable, Sequence
from typing import Any, Generic

from langgraph.checkpoint.base import DELTA_SENTINEL, SEED_UNSET, DeltaChannelWrites
from typing_extensions import Self

from langgraph._internal._typing import MISSING
from langgraph.channels.base import BaseChannel, Value
from langgraph.channels.binop import _get_overwrite
from langgraph.errors import EmptyChannelError

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

    Reconstruction replays every ancestor write through the operator, so
    per-get cost scales with thread depth. Compaction for deep threads is
    a follow-up — today, use this on threads of a few hundred turns.

    Usage::

        class State(TypedDict):
            messages: Annotated[list[AnyMessage], DeltaChannel(add_messages)]
    """

    __slots__ = (
        "value",
        "operator",
    )

    def __init__(
        self,
        operator: Callable[[Any, Any], Any],
    ) -> None:
        super().__init__(list)
        self.operator = operator
        self.value: Any = []

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
        new: DeltaChannel[Value] = DeltaChannel(self.operator)
        new.typ = self.typ
        new.key = self.key
        new.value = self.value if self.value is MISSING else _copy.copy(self.value)
        return new

    def _apply_write(self, value: Any, write: Any) -> Any:
        """Apply one write to `value` and return the new value.

        An `Overwrite` replaces the value; any other write is folded through
        the operator. Centralizes the Overwrite/reducer branching used by both
        `update` (live super-step) and `from_checkpoint` (ancestor replay).
        """
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
        new: DeltaChannel[Value] = DeltaChannel(self.operator)
        new.typ = self.typ
        new.key = self.key
        if checkpoint is MISSING:
            new.value = _empty(new.typ)
        elif isinstance(checkpoint, DeltaChannelWrites):
            # Saver reconstructed per-step writes; replay through the operator.
            # `seed` (if set) is a pre-delta accumulated value that terminates
            # the ancestor walk on the saver side: replay starts from it
            # instead of the channel's empty value.
            value: Any = (
                _empty(new.typ) if checkpoint.seed is SEED_UNSET else checkpoint.seed
            )
            for write in checkpoint.writes:
                value = new._apply_write(value, write)
            new.value = value
        else:
            # Backward compat: a pre-DeltaChannel thread stored the accumulated
            # value directly (no saver-side reconstruction happened). Trust it.
            new.value = checkpoint
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
            self.value = self._apply_write(self.value, value)
        return True

    def get(self) -> Any:
        if self.value is MISSING:
            raise EmptyChannelError()
        return self.value

    def is_available(self) -> bool:
        return self.value is not MISSING

    def checkpoint(self) -> Any:
        return DELTA_SENTINEL
