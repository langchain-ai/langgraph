from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any, Generic

from langgraph.checkpoint.base import DeltaChannelSentinel
from typing_extensions import Self

from langgraph._internal._typing import MISSING
from langgraph.channels.base import BaseChannel, Value
from langgraph.channels.binop import _get_overwrite
from langgraph.errors import EmptyChannelError

__all__ = ("DeltaChannel",)


class DeltaChannel(
    Generic[Value], BaseChannel[list[Value], Value, DeltaChannelSentinel]
):
    """A channel that stores only a sentinel in checkpoints; per-step writes are
    stored in checkpoint_writes and replayed through the operator at load time.

    Use with append-style reducers (e.g. `add_messages`) on long-running threads
    to eliminate O(N²) blob growth — storage is O(N) using the writes table that
    every checkpointer already maintains.

    Works with all checkpointers. Savers with dedicated implementations
    (InMemorySaver, PostgresSaver) reconstruct in one pass; others fall back to
    walking the checkpoint list.

    Usage::

        class State(TypedDict):
            messages: Annotated[list[AnyMessage], DeltaChannel(add_messages)]
            # Dict-type reducer (type inferred from the Annotated outer type):
            files: Annotated[dict, DeltaChannel(merge_files)]
    """

    __slots__ = ("value", "operator")

    def __init__(
        self,
        operator: Callable[[list[Value], Any], list[Value]],
    ) -> None:
        super().__init__(list)
        self.operator = operator
        self.value: list[Value] = []

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
        return list[self.typ]  # type: ignore[name-defined]

    @property
    def UpdateType(self) -> Any:
        return self.typ | list[self.typ]  # type: ignore[name-defined]

    def copy(self) -> Self:
        new = DeltaChannel(self.operator)
        new.typ = self.typ
        new.key = self.key
        new.value = self.value if self.value is MISSING else self.value.copy()
        return new

    def from_checkpoint(self, checkpoint: Any) -> Self:
        new = DeltaChannel(self.operator)
        new.typ = self.typ
        new.key = self.key
        if checkpoint is MISSING:
            try:
                new.value = new.typ()
            except Exception:
                new.value = []
        elif isinstance(checkpoint, list):
            # Flat list of write values (oldest→newest) from get_channel_writes.
            try:
                value: Any = new.typ()
            except Exception:
                value = []
            for write in checkpoint:
                value = new.operator(value, write)
            new.value = value
        else:
            # Backward compat: plain accumulated value (e.g. from a migrated thread).
            try:
                new.value = list(checkpoint)
            except Exception:
                new.value = []
        return new

    def update(self, values: Sequence[Any]) -> bool:
        if not values:
            return False
        seen_overwrite = False
        for value in values:
            is_overwrite, overwrite_value = _get_overwrite(value)
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
                self.value = (
                    list(overwrite_value) if overwrite_value is not None else self.typ()
                )
                seen_overwrite = True
            elif not seen_overwrite:
                base = self.typ() if self.value is MISSING else self.value
                self.value = self.operator(base, value)
        return True

    def get(self) -> list[Value]:
        if self.value is MISSING:
            raise EmptyChannelError()
        return self.value

    def is_available(self) -> bool:
        return self.value is not MISSING

    def checkpoint(self) -> DeltaChannelSentinel:
        return DeltaChannelSentinel()
