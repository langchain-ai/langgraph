from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Generic

from typing_extensions import Self

from langgraph._internal._typing import MISSING
from langgraph.channels.base import BaseChannel, Value
from langgraph.errors import EmptyChannelError, InvalidUpdateError

__all__ = ("UntrackedValue",)


class UntrackedValue(Generic[Value], BaseChannel[Value, Value, Value]):
    """Stores the last value received, never checkpointed."""

    __slots__ = ("value", "guard")

    guard: bool
    value: Value | Any

    def __init__(self, typ: type[Value], guard: bool = True) -> None:
        super().__init__(typ)
        self.guard = guard
        self.value = MISSING

    def __eq__(self, value: object) -> bool:
        return isinstance(value, UntrackedValue) and value.guard == self.guard

    @property
    def ValueType(self) -> type[Value]:
        """The type of the value stored in the channel."""
        return self.typ

    @property
    def UpdateType(self) -> type[Value]:
        """The type of the update received by the channel."""
        return self.typ

    def copy(self) -> Self:
        """Return a copy of the channel."""
        empty = self.__class__(self.typ, self.guard)
        empty.key = self.key
        empty.value = self.value
        return empty

    def checkpoint(self) -> Value | Any:
        return MISSING

    def from_checkpoint(self, checkpoint: Value) -> Self:
        empty = self.__class__(self.typ, self.guard)
        empty.key = self.key
        return empty

    def update(self, values: Sequence[Value]) -> bool:
        if len(values) == 0:
            return False
        if len(values) != 1 and self.guard:
            raise InvalidUpdateError(
                f"At key '{self.key}': UntrackedValue(guard=True) can receive only one value per step. Use guard=False if you want to store any one of multiple values."
            )

        self.value = values[-1]
        return True

    def get(self) -> Value:
        if self.value is MISSING:
            raise EmptyChannelError()
        return self.value

    def is_available(self) -> bool:
        return self.value is not MISSING
