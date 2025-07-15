from collections.abc import Sequence
from typing import Any, Generic

from typing_extensions import Self

from langgraph._internal._typing import UNSET, UnsetType
from langgraph.channels.base import BaseChannel, Value
from langgraph.errors import EmptyChannelError, InvalidUpdateError

__all__ = ("EphemeralValue",)


class EphemeralValue(Generic[Value], BaseChannel[Value, Value, Value]):
    """Stores the value received in the step immediately preceding, clears after."""

    __slots__ = ("value", "guard")

    value: Value | UnsetType
    guard: bool

    def __init__(self, typ: Any, guard: bool = True) -> None:
        super().__init__(typ)
        self.guard = guard
        self.value = UNSET

    def __eq__(self, value: object) -> bool:
        return isinstance(value, EphemeralValue) and value.guard == self.guard

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

    def from_checkpoint(self, checkpoint: Value | UnsetType) -> Self:
        empty = self.__class__(self.typ, self.guard)
        empty.key = self.key
        if isinstance(checkpoint, UnsetType):
            empty.value = checkpoint
        return empty

    def update(self, values: Sequence[Value | UnsetType]) -> bool:
        if len(values) == 0:
            if self.value is not UNSET:
                self.value = UNSET
                return True
            else:
                return False
        if len(values) != 1 and self.guard:
            raise InvalidUpdateError(
                f"At key '{self.key}': EphemeralValue(guard=True) can receive only one value per step. Use guard=False if you want to store any one of multiple values."
            )

        self.value = values[-1]
        return True

    def get(self) -> Value:
        if self.value is UNSET:
            raise EmptyChannelError()
        return self.value

    def is_available(self) -> bool:
        return self.value is not UNSET

    def checkpoint(self) -> Value:
        return self.value
