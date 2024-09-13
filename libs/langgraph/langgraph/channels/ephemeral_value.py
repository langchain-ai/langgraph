from typing import Generic, Optional, Sequence, Type

from typing_extensions import Self

from langgraph.channels.base import BaseChannel, Value
from langgraph.errors import EmptyChannelError, InvalidUpdateError


class EphemeralValue(Generic[Value], BaseChannel[Value, Value, Value]):
    """Stores the value received in the step immediately preceding, clears after."""

    def __init__(self, typ: Type[Value], guard: bool = True) -> None:
        self.typ = typ
        self.guard = guard

    def __eq__(self, value: object) -> bool:
        return isinstance(value, EphemeralValue) and value.guard == self.guard

    @property
    def ValueType(self) -> Type[Value]:
        """The type of the value stored in the channel."""
        return self.typ

    @property
    def UpdateType(self) -> Type[Value]:
        """The type of the update received by the channel."""
        return self.typ

    def from_checkpoint(self, checkpoint: Optional[Value]) -> Self:
        empty = self.__class__(self.typ, self.guard)
        empty.key = self.key
        if checkpoint is not None:
            empty.value = checkpoint
        return empty

    def update(self, values: Sequence[Value]) -> bool:
        if len(values) == 0:
            try:
                del self.value
                return True
            except AttributeError:
                return False
        if len(values) != 1 and self.guard:
            raise InvalidUpdateError(
                f"At key '{self.key}': EphemeralValue(guard=True) can receive only one value per step. Use guard=False if you want to store any one of multiple values."
            )

        self.value = values[-1]
        return True

    def get(self) -> Value:
        try:
            return self.value
        except AttributeError:
            raise EmptyChannelError()
