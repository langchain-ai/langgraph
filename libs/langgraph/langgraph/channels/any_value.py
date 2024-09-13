from typing import Generic, Optional, Sequence, Type

from typing_extensions import Self

from langgraph.channels.base import BaseChannel, Value
from langgraph.errors import EmptyChannelError


class AnyValue(Generic[Value], BaseChannel[Value, Value, Value]):
    """Stores the last value received, assumes that if multiple values are
    received, they are all equal."""

    def __init__(self, typ: Type[Value]) -> None:
        self.typ = typ

    def __eq__(self, value: object) -> bool:
        return isinstance(value, AnyValue)

    @property
    def ValueType(self) -> Type[Value]:
        """The type of the value stored in the channel."""
        return self.typ

    @property
    def UpdateType(self) -> Type[Value]:
        """The type of the update received by the channel."""
        return self.typ

    def from_checkpoint(self, checkpoint: Optional[Value]) -> Self:
        empty = self.__class__(self.typ)
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

        self.value = values[-1]
        return True

    def get(self) -> Value:
        try:
            return self.value
        except AttributeError:
            raise EmptyChannelError()
