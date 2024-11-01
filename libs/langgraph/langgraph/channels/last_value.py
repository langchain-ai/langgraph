from typing import Generic, Optional, Sequence, Type

from typing_extensions import Self

from langgraph.channels.base import BaseChannel, Value
from langgraph.errors import (
    EmptyChannelError,
    ErrorCode,
    InvalidUpdateError,
    create_error_message,
)


class LastValue(Generic[Value], BaseChannel[Value, Value, Value]):
    """Stores the last value received, can receive at most one value per step."""

    __slots__ = ("value",)

    def __eq__(self, value: object) -> bool:
        return isinstance(value, LastValue)

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
            return False
        if len(values) != 1:
            msg = create_error_message(
                message=f"At key '{self.key}': Can receive only one value per step. Use an Annotated key to handle multiple values.",
                error_code=ErrorCode.INVALID_CONCURRENT_GRAPH_UPDATE,
            )
            raise InvalidUpdateError(msg)

        self.value = values[-1]
        return True

    def get(self) -> Value:
        try:
            return self.value
        except AttributeError:
            raise EmptyChannelError()
