from collections.abc import Sequence
from typing import Any, Generic

from typing_extensions import Self

from langgraph.channels.base import BaseChannel, Value
from langgraph.constants import MISSING
from langgraph.errors import (
    EmptyChannelError,
    ErrorCode,
    InvalidUpdateError,
    create_error_message,
)


class LastValue(Generic[Value], BaseChannel[Value, Value, Value]):
    """Stores the last value received, can receive at most one value per step."""

    __slots__ = ("value",)

    def __init__(self, typ: Any, key: str = "") -> None:
        super().__init__(typ, key)
        self.value = MISSING

    def __eq__(self, value: object) -> bool:
        return isinstance(value, LastValue)

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
        empty = self.__class__(self.typ, self.key)
        empty.value = self.value
        return empty

    def from_checkpoint(self, checkpoint: Value) -> Self:
        empty = self.__class__(self.typ, self.key)
        if checkpoint is not MISSING:
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
        if self.value is MISSING:
            raise EmptyChannelError()
        return self.value

    def is_available(self) -> bool:
        return self.value is not MISSING

    def checkpoint(self) -> Value:
        return self.value
