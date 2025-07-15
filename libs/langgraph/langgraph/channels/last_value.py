from collections.abc import Sequence
from typing import Any, Generic

from typing_extensions import Self

from langgraph._internal._typing import UNSET, UnsetType
from langgraph.channels.base import BaseChannel, Value
from langgraph.errors import (
    EmptyChannelError,
    ErrorCode,
    InvalidUpdateError,
    create_error_message,
)

__all__ = ("LastValue", "LastValueAfterFinish")


class LastValue(Generic[Value], BaseChannel[Value, Value, Value]):
    """Stores the last value received, can receive at most one value per step."""

    __slots__ = ("value",)

    value: Value | UnsetType

    def __init__(self, typ: Any, key: str = "") -> None:
        super().__init__(typ, key)
        self.value = UNSET

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

    def from_checkpoint(self, checkpoint: Value | UnsetType) -> Self:
        empty = self.__class__(self.typ, self.key)
        if not isinstance(checkpoint, UnsetType):
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
        if self.value is UNSET:
            raise EmptyChannelError()
        return self.value

    def is_available(self) -> bool:
        return self.value is not UNSET

    def checkpoint(self) -> Value:
        return self.value


class LastValueAfterFinish(
    Generic[Value], BaseChannel[Value, Value, tuple[Value | UnsetType, bool]]
):
    """Stores the last value received, but only made available after finish().
    Once made available, clears the value."""

    __slots__ = ("value", "finished")

    value: Value | UnsetType
    finished: bool

    def __init__(self, typ: Any, key: str = "") -> None:
        super().__init__(typ, key)
        self.value = UNSET
        self.finished = False

    def __eq__(self, value: object) -> bool:
        return isinstance(value, LastValueAfterFinish)

    @property
    def ValueType(self) -> type[Value]:
        """The type of the value stored in the channel."""
        return self.typ

    @property
    def UpdateType(self) -> type[Value]:
        """The type of the update received by the channel."""
        return self.typ

    def checkpoint(self) -> tuple[Value | UnsetType, bool] | UnsetType:
        if isinstance(self.value, UnsetType):
            return UNSET
        return (self.value, self.finished)

    def from_checkpoint(
        self, checkpoint: tuple[Value | UnsetType, bool] | UnsetType
    ) -> Self:
        empty = self.__class__(self.typ)
        empty.key = self.key
        if not isinstance(checkpoint, UnsetType):
            empty.value, empty.finished = checkpoint
        return empty

    def update(self, values: Sequence[Value | UnsetType]) -> bool:
        if len(values) == 0:
            return False

        self.finished = False
        self.value = values[-1]
        return True

    def consume(self) -> bool:
        if self.finished:
            self.finished = False
            self.value = UNSET
            return True

        return False

    def finish(self) -> bool:
        if not self.finished and self.value is not UNSET:
            self.finished = True
            return True
        else:
            return False

    def get(self) -> Value:
        if isinstance(self.value, UnsetType) or not self.finished:
            raise EmptyChannelError()
        return self.value

    def is_available(self) -> bool:
        return self.value is not UNSET and self.finished
