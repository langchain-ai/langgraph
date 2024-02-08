from contextlib import contextmanager
from typing import Generator, Generic, Optional, Sequence, Type

from typing_extensions import Self

from langgraph.channels.base import (
    BaseChannel,
    EmptyChannelError,
    InvalidUpdateError,
    Value,
)


class LastValue(Generic[Value], BaseChannel[Value, Value, Value]):
    """Stores the last value received, can receive at most one value per step."""

    def __init__(self, typ: Type[Value]) -> None:
        self.typ = typ

    @property
    def ValueType(self) -> Type[Value]:
        """The type of the value stored in the channel."""
        return self.typ

    @property
    def UpdateType(self) -> Type[Value]:
        """The type of the update received by the channel."""
        return self.typ

    @contextmanager
    def empty(self, checkpoint: Optional[Value] = None) -> Generator[Self, None, None]:
        empty = self.__class__(self.typ)
        if checkpoint is not None:
            empty.value = checkpoint
        try:
            yield empty
        finally:
            try:
                del empty.value
            except AttributeError:
                pass

    def update(self, values: Sequence[Value]) -> None:
        if len(values) == 0:
            return
        if len(values) != 1:
            raise InvalidUpdateError("LastValue can only receive one value per step.")

        self.value = values[-1]

    def get(self) -> Value:
        try:
            return self.value
        except AttributeError:
            raise EmptyChannelError()

    def checkpoint(self) -> Value:
        try:
            return self.value
        except AttributeError:
            raise EmptyChannelError()
