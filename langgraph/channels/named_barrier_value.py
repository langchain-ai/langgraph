from contextlib import contextmanager
from typing import Generator, Generic, Optional, Sequence, Type

from typing_extensions import Self

from langgraph.channels.base import (
    BaseChannel,
    EmptyChannelError,
    InvalidUpdateError,
    Value,
)


class NamedBarrierValue(Generic[Value], BaseChannel[Value, Value, Value]):
    """A channel that waits until all named values are received before making the value available."""

    def __init__(self, typ: Type[Value], names: set[str]) -> None:
        self.typ = typ
        self.names = names
        self.seen = set()

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
        empty = self.__class__(self.typ, self.names)
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
        if self.seen == self.names:
            self.seen = set()
        for value in values:
            if value in self.names:
                self.seen.add(value)
            else:
                raise InvalidUpdateError(f"Value {value} not in {self.names}")

    def get(self) -> Value:
        if self.seen != self.names:
            raise EmptyChannelError()
        return None

    def checkpoint(self) -> Value:
        return self.seen
