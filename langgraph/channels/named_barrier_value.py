from contextlib import contextmanager
from typing import Generator, Generic, Optional, Sequence, Type

from typing_extensions import Self

from langgraph.channels.base import (
    BaseChannel,
    EmptyChannelError,
    InvalidUpdateError,
    Value,
)


class NamedBarrierValue(Generic[Value], BaseChannel[None, Value, set[Value]]):
    """A channel that waits until all named values are received before making the value available."""

    def __init__(self, typ: Type[Value], names: set[Value]) -> None:
        self.typ = typ
        self.names = names
        self.seen: set[Value] = set()

    @property
    def ValueType(self) -> Type[None]:
        """The type of the value stored in the channel."""
        return type(None)

    @property
    def UpdateType(self) -> Type[Value]:
        """The type of the update received by the channel."""
        return self.typ

    def checkpoint(self) -> set[Value]:
        return self.seen

    @contextmanager
    def from_checkpoint(
        self, checkpoint: Optional[set[Value]] = None
    ) -> Generator[Self, None, None]:
        empty = self.__class__(self.typ, self.names)
        if checkpoint is not None:
            empty.seen = checkpoint.copy()

        try:
            yield empty
        finally:
            pass

    def update(self, values: Sequence[Value]) -> None:
        if self.seen == self.names:
            self.seen = set()
        for value in values:
            if value in self.names:
                self.seen.add(value)
            else:
                raise InvalidUpdateError(f"Value {value} not in {self.names}")

    def get(self) -> None:
        if self.seen != self.names:
            raise EmptyChannelError()
        return None
