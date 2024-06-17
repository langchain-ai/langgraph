from contextlib import contextmanager
from typing import Generator, Generic, Optional, Sequence, Type

from typing_extensions import Self

from langgraph.channels.base import BaseChannel, Value
from langgraph.errors import EmptyChannelError, InvalidUpdateError


class NamedBarrierValue(Generic[Value], BaseChannel[Value, Value, set[Value]]):
    """A channel that waits until all named values are received before making the value available."""

    def __init__(self, typ: Type[Value], names: set[Value]) -> None:
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

    def update(self, values: Sequence[Value]) -> bool:
        updated = False
        for value in values:
            if value in self.names:
                if value not in self.seen:
                    self.seen.add(value)
                    updated = True
            else:
                raise InvalidUpdateError(f"Value {value} not in {self.names}")
        return updated

    def get(self) -> Value:
        if self.seen != self.names:
            raise EmptyChannelError()
        return None

    def consume(self) -> bool:
        if self.seen == self.names:
            self.seen = set()
            return True
        return False
