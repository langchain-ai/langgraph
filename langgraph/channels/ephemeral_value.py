from contextlib import contextmanager
from typing import Generator, Generic, Optional, Sequence, Type

from typing_extensions import Self

from langgraph.channels.base import BaseChannel, Value
from langgraph.errors import EmptyChannelError, InvalidUpdateError


class EphemeralValue(Generic[Value], BaseChannel[Value, Value, Value]):
    """Stores the value received in the step immediately preceding, clears after."""

    def __init__(self, typ: Type[Value], guard: bool = True) -> None:
        self.typ = typ
        self.guard = guard

    @property
    def ValueType(self) -> Type[Value]:
        """The type of the value stored in the channel."""
        return self.typ

    @property
    def UpdateType(self) -> Type[Value]:
        """The type of the update received by the channel."""
        return self.typ

    def checkpoint(self) -> Value:
        try:
            return self.value
        except AttributeError:
            raise EmptyChannelError()

    @contextmanager
    def from_checkpoint(
        self, checkpoint: Optional[Value] = None
    ) -> Generator[Self, None, None]:
        empty = self.__class__(self.typ, self.guard)
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
            try:
                del self.value
            except AttributeError:
                pass
            finally:
                return
        if len(values) != 1 and self.guard:
            raise InvalidUpdateError("LastValue can only receive one value per step.")

        self.value = values[-1]

    def get(self) -> Value:
        try:
            return self.value
        except AttributeError:
            raise EmptyChannelError()
