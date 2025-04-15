from typing import Any, Generic, Optional, Sequence, Type

from typing_extensions import Self

from langgraph.channels.base import BaseChannel, Value
from langgraph.constants import MISSING
from langgraph.errors import EmptyChannelError


class OpenAtTheCloseValue(
    Generic[Value], BaseChannel[Value, Value, tuple[Value, bool]]
):
    """Stores the last value received, but only made available after finish().
    Once made available, clears the value."""

    __slots__ = ("value", "finished")

    def __init__(self, typ: Any, key: str = "") -> None:
        super().__init__(typ, key)
        self.value = MISSING
        self.finished = False

    def __eq__(self, value: object) -> bool:
        return isinstance(value, OpenAtTheCloseValue)

    @property
    def ValueType(self) -> Type[Value]:
        """The type of the value stored in the channel."""
        return self.typ

    @property
    def UpdateType(self) -> Type[Value]:
        """The type of the update received by the channel."""
        return self.typ

    def checkpoint(self) -> tuple[Value, bool]:
        if self.value is MISSING:
            return MISSING
        return (self.value, self.finished)

    def from_checkpoint(self, checkpoint: Optional[tuple[Value, bool]]) -> Self:
        empty = self.__class__(self.typ)
        empty.key = self.key
        if checkpoint is not MISSING:
            empty.value, empty.finished = checkpoint
        return empty

    def update(self, values: Sequence[Value]) -> bool:
        if len(values) == 0:
            return False

        self.finished = False
        self.value = values[-1]
        return True

    def consume(self) -> bool:
        if self.finished:
            self.finished = False
            self.value = MISSING
            return True

        return False

    def finish(self) -> None:
        if not self.finished and self.value is not MISSING:
            self.finished = True
            return True
        else:
            return False

    def get(self) -> Value:
        if self.value is MISSING or not self.finished:
            raise EmptyChannelError()
        return self.value

    def is_available(self) -> bool:
        return self.value is not MISSING and self.finished
