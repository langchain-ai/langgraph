from collections.abc import Sequence
from typing import Any, Generic

from typing_extensions import Self

from langgraph._internal._typing import UNSET, UnsetType
from langgraph.channels.base import BaseChannel, Value
from langgraph.errors import EmptyChannelError

__all__ = ("AnyValue",)


class AnyValue(Generic[Value], BaseChannel[Value, Value, Value]):
    """Stores the last value received, assumes that if multiple values are
    received, they are all equal."""

    __slots__ = ("typ", "value")

    value: Value | UnsetType

    def __init__(self, typ: Any, key: str = "") -> None:
        super().__init__(typ, key)
        self.value = UNSET

    def __eq__(self, value: object) -> bool:
        return isinstance(value, AnyValue)

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
            if self.value is UNSET:
                return False
            else:
                self.value = UNSET
                return True

        self.value = values[-1]
        return True

    def get(self) -> Value:
        if isinstance(self.value, UnsetType):
            raise EmptyChannelError()
        return self.value

    def is_available(self) -> bool:
        return self.value is not UNSET

    def checkpoint(self) -> Value | UnsetType:
        return self.value
