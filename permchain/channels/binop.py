from contextlib import contextmanager
from typing import Callable, Generator, Generic, Optional, Sequence, Type

from typing_extensions import Self

from permchain.channels.base import BaseChannel, EmptyChannelError, Value


class BinaryOperatorAggregate(Generic[Value], BaseChannel[Value, Value, Value]):
    """Stores the result of applying a binary operator to the current value and each new value.

    ```python
    import operator

    total = Channels.BinaryOperatorAggregate(int, operator.add)
    ```
    """

    def __init__(self, typ: Type[Value], operator: Callable[[Value, Value], Value]):
        self.typ = typ
        self.operator = operator

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
        empty = self.__class__(self.typ, self.operator)
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
        if not values:
            return
        if not hasattr(self, "value"):
            self.value = values[0]
            values = values[1:]

        for value in values:
            self.value = self.operator(self.value, value)

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
