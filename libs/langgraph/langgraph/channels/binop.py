import collections.abc
from contextlib import contextmanager
from typing import (
    Callable,
    Generator,
    Generic,
    Optional,
    Sequence,
    Type,
)

from langchain_core.runnables import RunnableConfig
from typing_extensions import NotRequired, Required, Self

from langgraph.channels.base import BaseChannel, Value
from langgraph.errors import EmptyChannelError


# Adapted from typing_extensions
def _strip_extras(t):
    """Strips Annotated, Required and NotRequired from a given type."""
    if hasattr(t, "__origin__"):
        return _strip_extras(t.__origin__)
    if hasattr(t, "__origin__") and t.__origin__ in (Required, NotRequired):
        return _strip_extras(t.__args__[0])

    return t


class BinaryOperatorAggregate(Generic[Value], BaseChannel[Value, Value, Value]):
    """Stores the result of applying a binary operator to the current value and each new value.

    ```python
    import operator

    total = Channels.BinaryOperatorAggregate(int, operator.add)
    ```
    """

    def __init__(self, typ: Type[Value], operator: Callable[[Value, Value], Value]):
        self.operator = operator
        # keep the type exposed by ValueType/UpdateType as-is
        self.typ = typ
        # special forms from typing or collections.abc are not instantiable
        # so we need to replace them with their concrete counterparts
        typ = _strip_extras(typ)
        if typ in (collections.abc.Sequence, collections.abc.MutableSequence):
            typ = list
        if typ in (collections.abc.Set, collections.abc.MutableSet):
            typ = set
        if typ in (collections.abc.Mapping, collections.abc.MutableMapping):
            typ = dict
        try:
            self.value = typ()
        except Exception:
            pass

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
        self, checkpoint: Optional[Value], config: RunnableConfig
    ) -> Generator[Self, None, None]:
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

    def update(self, values: Sequence[Value]) -> bool:
        if not values:
            return False
        if not hasattr(self, "value"):
            self.value = values[0]
            values = values[1:]
        for value in values:
            self.value = self.operator(self.value, value)
        return True

    def get(self) -> Value:
        try:
            return self.value
        except AttributeError:
            raise EmptyChannelError()
