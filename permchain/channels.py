from abc import ABC, abstractmethod
from typing import Callable, FrozenSet, Generic, Self, Sequence, TypeVar

Value = TypeVar("Value")
Update = TypeVar("Update")


class EmptyChannelError(Exception):
    pass


class InvalidUpdateError(Exception):
    pass


class Channel(Generic[Value, Update], ABC):
    def _empty(self) -> Self:
        return self.__class__()

    @abstractmethod
    def _update(self, values: Sequence[Update]) -> None:
        ...

    @abstractmethod
    def _get(self) -> Value:
        ...


class BinaryOperatorAggregate(Generic[Value], Channel[Value, Value]):
    def __init__(self, operator: Callable[[Value, Value], Value]):
        self.operator = operator

    def _empty(self) -> Self:
        return self.__class__(self.operator)

    def _update(self, values):
        if not hasattr(self, "value"):
            self.value = values[0]
            values = values[1:]

        for value in values:
            self.value = self.operator(self.value, value)

    def _get(self):
        try:
            return self.value
        except AttributeError:
            raise EmptyChannelError()


class LastValue(Generic[Value], Channel[Value, Value]):
    def _update(self, values):
        if len(values) != 1:
            raise InvalidUpdateError()

        self.value = values[-1]

    def _get(self):
        try:
            return self.value
        except AttributeError:
            raise EmptyChannelError()


class Inbox(Generic[Value], Channel[Sequence[Value], Value]):
    def _update(self, values):
        self.queue = tuple(values)

    def _get(self):
        try:
            return self.queue
        except AttributeError:
            raise EmptyChannelError()


class Set(Generic[Value], Channel[FrozenSet[Value], Value]):
    def _update(self, values) -> None:
        if not hasattr(self, "set"):
            self.set = set()
        self.set.update(values)

    def _get(self) -> FrozenSet[Value]:
        try:
            return frozenset(self.set)
        except AttributeError:
            raise EmptyChannelError()
