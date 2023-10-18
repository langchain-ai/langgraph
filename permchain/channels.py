import json
from abc import ABC, abstractmethod
from types import TracebackType
from typing import (
    Callable,
    FrozenSet,
    Generic,
    Optional,
    Sequence,
    Type,
    TypeVar,
)

from typing_extensions import Self

Value = TypeVar("Value")
Update = TypeVar("Update")


class EmptyChannelError(Exception):
    pass


class InvalidUpdateError(Exception):
    pass


class Channel(Generic[Value, Update], ABC):
    @property
    @abstractmethod
    def ValueType(self) -> type[Value]:
        """The type of the value stored in the channel."""

    @property
    @abstractmethod
    def UpdateType(self) -> type[Update]:
        """The type of the update received by the channel."""

    @abstractmethod
    def __enter__(self, checkpoint: Optional[str] = None) -> Self:
        """Return a new identical channel, optionally initialized from a checkpoint."""

    @abstractmethod
    def __exit__(
        self,
        __exc_type: type[BaseException] | None,
        __exc_value: BaseException | None,
        __traceback: TracebackType | None,
    ) -> bool | None:
        """Clean up the channel, or deallocate resources as needed."""
        ...

    async def __aenter__(self, checkpoint: Optional[str] = None) -> Self:
        return self.__enter__(checkpoint)

    async def __aexit__(
        self,
        __exc_type: type[BaseException] | None,
        __exc_value: BaseException | None,
        __traceback: TracebackType | None,
    ) -> None:
        self.__exit__(__exc_type, __exc_value, __traceback)

    @abstractmethod
    def _update(self, values: Sequence[Update]) -> None:
        ...

    @abstractmethod
    def _get(self) -> Value:
        ...

    @abstractmethod
    def _checkpoint(self) -> str:
        ...


class BinaryOperatorAggregate(Generic[Value], Channel[Value, Value]):
    """Stores the result of applying a binary operator to the current value and each new value.

    ```python
    import operator

    total = BinaryOperatorAggregate(int, operator.add)
    ```
    """

    def __init__(self, typ: Type[Value], operator: Callable[[Value, Value], Value]):
        self.typ = typ
        self.operator = operator

    @property
    def ValueType(self) -> type[Value]:
        """The type of the value stored in the channel."""
        return self.typ

    @property
    def UpdateType(self) -> type[Value]:
        """The type of the update received by the channel."""
        return self.typ

    def __enter__(self, checkpoint: Optional[str] = None) -> Self:
        empty = self.__class__(self.typ, self.operator)
        if checkpoint is not None:
            empty.value = json.loads(checkpoint)
        return empty

    def __exit__(
        self,
        __exc_type: type[BaseException] | None,
        __exc_value: BaseException | None,
        __traceback: TracebackType | None,
    ) -> None:
        try:
            del self.value
        except AttributeError:
            pass

    def _update(self, values: Sequence[Value]) -> None:
        if not hasattr(self, "value"):
            self.value = values[0]
            values = values[1:]

        for value in values:
            self.value = self.operator(self.value, value)

    def _get(self) -> Value:
        try:
            return self.value
        except AttributeError:
            raise EmptyChannelError()

    def _checkpoint(self) -> str:
        return json.dumps(self.value)


class LastValue(Generic[Value], Channel[Value, Value]):
    """Stores the last value received."""

    def __init__(self, typ: Type[Value]) -> None:
        self.typ = typ

    @property
    def ValueType(self) -> type[Value]:
        """The type of the value stored in the channel."""
        return self.typ

    @property
    def UpdateType(self) -> type[Value]:
        """The type of the update received by the channel."""
        return self.typ

    def __enter__(self, checkpoint: Optional[str] = None) -> Self:
        empty = self.__class__(self.typ)
        if checkpoint is not None:
            empty.value = json.loads(checkpoint)
        return empty

    def __exit__(
        self,
        __exc_type: type[BaseException] | None,
        __exc_value: BaseException | None,
        __traceback: TracebackType | None,
    ) -> None:
        try:
            del self.value
        except AttributeError:
            pass

    def _update(self, values: Sequence[Value]) -> None:
        if len(values) != 1:
            raise InvalidUpdateError()

        self.value = values[-1]

    def _get(self) -> Value:
        try:
            return self.value
        except AttributeError:
            raise EmptyChannelError()

    def _checkpoint(self) -> str:
        return json.dumps(self.value)


class Inbox(Generic[Value], Channel[Sequence[Value], Value]):
    """Stores all values received, resets in each step."""

    def __init__(self, typ: Type[Value]) -> None:
        self.typ = typ

    @property
    def ValueType(self) -> type[Sequence[Value]]:
        """The type of the value stored in the channel."""
        return Sequence[self.typ]  # type: ignore[name-defined]

    @property
    def UpdateType(self) -> type[Value]:
        """The type of the update received by the channel."""
        return self.typ

    def __enter__(self, checkpoint: Optional[str] = None) -> Self:
        empty = self.__class__(self.typ)
        if checkpoint is not None:
            empty.queue = tuple(json.loads(checkpoint))
        return empty

    def __exit__(
        self,
        __exc_type: type[BaseException] | None,
        __exc_value: BaseException | None,
        __traceback: TracebackType | None,
    ) -> None:
        try:
            del self.queue
        except AttributeError:
            pass

    def _update(self, values: Sequence[Value]) -> None:
        self.queue = tuple(values)

    def _get(self) -> Sequence[Value]:
        try:
            return self.queue
        except AttributeError:
            raise EmptyChannelError()

    def _checkpoint(self) -> str:
        return json.dumps(self.queue)


class Set(Generic[Value], Channel[FrozenSet[Value], Value]):
    """Stores all unique values received."""

    def __init__(self, typ: Type[Value]) -> None:
        self.typ = typ

    @property
    def ValueType(self) -> type[FrozenSet[Value]]:
        """The type of the value stored in the channel."""
        return FrozenSet[self.typ]  # type: ignore[name-defined]

    @property
    def UpdateType(self) -> type[Value]:
        """The type of the update received by the channel."""
        return self.typ

    def __enter__(self, checkpoint: Optional[str] = None) -> Self:
        empty = self.__class__(self.typ)
        if checkpoint is not None:
            empty.set = set(json.loads(checkpoint))
        return empty

    def __exit__(
        self,
        __exc_type: type[BaseException] | None,
        __exc_value: BaseException | None,
        __traceback: TracebackType | None,
    ) -> None:
        try:
            del self.set
        except AttributeError:
            pass

    def _update(self, values: Sequence[Value]) -> None:
        if not hasattr(self, "set"):
            self.set = set[Value]()
        self.set.update(values)

    def _get(self) -> FrozenSet[Value]:
        try:
            return frozenset(self.set)
        except AttributeError:
            raise EmptyChannelError()

    def _checkpoint(self) -> str:
        return json.dumps(list(self.set))
