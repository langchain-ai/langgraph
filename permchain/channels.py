import json
from abc import ABC, abstractmethod
from contextlib import asynccontextmanager, contextmanager
from typing import (
    AsyncContextManager,
    AsyncGenerator,
    Callable,
    FrozenSet,
    Generator,
    Generic,
    Optional,
    Sequence,
    Type,
    TypeVar,
)
from typing import ContextManager as ContextManagerType

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
    def ValueType(self) -> Type[Value]:
        """The type of the value stored in the channel."""

    @property
    @abstractmethod
    def UpdateType(self) -> Type[Update]:
        """The type of the update received by the channel."""

    @contextmanager
    @abstractmethod
    def _empty(self, checkpoint: Optional[str] = None) -> Generator[Self, None, None]:
        """Return a new identical channel, optionally initialized from a checkpoint."""

    @asynccontextmanager
    async def _aempty(
        self, checkpoint: Optional[str] = None
    ) -> AsyncGenerator[Self, None]:
        """Return a new identical channel, optionally initialized from a checkpoint."""
        with self._empty(checkpoint) as value:
            yield value

    @abstractmethod
    def _update(self, values: Sequence[Update]) -> None:
        ...

    @abstractmethod
    def _get(self) -> Value:
        ...

    @abstractmethod
    def _checkpoint(self) -> str | None:
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
    def ValueType(self) -> Type[Value]:
        """The type of the value stored in the channel."""
        return self.typ

    @property
    def UpdateType(self) -> Type[Value]:
        """The type of the update received by the channel."""
        return self.typ

    @contextmanager
    def _empty(self, checkpoint: Optional[str] = None) -> Generator[Self, None, None]:
        empty = self.__class__(self.typ, self.operator)
        if checkpoint is not None:
            empty.value = json.loads(checkpoint)
        try:
            yield empty
        finally:
            try:
                del empty.value
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
    def ValueType(self) -> Type[Value]:
        """The type of the value stored in the channel."""
        return self.typ

    @property
    def UpdateType(self) -> Type[Value]:
        """The type of the update received by the channel."""
        return self.typ

    @contextmanager
    def _empty(self, checkpoint: Optional[str] = None) -> Generator[Self, None, None]:
        empty = self.__class__(self.typ)
        if checkpoint is not None:
            empty.value = json.loads(checkpoint)
        try:
            yield empty
        finally:
            try:
                del empty.value
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
    def ValueType(self) -> Type[Sequence[Value]]:
        """The type of the value stored in the channel."""
        return Sequence[self.typ]  # type: ignore[name-defined]

    @property
    def UpdateType(self) -> Type[Value]:
        """The type of the update received by the channel."""
        return self.typ

    @contextmanager
    def _empty(self, checkpoint: Optional[str] = None) -> Generator[Self, None, None]:
        empty = self.__class__(self.typ)
        if checkpoint is not None:
            empty.queue = tuple(json.loads(checkpoint))
        try:
            yield empty
        finally:
            try:
                del empty.queue
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
    def ValueType(self) -> Type[FrozenSet[Value]]:
        """The type of the value stored in the channel."""
        return FrozenSet[self.typ]  # type: ignore[name-defined]

    @property
    def UpdateType(self) -> Type[Value]:
        """The type of the update received by the channel."""
        return self.typ

    @contextmanager
    def _empty(self, checkpoint: Optional[str] = None) -> Generator[Self, None, None]:
        empty = self.__class__(self.typ)
        if checkpoint is not None:
            empty.set = set(json.loads(checkpoint))
        try:
            yield empty
        finally:
            try:
                del empty.set
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


AsyncValue = TypeVar("AsyncValue")


class ContextManager(Generic[Value], Channel[Value, None]):
    value: Value

    def __init__(
        self,
        typ: Type[Value],
        ctx: Optional[Callable[[], ContextManagerType[Value]]] = None,
        actx: Optional[Callable[[], AsyncContextManager[Value]]] = None,
    ) -> None:
        if ctx is None and actx is None:
            raise ValueError("Must provide either sync or async context manager.")

        self.typ = typ
        self.ctx = ctx
        self.actx = actx

    @property
    def ValueType(self) -> Type[Value]:
        """The type of the value stored in the channel."""
        return self.typ

    @property
    def UpdateType(self) -> Type[None]:
        """The type of the update received by the channel."""
        raise InvalidUpdateError()

    @contextmanager
    def _empty(self, checkpoint: Optional[str] = None) -> Generator[Self, None, None]:
        if self.ctx is None:
            raise ValueError("Cannot enter sync context manager.")

        empty = self.__class__(self.typ, ctx=self.ctx, actx=self.actx)
        # ContextManager doesn't have a checkpoint
        ctx = self.ctx()
        empty.value = ctx.__enter__()
        try:
            yield empty
        finally:
            ctx.__exit__(None, None, None)

    @asynccontextmanager
    async def _aempty(
        self, checkpoint: Optional[str] = None
    ) -> AsyncGenerator[Self, None]:
        if self.actx is not None:
            empty = self.__class__(self.typ, ctx=self.ctx, actx=self.actx)
            # ContextManager doesn't have a checkpoint
            actx = self.actx()
            empty.value = await actx.__aenter__()
            try:
                yield empty
            finally:
                await actx.__aexit__(None, None, None)
        else:
            with self._empty() as empty:
                yield empty

    def _update(self, values: Sequence[None]) -> None:
        raise InvalidUpdateError()

    def _get(self) -> Value:
        try:
            return self.value
        except AttributeError:
            raise EmptyChannelError()

    def _checkpoint(self) -> None:
        return None
