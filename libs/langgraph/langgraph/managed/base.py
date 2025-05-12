from abc import ABC, abstractmethod
from collections.abc import AsyncIterator, Iterator, Sequence
from contextlib import asynccontextmanager, contextmanager
from inspect import isclass
from typing import (
    Any,
    Generic,
    NamedTuple,
    TypeVar,
    Union,
)

from typing_extensions import Self, TypeGuard

from langgraph.types import LoopProtocol

V = TypeVar("V")
U = TypeVar("U")


class ManagedValue(ABC, Generic[V]):
    def __init__(self, loop: LoopProtocol) -> None:
        self.loop = loop

    @classmethod
    @contextmanager
    def enter(cls, loop: LoopProtocol, **kwargs: Any) -> Iterator[Self]:
        try:
            value = cls(loop, **kwargs)
            yield value
        finally:
            # because managed value and Pregel have reference to each other
            # let's make sure to break the reference on exit
            try:
                del value
            except UnboundLocalError:
                pass

    @classmethod
    @asynccontextmanager
    async def aenter(cls, loop: LoopProtocol, **kwargs: Any) -> AsyncIterator[Self]:
        try:
            value = cls(loop, **kwargs)
            yield value
        finally:
            # because managed value and Pregel have reference to each other
            # let's make sure to break the reference on exit
            try:
                del value
            except UnboundLocalError:
                pass

    @abstractmethod
    def __call__(self) -> V: ...


class WritableManagedValue(Generic[V, U], ManagedValue[V], ABC):
    @abstractmethod
    def update(self, writes: Sequence[U]) -> None: ...

    @abstractmethod
    async def aupdate(self, writes: Sequence[U]) -> None: ...


class ConfiguredManagedValue(NamedTuple):
    cls: type[ManagedValue]
    kwargs: dict[str, Any]


ManagedValueSpec = Union[type[ManagedValue], ConfiguredManagedValue]


def is_managed_value(value: Any) -> TypeGuard[ManagedValueSpec]:
    return (isclass(value) and issubclass(value, ManagedValue)) or isinstance(
        value, ConfiguredManagedValue
    )


def is_readonly_managed_value(value: Any) -> TypeGuard[type[ManagedValue]]:
    return (
        isclass(value)
        and issubclass(value, ManagedValue)
        and not issubclass(value, WritableManagedValue)
    ) or (
        isinstance(value, ConfiguredManagedValue)
        and not issubclass(value.cls, WritableManagedValue)
    )


def is_writable_managed_value(value: Any) -> TypeGuard[type[WritableManagedValue]]:
    return (isclass(value) and issubclass(value, WritableManagedValue)) or (
        isinstance(value, ConfiguredManagedValue)
        and issubclass(value.cls, WritableManagedValue)
    )


ChannelKeyPlaceholder = object()
ChannelTypePlaceholder = object()


ManagedValueMapping = dict[str, ManagedValue]
