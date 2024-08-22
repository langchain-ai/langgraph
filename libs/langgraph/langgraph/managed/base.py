from abc import ABC, abstractmethod
from contextlib import asynccontextmanager, contextmanager
from inspect import isclass
from typing import (
    Any,
    AsyncIterator,
    Generic,
    Iterator,
    NamedTuple,
    Sequence,
    Type,
    TypeVar,
    Union,
)

from langchain_core.runnables import RunnableConfig
from typing_extensions import Self, TypeGuard

V = TypeVar("V")
U = TypeVar("U")


class ManagedValue(ABC, Generic[V]):
    def __init__(self, config: RunnableConfig) -> None:
        self.config = config

    @classmethod
    @contextmanager
    def enter(cls, config: RunnableConfig, **kwargs: Any) -> Iterator[Self]:
        try:
            value = cls(config, **kwargs)
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
    async def aenter(cls, config: RunnableConfig, **kwargs: Any) -> AsyncIterator[Self]:
        try:
            value = cls(config, **kwargs)
            yield value
        finally:
            # because managed value and Pregel have reference to each other
            # let's make sure to break the reference on exit
            try:
                del value
            except UnboundLocalError:
                pass

    @abstractmethod
    def __call__(self, step: int) -> V:
        ...


class WritableManagedValue(Generic[V, U], ManagedValue[V], ABC):
    @abstractmethod
    def update(self, writes: Sequence[U]) -> None:
        ...

    @abstractmethod
    async def aupdate(self, writes: Sequence[U]) -> None:
        ...


class ConfiguredManagedValue(NamedTuple):
    cls: Type[ManagedValue]
    kwargs: dict[str, Any]


ManagedValueSpec = Union[Type[ManagedValue], ConfiguredManagedValue]

ManagedValueMapping = dict[str, ManagedValue]


def is_managed_value(value: Any) -> TypeGuard[ManagedValueSpec]:
    return (isclass(value) and issubclass(value, ManagedValue)) or isinstance(
        value, ConfiguredManagedValue
    )


def is_readonly_managed_value(value: Any) -> TypeGuard[Type[ManagedValue]]:
    return (
        isclass(value)
        and issubclass(value, ManagedValue)
        and not issubclass(value, WritableManagedValue)
    ) or (
        isinstance(value, ConfiguredManagedValue)
        and not issubclass(value.cls, WritableManagedValue)
    )


def is_writable_managed_value(value: Any) -> TypeGuard[Type[WritableManagedValue]]:
    return (isclass(value) and issubclass(value, WritableManagedValue)) or (
        isinstance(value, ConfiguredManagedValue)
        and issubclass(value.cls, WritableManagedValue)
    )


ChannelKeyPlaceholder = object()
ChannelTypePlaceholder = object()
