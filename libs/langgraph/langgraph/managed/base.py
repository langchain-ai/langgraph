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

from langgraph.constants import RUNTIME_PLACEHOLDER

V = TypeVar("V")
U = TypeVar("U")


class ManagedValue(ABC, Generic[V]):
    runtime: bool = False
    """Whether the managed value is always created at runtime, ie. never stored."""

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


class ManagedValueMapping(dict[str, ManagedValue]):
    def replace_runtime_values(self, step: int, values: Union[dict[str, Any], Any]):
        if isinstance(values, dict):
            for key, value in values.items():
                for chan, mv in self.items():
                    if mv.runtime and mv(step) is value:
                        values[key] = {RUNTIME_PLACEHOLDER: chan}
        elif hasattr(values, "__dir__") and callable(values.__dir__):
            for key in dir(values):
                value = getattr(values, key)
                for chan, mv in self.items():
                    if mv.runtime and mv(step) is value:
                        setattr(values, key, {RUNTIME_PLACEHOLDER: chan})

    def replace_runtime_placeholders(
        self, step: int, values: Union[dict[str, Any], Any]
    ):
        if isinstance(values, dict):
            for key, value in values.items():
                if isinstance(value, dict) and RUNTIME_PLACEHOLDER in value:
                    values[key] = self[value[RUNTIME_PLACEHOLDER]](step)
        elif hasattr(values, "__dir__") and callable(values.__dir__):
            for key in dir(values):
                value = getattr(values, key)
                if isinstance(value, dict) and RUNTIME_PLACEHOLDER in value:
                    setattr(values, key, self[value[RUNTIME_PLACEHOLDER]](step))
