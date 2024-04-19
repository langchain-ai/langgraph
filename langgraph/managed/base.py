from abc import ABC, abstractmethod
from contextlib import asynccontextmanager, contextmanager
from inspect import isclass
from typing import (
    Any,
    AsyncGenerator,
    Generator,
    Generic,
    Sequence,
    Type,
    TypeGuard,
    TypeVar,
)

from langchain_core.runnables import RunnableConfig

from langgraph.pregel.types import PregelTaskDescription

V = TypeVar("V")


class ManagedValue(ABC, Generic[V]):
    def __init__(self, config: RunnableConfig) -> None:
        self.config = config

    @abstractmethod
    def __call__(self, step: int, task: PregelTaskDescription) -> V:
        ...


def is_managed_value(value: Any) -> TypeGuard[Type[ManagedValue]]:
    return isclass(value) and issubclass(value, ManagedValue)


@contextmanager
def ManagedValuesManager(
    values: Sequence[Type[ManagedValue]],
    config: RunnableConfig,
) -> Generator[Sequence[ManagedValue], None, None]:
    unique: list[Type[ManagedValue]] = []
    for value in values:
        if value not in unique:
            unique.append(value)
    yield [value(config) for value in unique]


@asynccontextmanager
async def AsyncManagedValuesManager(
    values: Sequence[Type[ManagedValue]],
    config: RunnableConfig,
) -> AsyncGenerator[Sequence[ManagedValue], None]:
    unique: list[Type[ManagedValue]] = []
    for value in values:
        if value not in unique:
            unique.append(value)
    yield [value(config) for value in unique]
