import asyncio
from abc import ABC, abstractmethod
from contextlib import AsyncExitStack, ExitStack, asynccontextmanager, contextmanager
from inspect import isclass
from typing import (
    TYPE_CHECKING,
    Any,
    AsyncGenerator,
    Generator,
    Generic,
    Sequence,
    Type,
    TypeVar,
)

from langchain_core.runnables import RunnableConfig
from typing_extensions import Self, TypeGuard

from langgraph.pregel.types import PregelTaskDescription

if TYPE_CHECKING:
    from langgraph.pregel import Pregel

V = TypeVar("V")


class ManagedValue(ABC, Generic[V]):
    def __init__(self, config: RunnableConfig, graph: "Pregel") -> None:
        self.config = config
        self.graph = graph

    @classmethod
    @contextmanager
    def enter(
        cls, config: RunnableConfig, graph: "Pregel"
    ) -> Generator[Self, None, None]:
        try:
            value = cls(config, graph)
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
    async def aenter(
        cls, config: RunnableConfig, graph: "Pregel"
    ) -> AsyncGenerator[Self, None]:
        try:
            value = cls(config, graph)
            yield value
        finally:
            # because managed value and Pregel have reference to each other
            # let's make sure to break the reference on exit
            try:
                del value
            except UnboundLocalError:
                pass

    @abstractmethod
    def __call__(self, step: int, task: PregelTaskDescription) -> V:
        ...


def is_managed_value(value: Any) -> TypeGuard[Type[ManagedValue]]:
    return isclass(value) and issubclass(value, ManagedValue)


@contextmanager
def ManagedValuesManager(
    values: Sequence[Type[ManagedValue]],
    config: RunnableConfig,
    graph: "Pregel",
) -> Generator[Sequence[ManagedValue], None, None]:
    with ExitStack() as stack:
        unique: list[Type[ManagedValue]] = []
        for value in values:
            if value not in unique:
                unique.append(value)

        yield [stack.enter_context(value.enter(config, graph)) for value in unique]


@asynccontextmanager
async def AsyncManagedValuesManager(
    values: Sequence[Type[ManagedValue]],
    config: RunnableConfig,
    graph: "Pregel",
) -> AsyncGenerator[Sequence[ManagedValue], None]:
    async with AsyncExitStack() as stack:
        unique: list[Type[ManagedValue]] = []
        for value in values:
            if value not in unique:
                unique.append(value)

        yield await asyncio.gather(
            *(
                stack.enter_async_context(value.aenter(config, graph))
                for value in unique
            )
        )
