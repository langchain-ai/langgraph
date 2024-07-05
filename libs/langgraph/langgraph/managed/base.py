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
    NamedTuple,
    Type,
    TypeVar,
    Union,
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
        cls, config: RunnableConfig, graph: "Pregel", **kwargs: Any
    ) -> Generator[Self, None, None]:
        try:
            value = cls(config, graph, **kwargs)
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
        cls, config: RunnableConfig, graph: "Pregel", **kwargs: Any
    ) -> AsyncGenerator[Self, None]:
        try:
            value = cls(config, graph, **kwargs)
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


class ConfiguredManagedValue(NamedTuple):
    cls: Type[ManagedValue]
    kwargs: dict[str, Any]


ManagedValueSpec = Union[Type[ManagedValue], ConfiguredManagedValue]

ManagedValueMapping = dict[str, ManagedValue]


def is_managed_value(value: Any) -> TypeGuard[ManagedValueSpec]:
    return (isclass(value) and issubclass(value, ManagedValue)) or isinstance(
        value, ConfiguredManagedValue
    )


@contextmanager
def ManagedValuesManager(
    values: dict[str, ManagedValueSpec],
    config: RunnableConfig,
    graph: "Pregel",
) -> Generator[ManagedValueMapping, None, None]:
    if values:
        with ExitStack() as stack:
            yield {
                key: stack.enter_context(
                    value.cls.enter(config, graph, **value.kwargs)
                    if isinstance(value, ConfiguredManagedValue)
                    else value.enter(config, graph)
                )
                for key, value in values.items()
            }
    else:
        yield {}


@asynccontextmanager
async def AsyncManagedValuesManager(
    values: dict[str, ManagedValueSpec],
    config: RunnableConfig,
    graph: "Pregel",
) -> AsyncGenerator[ManagedValueMapping, None]:
    if values:
        async with AsyncExitStack() as stack:
            # create enter tasks with reference to spec
            tasks = {
                asyncio.create_task(
                    stack.enter_async_context(
                        value.cls.aenter(config, graph, **value.kwargs)
                        if isinstance(value, ConfiguredManagedValue)
                        else value.aenter(config, graph)
                    )
                ): key
                for key, value in values.items()
            }
            # wait for all enter tasks
            done, _ = await asyncio.wait(tasks, return_when=asyncio.ALL_COMPLETED)
            # build mapping from spec to result
            yield {tasks[task]: task.result() for task in done}
    else:
        yield {}
