from contextlib import asynccontextmanager, contextmanager
from typing import (
    TYPE_CHECKING,
    Any,
    AsyncGenerator,
    AsyncIterator,
    Generator,
    Generic,
    Iterator,
    Sequence,
)

from langchain_core.runnables import RunnableConfig
from typing_extensions import Self

from langgraph.channels.base import AsyncChannelsManager, ChannelsManager
from langgraph.managed.base import ConfiguredManagedValue, ManagedValue, V
from langgraph.pregel import Pregel
from langgraph.pregel.io import read_channels
from langgraph.pregel.types import PregelTaskDescription

if TYPE_CHECKING:
    from langgraph.pregel import Pregel


class FewShotExamples(ManagedValue[Sequence[V]], Generic[V]):
    examples: list[V]

    def __init__(
        self,
        config: RunnableConfig,
        graph: Pregel,
        k: int = 5,
        metadata_filter: dict[str, Any] = None,
    ) -> None:
        super().__init__(config, graph)
        self.k = k
        self.metadata_filter = metadata_filter or {}

    @classmethod
    def configure(
        cls, k: int = 5, metadata_filter: dict[str, Any] = None
    ) -> ConfiguredManagedValue:
        return ConfiguredManagedValue(
            cls,
            {
                "k": k,
                "metadata_filter": metadata_filter,
            },
        )

    def iter(self, score: int = 1) -> Iterator[V]:
        for example in self.graph.checkpointer.search(
            {"score": score, **self.metadata_filter}, limit=self.k
        ):
            with ChannelsManager(self.graph.channels, example.checkpoint) as channels:
                yield read_channels(channels, self.graph.output_channels)

    async def aiter(self, score: int = 1) -> AsyncIterator[V]:
        async for example in self.graph.checkpointer.asearch(
            {"score": score, **self.metadata_filter}, limit=self.k
        ):
            async with AsyncChannelsManager(
                self.graph.channels, example.checkpoint
            ) as channels:
                yield read_channels(channels, self.graph.output_channels)

    @classmethod
    @contextmanager
    def enter(
        cls, config: RunnableConfig, graph: "Pregel", **kwargs: Any
    ) -> Generator[Self, None, None]:
        with super().enter(config, graph, **kwargs) as value:
            value.examples = list(value.iter())
            yield value

    @classmethod
    @asynccontextmanager
    async def aenter(
        cls, config: RunnableConfig, graph: "Pregel", **kwargs: Any
    ) -> AsyncGenerator[Self, None]:
        async with super().aenter(config, graph, **kwargs) as value:
            value.examples = [e async for e in value.aiter()]
            yield value

    def __call__(self, step: int, task: PregelTaskDescription) -> Sequence[V]:
        return self.examples
