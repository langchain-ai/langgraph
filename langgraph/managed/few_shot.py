from contextlib import asynccontextmanager, contextmanager
from typing import (
    TYPE_CHECKING,
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
from langgraph.checkpoint.base import CheckpointTuple
from langgraph.managed.base import ManagedValue, V
from langgraph.pregel.io import read_channels
from langgraph.pregel.types import PregelTaskDescription

if TYPE_CHECKING:
    from langgraph.pregel import Pregel


class FewShotExamples(ManagedValue[Sequence[V]], Generic[V]):
    examples: list[V]

    def iter(self, score: int = 1, k: int = 5) -> Iterator[V]:
        for example in self.graph.checkpointer.search({"score": score}, limit=k):
            with ChannelsManager(self.graph.channels, example.checkpoint) as channels:
                yield read_channels(channels, self.graph.output_channels)

    async def aiter(self, score: int = 1, k: int = 5) -> AsyncIterator[V]:
        async for example in self.graph.checkpointer.asearch({"score": score}, limit=k):
            async with AsyncChannelsManager(
                self.graph.channels, example.checkpoint
            ) as channels:
                yield read_channels(channels, self.graph.output_channels)

    @classmethod
    @contextmanager
    def enter(
        cls, config: RunnableConfig, graph: "Pregel"
    ) -> Generator[Self, None, None]:
        with super().enter(config, graph) as value:
            value.examples = list(value.iter())
            yield value

    @classmethod
    @asynccontextmanager
    async def aenter(
        cls, config: RunnableConfig, graph: "Pregel"
    ) -> AsyncGenerator[Self, None]:
        async with super().aenter(config, graph) as value:
            value.examples = [e async for e in value.aiter()]
            yield value

    def __call__(self, step: int, task: PregelTaskDescription) -> Sequence[V]:
        return self.examples
