from contextlib import AsyncExitStack, ExitStack, asynccontextmanager, contextmanager
from typing import AsyncGenerator, Generator, Mapping

from langchain_core.runnables import RunnableConfig

from langgraph.channels.base import BaseChannel
from langgraph.checkpoint.base import Checkpoint


@contextmanager
def ChannelsManager(
    channels: Mapping[str, BaseChannel],
    checkpoint: Checkpoint,
    config: RunnableConfig,
) -> Generator[Mapping[str, BaseChannel], None, None]:
    """Manage channels for the lifetime of a Pregel invocation (multiple steps)."""
    with ExitStack() as stack:
        yield {
            k: stack.enter_context(
                v.from_checkpoint(checkpoint["channel_values"].get(k), config)
            )
            for k, v in channels.items()
        }


@asynccontextmanager
async def AsyncChannelsManager(
    channels: Mapping[str, BaseChannel],
    checkpoint: Checkpoint,
    config: RunnableConfig,
) -> AsyncGenerator[Mapping[str, BaseChannel], None]:
    """Manage channels for the lifetime of a Pregel invocation (multiple steps)."""
    async with AsyncExitStack() as stack:
        yield {
            k: await stack.enter_async_context(
                v.afrom_checkpoint(checkpoint["channel_values"].get(k), config)
            )
            for k, v in channels.items()
        }
