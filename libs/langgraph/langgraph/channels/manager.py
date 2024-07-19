from contextlib import AsyncExitStack, ExitStack, asynccontextmanager, contextmanager
from datetime import datetime, timezone
from typing import Any, AsyncGenerator, Generator, Mapping

from langchain_core.runnables import RunnableConfig

from langgraph.channels.base import BaseChannel
from langgraph.checkpoint.base import Checkpoint
from langgraph.checkpoint.id import uuid6
from langgraph.errors import EmptyChannelError


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


def create_checkpoint(
    checkpoint: Checkpoint, channels: Mapping[str, BaseChannel], step: int
) -> Checkpoint:
    """Create a checkpoint for the given channels."""
    ts = datetime.now(timezone.utc).isoformat()
    values: dict[str, Any] = {}
    for k, v in channels.items():
        try:
            values[k] = v.checkpoint()
        except EmptyChannelError:
            pass
    return Checkpoint(
        v=1,
        ts=ts,
        id=str(uuid6(clock_seq=step)),
        channel_values=values,
        channel_versions=checkpoint["channel_versions"],
        versions_seen=checkpoint["versions_seen"],
        pending_sends=checkpoint.get("pending_sends", []),
        # checkpoints are saved only at the end of a step, ie. when current tasks should be cleared
        current_tasks={},
    )
