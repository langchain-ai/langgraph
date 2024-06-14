from abc import ABC, abstractmethod
from contextlib import asynccontextmanager, contextmanager
from datetime import datetime, timezone
from typing import (
    Any,
    AsyncGenerator,
    Generator,
    Generic,
    Mapping,
    Optional,
    Sequence,
    TypeVar,
)

from typing_extensions import Self

from langgraph.checkpoint.base import Checkpoint
from langgraph.checkpoint.id import uuid6
from langgraph.errors import EmptyChannelError, InvalidUpdateError

Value = TypeVar("Value")
Update = TypeVar("Update")
C = TypeVar("C")


class BaseChannel(Generic[Value, Update, C], ABC):
    @property
    @abstractmethod
    def ValueType(self) -> Any:
        """The type of the value stored in the channel."""

    @property
    @abstractmethod
    def UpdateType(self) -> Any:
        """The type of the update received by the channel."""

    # serialize/deserialize methods

    @abstractmethod
    def checkpoint(self) -> Optional[C]:
        """Return a serializable representation of the channel's current state.
        Raises EmptyChannelError if the channel is empty (never updated yet),
        or doesn't support checkpoints."""

    @contextmanager
    @abstractmethod
    def from_checkpoint(
        self, checkpoint: Optional[C] = None
    ) -> Generator[Self, None, None]:
        """Return a new identical channel, optionally initialized from a checkpoint.
        If the checkpoint contains complex data structures, they should be copied."""

    @asynccontextmanager
    async def afrom_checkpoint(
        self, checkpoint: Optional[C] = None
    ) -> AsyncGenerator[Self, None]:
        """Return a new identical channel, optionally initialized from a checkpoint.
        If the checkpoint contains complex data structures, they should be copied."""
        with self.from_checkpoint(checkpoint) as value:
            yield value

    # state methods

    @abstractmethod
    def update(self, values: Sequence[Update]) -> None:
        """Update the channel's value with the given sequence of updates.
        The order of the updates in the sequence is arbitrary.
        This method is called by Pregel for all channels at the end of each step.
        If there are no updates, it is called with an empty sequence.
        Raises InvalidUpdateError if the sequence of updates is invalid."""

    @abstractmethod
    def get(self) -> Value:
        """Return the current value of the channel.

        Raises EmptyChannelError if the channel is empty (never updated yet)."""

    def consume(self) -> None:
        """Mark the current value of the channel as consumed. By default, no-op.
        This is called by Pregel before the start of the next step, for all
        channels that triggered a node.
        """
        pass


@contextmanager
def ChannelsManager(
    channels: Mapping[str, BaseChannel],
    checkpoint: Checkpoint,
) -> Generator[Mapping[str, BaseChannel], None, None]:
    """Manage channels for the lifetime of a Pregel invocation (multiple steps)."""
    # TODO use https://docs.python.org/3/library/contextlib.html#contextlib.ExitStack
    empty = {
        k: v.from_checkpoint(checkpoint["channel_values"].get(k))
        for k, v in channels.items()
    }
    try:
        yield {k: v.__enter__() for k, v in empty.items()}
    finally:
        for v in empty.values():
            v.__exit__(None, None, None)


@asynccontextmanager
async def AsyncChannelsManager(
    channels: Mapping[str, BaseChannel],
    checkpoint: Checkpoint,
) -> AsyncGenerator[Mapping[str, BaseChannel], None]:
    """Manage channels for the lifetime of a Pregel invocation (multiple steps)."""
    empty = {
        k: v.afrom_checkpoint(checkpoint["channel_values"].get(k))
        for k, v in channels.items()
    }
    try:
        yield {k: await v.__aenter__() for k, v in empty.items()}
    finally:
        for v in empty.values():
            await v.__aexit__(None, None, None)


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
    )


__all__ = [
    "BaseChannel",
    "ChannelsManager",
    "AsyncChannelsManager",
    "create_checkpoint",
    "EmptyChannelError",
    "InvalidUpdateError",
]
