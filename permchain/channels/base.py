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

from permchain.checkpoint.base import Checkpoint

Value = TypeVar("Value")
Update = TypeVar("Update")
C = TypeVar("C")


class EmptyChannelError(Exception):
    """Raised when attempting to get the value of a channel that hasn't been updated
    for the first time yet."""

    pass


class InvalidUpdateError(Exception):
    """Raised when attempting to update a channel with an invalid sequence of updates."""

    pass


class BaseChannel(Generic[Value, Update, C], ABC):
    @property
    @abstractmethod
    def ValueType(self) -> Any:
        """The type of the value stored in the channel."""

    @property
    @abstractmethod
    def UpdateType(self) -> Any:
        """The type of the update received by the channel."""

    @contextmanager
    @abstractmethod
    def empty(self, checkpoint: Optional[C] = None) -> Generator[Self, None, None]:
        """Return a new identical channel, optionally initialized from a checkpoint."""

    @asynccontextmanager
    async def aempty(
        self, checkpoint: Optional[C] = None
    ) -> AsyncGenerator[Self, None]:
        """Return a new identical channel, optionally initialized from a checkpoint."""
        with self.empty(checkpoint) as value:
            yield value

    @abstractmethod
    def update(self, values: Sequence[Update]) -> None:
        """Update the channel's value with the given sequence of updates.
        The order of the updates in the sequence is arbitrary.

        Raises InvalidUpdateError if the sequence of updates is invalid."""

    @abstractmethod
    def get(self) -> Value:
        """Return the current value of the channel.

        Raises EmptyChannelError if the channel is empty (never updated yet)."""

    @abstractmethod
    def checkpoint(self) -> C | None:
        """Return a string representation of the channel's current state.

        Raises EmptyChannelError if the channel is empty (never updated yet),
        or doesn't supportcheckpoints."""


@contextmanager
def ChannelsManager(
    channels: Mapping[str, BaseChannel],
    checkpoint: Checkpoint,
) -> Generator[Mapping[str, BaseChannel], None, None]:
    """Manage channels for the lifetime of a Pregel invocation (multiple steps)."""
    # TODO use https://docs.python.org/3/library/contextlib.html#contextlib.ExitStack
    empty = {
        k: v.empty(checkpoint["channel_values"].get(k)) for k, v in channels.items()
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
        k: v.aempty(checkpoint["channel_values"].get(k)) for k, v in channels.items()
    }
    try:
        yield {k: await v.__aenter__() for k, v in empty.items()}
    finally:
        for v in empty.values():
            await v.__aexit__(None, None, None)


def create_checkpoint(
    checkpoint: Checkpoint, channels: Mapping[str, BaseChannel]
) -> Checkpoint:
    """Create a checkpoint for the given channels."""
    checkpoint = Checkpoint(
        v=1,
        ts=datetime.now(timezone.utc).isoformat(),
        channel_values=checkpoint["channel_values"],
        channel_versions=checkpoint["channel_versions"],
        versions_seen=checkpoint["versions_seen"],
    )
    for k, v in channels.items():
        try:
            checkpoint["channel_values"][k] = v.checkpoint()
        except EmptyChannelError:
            pass
    return checkpoint
