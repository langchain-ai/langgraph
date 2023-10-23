from abc import ABC, abstractmethod
from contextlib import asynccontextmanager, contextmanager
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

Value = TypeVar("Value")
Update = TypeVar("Update")


class EmptyChannelError(Exception):
    """Raised when attempting to get the value of a channel that hasn't been updated
    for the first time yet."""

    pass


class InvalidUpdateError(Exception):
    """Raised when attempting to update a channel with an invalid sequence of updates."""

    pass


class Channel(Generic[Value, Update], ABC):
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
    def empty(self, checkpoint: Optional[str] = None) -> Generator[Self, None, None]:
        """Return a new identical channel, optionally initialized from a checkpoint."""

    @asynccontextmanager
    async def aempty(
        self, checkpoint: Optional[str] = None
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
    def checkpoint(self) -> str | None:
        """Return a string representation of the channel's current state,
        or None if the channel doesn't support checkpoints.

        Raises EmptyChannelError if the channel is empty (never updated yet)."""


@contextmanager
def ChannelsManager(
    channels: Mapping[str, Channel]
) -> Generator[Mapping[str, Channel], None, None]:
    """Manage channels for the lifetime of a Pregel invocation (multiple steps)."""
    empty = {k: v.empty() for k, v in channels.items()}
    try:
        yield {k: v.__enter__() for k, v in empty.items()}
    finally:
        for v in empty.values():
            v.__exit__(None, None, None)


@asynccontextmanager
async def AsyncChannelsManager(
    channels: Mapping[str, Channel]
) -> AsyncGenerator[Mapping[str, Channel], None]:
    """Manage channels for the lifetime of a Pregel invocation (multiple steps)."""
    empty = {k: v.aempty() for k, v in channels.items()}
    try:
        yield {k: await v.__aenter__() for k, v in empty.items()}
    finally:
        for v in empty.values():
            await v.__aexit__(None, None, None)
