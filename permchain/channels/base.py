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

from permchain.constants import CHECKPOINT_KEY_TS, CHECKPOINT_KEY_VERSION

Value = TypeVar("Value")
Update = TypeVar("Update")
Checkpoint = TypeVar("Checkpoint")


class EmptyChannelError(Exception):
    """Raised when attempting to get the value of a channel that hasn't been updated
    for the first time yet."""

    pass


class InvalidUpdateError(Exception):
    """Raised when attempting to update a channel with an invalid sequence of updates."""

    pass


class BaseChannel(Generic[Value, Update, Checkpoint], ABC):
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
    def empty(
        self, checkpoint: Optional[Checkpoint] = None
    ) -> Generator[Self, None, None]:
        """Return a new identical channel, optionally initialized from a checkpoint."""

    @asynccontextmanager
    async def aempty(
        self, checkpoint: Optional[Checkpoint] = None
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
    def checkpoint(self) -> Checkpoint | None:
        """Return a string representation of the channel's current state.

        Raises EmptyChannelError if the channel is empty (never updated yet),
        or doesn't supportcheckpoints."""


@contextmanager
def ChannelsManager(
    channels: Mapping[str, BaseChannel],
    checkpoint: Optional[Mapping[str, Any]],
) -> Generator[Mapping[str, BaseChannel], None, None]:
    """Manage channels for the lifetime of a Pregel invocation (multiple steps)."""
    # TODO use https://docs.python.org/3/library/contextlib.html#contextlib.ExitStack
    checkpoint = checkpoint or {}
    empty = {k: v.empty(checkpoint.get(k)) for k, v in channels.items()}
    try:
        yield {k: v.__enter__() for k, v in empty.items()}
    finally:
        for v in empty.values():
            v.__exit__(None, None, None)


@asynccontextmanager
async def AsyncChannelsManager(
    channels: Mapping[str, BaseChannel],
    checkpoint: Optional[Mapping[str, Any]],
) -> AsyncGenerator[Mapping[str, BaseChannel], None]:
    """Manage channels for the lifetime of a Pregel invocation (multiple steps)."""
    checkpoint = checkpoint or {}
    empty = {k: v.aempty(checkpoint.get(k)) for k, v in channels.items()}
    try:
        yield {k: await v.__aenter__() for k, v in empty.items()}
    finally:
        for v in empty.values():
            await v.__aexit__(None, None, None)


def create_checkpoint(channels: Mapping[str, BaseChannel]) -> Mapping[str, Any]:
    """Create a checkpoint for the given channels."""
    checkpoint = {
        CHECKPOINT_KEY_VERSION: 1,
        CHECKPOINT_KEY_TS: datetime.now(timezone.utc).isoformat(),
    }
    for k, v in channels.items():
        try:
            checkpoint[k] = v.checkpoint()
        except EmptyChannelError:
            pass
    return checkpoint
