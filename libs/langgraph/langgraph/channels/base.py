from abc import ABC, abstractmethod
from typing import Any, Generic, Optional, Sequence, TypeVar

from typing_extensions import Self

from langgraph.errors import EmptyChannelError, InvalidUpdateError

Value = TypeVar("Value")
Update = TypeVar("Update")
C = TypeVar("C")


class BaseChannel(Generic[Value, Update, C], ABC):
    __slots__ = ("key", "typ")

    def __init__(self, typ: Any, key: str = "") -> None:
        self.typ = typ
        self.key = key

    @property
    @abstractmethod
    def ValueType(self) -> Any:
        """The type of the value stored in the channel."""

    @property
    @abstractmethod
    def UpdateType(self) -> Any:
        """The type of the update received by the channel."""

    # serialize/deserialize methods

    def checkpoint(self) -> Optional[C]:
        """Return a serializable representation of the channel's current state.
        Raises EmptyChannelError if the channel is empty (never updated yet),
        or doesn't support checkpoints."""
        return self.get()

    @abstractmethod
    def from_checkpoint(self, checkpoint: Optional[C]) -> Self:
        """Return a new identical channel, optionally initialized from a checkpoint.
        If the checkpoint contains complex data structures, they should be copied."""

    # state methods

    @abstractmethod
    def update(self, values: Sequence[Update]) -> bool:
        """Update the channel's value with the given sequence of updates.
        The order of the updates in the sequence is arbitrary.
        This method is called by Pregel for all channels at the end of each step.
        If there are no updates, it is called with an empty sequence.
        Raises InvalidUpdateError if the sequence of updates is invalid.
        Returns True if the channel was updated, False otherwise."""

    @abstractmethod
    def get(self) -> Value:
        """Return the current value of the channel.

        Raises EmptyChannelError if the channel is empty (never updated yet)."""

    def consume(self) -> bool:
        """Mark the current value of the channel as consumed. By default, no-op.
        This is called by Pregel before the start of the next step, for all
        channels that triggered a node. If the channel was updated, return True.
        """
        return False


__all__ = [
    "BaseChannel",
    "EmptyChannelError",
    "InvalidUpdateError",
]
