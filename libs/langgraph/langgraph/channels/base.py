from abc import ABC, abstractmethod
from collections.abc import Sequence
from typing import Any, Generic, TypeVar

from typing_extensions import Self

from langgraph.constants import MISSING
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

    def copy(self) -> Self:
        """Return a copy of the channel.
        By default, delegates to checkpoint() and from_checkpoint().
        Subclasses can override this method with a more efficient implementation."""
        return self.from_checkpoint(self.checkpoint())

    def checkpoint(self) -> C:
        """Return a serializable representation of the channel's current state.
        Raises EmptyChannelError if the channel is empty (never updated yet),
        or doesn't support checkpoints."""
        try:
            return self.get()
        except EmptyChannelError:
            return MISSING

    @abstractmethod
    def from_checkpoint(self, checkpoint: C) -> Self:
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

    def is_available(self) -> bool:
        """Return True if the channel is available (not empty), False otherwise.
        Subclasses should override this method to provide a more efficient
        implementation than calling get() and catching EmptyChannelError.
        """
        try:
            self.get()
            return True
        except EmptyChannelError:
            return False


__all__ = [
    "BaseChannel",
    "EmptyChannelError",
    "InvalidUpdateError",
]
