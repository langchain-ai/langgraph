from collections.abc import Sequence
from typing import (
    Any,
    Protocol,
    TypeVar,
    runtime_checkable,
)

from typing_extensions import Self

ERROR = "__error__"
SCHEDULED = "__scheduled__"
INTERRUPT = "__interrupt__"
RESUME = "__resume__"
TASKS = "__pregel_tasks"

Value = TypeVar("Value", covariant=True)
Update = TypeVar("Update", contravariant=True)
C = TypeVar("C")


class ChannelProtocol(Protocol[Value, Update, C]):
    """Protocol for channel state management in the checkpoint system.

    Mirrors `langgraph.channels.base.BaseChannel`. Channels hold the state
    values that are persisted in checkpoints and updated during graph execution.
    """

    @property
    def ValueType(self) -> Any:
        """The type of the value stored in the channel."""
        ...

    @property
    def UpdateType(self) -> Any:
        """The type of the update accepted by the channel."""
        ...

    def checkpoint(self) -> C | None:
        """Return the channel's current state for serialization into a checkpoint."""
        ...

    def from_checkpoint(self, checkpoint: C | None) -> Self:
        """Restore the channel from a previously saved checkpoint value.

        Args:
            checkpoint: The serialized state returned by a prior `checkpoint()` call,
                or `None` if no prior state exists.

        Returns:
            A new channel instance initialized from the given checkpoint.
        """
        ...

    def update(self, values: Sequence[Update]) -> bool:
        """Apply a sequence of updates to the channel.

        Args:
            values: The updates to apply to the channel's current value.

        Returns:
            True if the channel's value was modified, False otherwise.
        """
        ...

    def get(self) -> Value:
        """Return the current value of the channel."""
        ...

    def consume(self) -> bool:
        """Consume the channel's current value, resetting it to its empty state.

        Returns:
            True if the channel held a value that was consumed, False if it was
            already empty.
        """
        ...


@runtime_checkable
class SendProtocol(Protocol):
    """Protocol for sending messages to specific graph nodes.

    Mirrors `langgraph.constants.Send`. Used to route data to a named node
    during graph execution.
    """

    node: str
    """The name of the target node."""

    arg: Any
    """The data payload to send to the node."""

    def __hash__(self) -> int: ...

    def __repr__(self) -> str: ...

    def __eq__(self, value: object) -> bool: ...
