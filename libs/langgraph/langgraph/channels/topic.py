from collections.abc import Iterator, Sequence
from typing import Any, Generic, Union

from typing_extensions import Self

from langgraph.channels.base import BaseChannel, Value
from langgraph.constants import MISSING
from langgraph.errors import EmptyChannelError


def flatten(values: Sequence[Union[Value, list[Value]]]) -> Iterator[Value]:
    for value in values:
        if isinstance(value, list):
            yield from value
        else:
            yield value


class Topic(
    Generic[Value],
    BaseChannel[Sequence[Value], Union[Value, list[Value]], list[Value]],
):
    """A configurable PubSub Topic.

    Args:
        typ: The type of the value stored in the channel.
        accumulate: Whether to accumulate values across steps. If False, the channel will be emptied after each step.
    """

    __slots__ = ("values", "accumulate")

    def __init__(self, typ: type[Value], accumulate: bool = False) -> None:
        super().__init__(typ)
        # attrs
        self.accumulate = accumulate
        # state
        self.values = list[Value]()

    def __eq__(self, value: object) -> bool:
        return isinstance(value, Topic) and value.accumulate == self.accumulate

    @property
    def ValueType(self) -> Any:
        """The type of the value stored in the channel."""
        return Sequence[self.typ]  # type: ignore[name-defined]

    @property
    def UpdateType(self) -> Any:
        """The type of the update received by the channel."""
        return Union[self.typ, list[self.typ]]  # type: ignore[name-defined]

    def copy(self) -> Self:
        """Return a copy of the channel."""
        empty = self.__class__(self.typ, self.accumulate)
        empty.key = self.key
        empty.values = self.values.copy()
        return empty

    def checkpoint(self) -> list[Value]:
        return self.values

    def from_checkpoint(self, checkpoint: list[Value]) -> Self:
        empty = self.__class__(self.typ, self.accumulate)
        empty.key = self.key
        if checkpoint is not MISSING:
            if isinstance(checkpoint, tuple):
                # backwards compatibility
                empty.values = checkpoint[1]
            else:
                empty.values = checkpoint
        return empty

    def update(self, values: Sequence[Union[Value, list[Value]]]) -> bool:
        current = list(self.values)
        if not self.accumulate:
            self.values = list[Value]()
        if flat_values := flatten(values):
            self.values.extend(flat_values)
        return self.values != current

    def get(self) -> Sequence[Value]:
        if self.values:
            return list(self.values)
        else:
            raise EmptyChannelError

    def is_available(self) -> bool:
        return bool(self.values)
