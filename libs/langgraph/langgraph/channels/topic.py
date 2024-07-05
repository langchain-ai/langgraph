from contextlib import contextmanager
from typing import Any, Generator, Generic, Iterator, Optional, Sequence, Type, Union

from langchain_core.runnables import RunnableConfig
from typing_extensions import Self

from langgraph.channels.base import BaseChannel, Value
from langgraph.errors import EmptyChannelError


def flatten(values: Sequence[Union[Value, list[Value]]]) -> Iterator[Value]:
    for value in values:
        if isinstance(value, list):
            yield from value
        else:
            yield value


class Topic(
    Generic[Value],
    BaseChannel[
        Sequence[Value], Union[Value, list[Value]], tuple[set[Value], list[Value]]
    ],
):
    """A configurable PubSub Topic.

    Args:
        typ: The type of the value stored in the channel.
        unique: Whether to discard duplicate values.
        accumulate: Whether to accumulate values across steps. If False, the channel will be emptied after each step.
    """

    def __init__(
        self, typ: Type[Value], unique: bool = False, accumulate: bool = False
    ) -> None:
        # attrs
        self.typ = typ
        self.unique = unique
        self.accumulate = accumulate
        # state
        self.seen = set[Value]()
        self.values = list[Value]()

    @property
    def ValueType(self) -> Any:
        """The type of the value stored in the channel."""
        return Sequence[self.typ]  # type: ignore[name-defined]

    @property
    def UpdateType(self) -> Any:
        """The type of the update received by the channel."""
        return Union[self.typ, list[self.typ]]  # type: ignore[name-defined]

    def checkpoint(self) -> tuple[set[Value], list[Value]]:
        return (self.seen, self.values)

    @contextmanager
    def from_checkpoint(
        self,
        checkpoint: Optional[tuple[set[Value], list[Value]]],
        config: RunnableConfig,
    ) -> Generator[Self, None, None]:
        empty = self.__class__(self.typ, self.unique, self.accumulate)
        if checkpoint is not None:
            empty.seen = checkpoint[0].copy()
            empty.values = checkpoint[1].copy()
        try:
            yield empty
        finally:
            pass

    def update(self, values: Sequence[Union[Value, list[Value]]]) -> None:
        current = list(self.values)
        if not self.accumulate:
            self.values = list[Value]()
        if flat_values := flatten(values):
            if self.unique:
                for value in flat_values:
                    if value not in self.seen:
                        self.seen.add(value)
                        self.values.append(value)
            else:
                self.values.extend(flat_values)
        return self.values != current

    def get(self) -> Sequence[Value]:
        if self.values:
            return list(self.values)
        else:
            raise EmptyChannelError
