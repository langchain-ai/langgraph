import json
from contextlib import contextmanager
from typing import Any, Generator, Generic, Iterator, Optional, Sequence, Type, Union

from typing_extensions import Self

from permchain.channels.base import BaseChannel, Value


def flatten(values: Sequence[Value | list[Value]]) -> Iterator[Value]:
    for value in values:
        if isinstance(value, list):
            yield from value
        else:
            yield value


class Topic(Generic[Value], BaseChannel[Sequence[Value], Value | list[Value]]):
    """A configurable PubSub Topic.

    Args:
        typ: The type of the value stored in the channel.
        unique: Whether to discard duplicate values.
        accumulate: Whether to accummulate values across steps. If False, the channel will be emptied after each step.
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
    def UpdateType(self) -> Type[Value]:
        """The type of the update received by the channel."""
        return Union[self.typ, list[self.typ]]  # type: ignore[name-defined]

    @contextmanager
    def empty(self, checkpoint: Optional[str] = None) -> Generator[Self, None, None]:
        empty = self.__class__(self.typ, self.unique, self.accumulate)
        if checkpoint is not None:
            parsed = json.loads(checkpoint)
            empty.seen = set(parsed["seen"])
            empty.values = list(parsed["values"])
        try:
            yield empty
        finally:
            pass

    def update(self, values: Sequence[Value | list[Value]]) -> None:
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

    def get(self) -> Sequence[Value]:
        return list(self.values)

    def checkpoint(self) -> str:
        return json.dumps({"seen": list(self.seen), "values": self.values})
