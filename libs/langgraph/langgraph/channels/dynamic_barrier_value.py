from collections.abc import Sequence, Set
from typing import Any, Generic, NamedTuple, Optional, Union

from typing_extensions import Self

from langgraph.channels.base import BaseChannel, Value
from langgraph.constants import MISSING
from langgraph.errors import EmptyChannelError, InvalidUpdateError


class WaitForNames(NamedTuple):
    names: Set[Any]


class DynamicBarrierValue(
    Generic[Value], BaseChannel[Value, Union[Value, WaitForNames], Set[Value]]
):
    """A channel that switches between two states

    - in the "priming" state it can't be read from.
        - if it receives a WaitForNames update, it switches to the "waiting" state.
    - in the "waiting" state it collects named values until all are received.
        - once all named values are received, it can be read once, and it switches
          back to the "priming" state.
    """

    __slots__ = ("names", "seen")

    names: Optional[Set[Value]]
    seen: set[Value]

    def __init__(self, typ: type[Value]) -> None:
        super().__init__(typ)
        self.names = None
        self.seen = set()

    def __eq__(self, value: object) -> bool:
        return isinstance(value, DynamicBarrierValue) and value.names == self.names

    @property
    def ValueType(self) -> type[Value]:
        """The type of the value stored in the channel."""
        return self.typ

    @property
    def UpdateType(self) -> type[Value]:
        """The type of the update received by the channel."""
        return self.typ

    def copy(self) -> Self:
        """Return a copy of the channel."""
        empty = self.__class__(self.typ)
        empty.key = self.key
        empty.names = self.names
        empty.seen = self.seen.copy()
        return empty

    def checkpoint(self) -> tuple[Optional[Set[Value]], set[Value]]:
        return (self.names, self.seen)

    def from_checkpoint(
        self, checkpoint: tuple[Optional[Set[Value]], set[Value]]
    ) -> Self:
        empty = self.__class__(self.typ)
        empty.key = self.key
        if checkpoint is not MISSING:
            names, seen = checkpoint
            empty.names = names if names is not None else None
            empty.seen = seen
        return empty

    def update(self, values: Sequence[Union[Value, WaitForNames]]) -> bool:
        if wait_for_names := [v for v in values if isinstance(v, WaitForNames)]:
            if len(wait_for_names) > 1:
                raise InvalidUpdateError(
                    f"At key '{self.key}': Received multiple WaitForNames updates in the same step."
                )
            self.names = wait_for_names[0].names
            return True
        elif self.names is not None:
            updated = False
            for value in values:
                assert not isinstance(value, WaitForNames)
                if value in self.names:
                    if value not in self.seen:
                        self.seen.add(value)
                        updated = True
                else:
                    raise InvalidUpdateError(f"Value {value} not in {self.names}")
            return updated

    def get(self) -> Value:
        if self.seen != self.names:
            raise EmptyChannelError()
        return None

    def is_available(self) -> bool:
        return self.seen == self.names

    def consume(self) -> bool:
        if self.seen == self.names:
            self.seen = set()
            self.names = None
            return True
        return False
