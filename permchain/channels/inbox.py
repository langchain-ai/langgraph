import json
from contextlib import contextmanager
from typing import (
    Any,
    FrozenSet,
    Generator,
    Generic,
    Iterator,
    Optional,
    Sequence,
    Type,
    Union,
)

from typing_extensions import Self

from permchain.channels.base import Channel, EmptyChannelError, Value


def flatten(values: Sequence[Value | list[Value]]) -> Iterator[Value]:
    for value in values:
        if isinstance(value, list):
            yield from value
        else:
            yield value


class Inbox(Generic[Value], Channel[Sequence[Value], Value | list[Value]]):
    """Stores all values received, resets in each step."""

    def __init__(self, typ: Type[Value]) -> None:
        self.typ = typ

    @property
    def ValueType(self) -> Type[Sequence[Value]]:
        """The type of the value stored in the channel."""
        return Sequence[self.typ]  # type: ignore[name-defined]

    @property
    def UpdateType(self) -> Any:
        """The type of the update received by the channel."""
        return Union[self.typ, Sequence[self.typ]]  # type: ignore[name-defined]

    @contextmanager
    def empty(self, checkpoint: Optional[str] = None) -> Generator[Self, None, None]:
        empty = self.__class__(self.typ)
        if checkpoint is not None:
            empty.queue = tuple(json.loads(checkpoint))
        try:
            yield empty
        finally:
            try:
                del empty.queue
            except AttributeError:
                pass

    def update(self, values: Sequence[Value | list[Value]]) -> None:
        self.queue = tuple(flatten(values))

    def get(self) -> Sequence[Value]:
        try:
            return self.queue
        except AttributeError:
            raise EmptyChannelError()

    def checkpoint(self) -> str:
        try:
            return json.dumps(self.queue)
        except AttributeError:
            raise EmptyChannelError()


class UniqueInbox(Generic[Value], Channel[FrozenSet[Value], Value | list[Value]]):
    """Stores all unique values received, resets in each step."""

    def __init__(self, typ: Type[Value]) -> None:
        self.typ = typ

    @property
    def ValueType(self) -> Type[FrozenSet[Value]]:
        """The type of the value stored in the channel."""
        return FrozenSet[self.typ]  # type: ignore[name-defined]

    @property
    def UpdateType(self) -> Any:
        """The type of the update received by the channel."""
        return Union[self.typ, Sequence[self.typ]]  # type: ignore[name-defined]

    @contextmanager
    def empty(self, checkpoint: Optional[str] = None) -> Generator[Self, None, None]:
        empty = self.__class__(self.typ)
        if checkpoint is not None:
            empty.queue = frozenset(json.loads(checkpoint))
        try:
            yield empty
        finally:
            try:
                del empty.queue
            except AttributeError:
                pass

    def update(self, values: Sequence[Value | list[Value]]) -> None:
        self.queue = frozenset(flatten(values))

    def get(self) -> FrozenSet[Value]:
        try:
            return self.queue
        except AttributeError:
            raise EmptyChannelError()

    def checkpoint(self) -> str:
        try:
            return json.dumps(self.queue)
        except AttributeError:
            raise EmptyChannelError()
