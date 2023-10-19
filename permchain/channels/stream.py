import json
from contextlib import contextmanager
from typing import Any, FrozenSet, Generator, Generic, Optional, Sequence, Type

from typing_extensions import Self

from permchain.channels.base import Channel, EmptyChannelError, Value


class Set(Generic[Value], Channel[FrozenSet[Value], Value]):
    """Stores all unique values received."""

    def __init__(self, typ: Type[Value]) -> None:
        self.typ = typ
        self.set = set[Value]()

    @property
    def ValueType(self) -> Type[FrozenSet[Value]]:
        """The type of the value stored in the channel."""
        return FrozenSet[self.typ]  # type: ignore[name-defined]

    @property
    def UpdateType(self) -> Type[Value]:
        """The type of the update received by the channel."""
        return self.typ

    @contextmanager
    def empty(self, checkpoint: Optional[str] = None) -> Generator[Self, None, None]:
        empty = self.__class__(self.typ)
        if checkpoint is not None:
            empty.set = set(json.loads(checkpoint))
        try:
            yield empty
        finally:
            pass

    def update(self, values: Sequence[Value]) -> None:
        self.set.update(values)

    def get(self) -> FrozenSet[Value]:
        try:
            return frozenset(self.set)
        except AttributeError:
            raise EmptyChannelError()

    def checkpoint(self) -> str:
        return json.dumps(list(self.set))


class Stream(Generic[Value], Channel[Sequence[Value], Value]):
    """Stores all unique values received."""

    def __init__(self, typ: Type[Value]) -> None:
        self.typ = typ
        self.set = list[Value]()

    @property
    def ValueType(self) -> Any:
        """The type of the value stored in the channel."""
        return Sequence[self.typ]  # type: ignore[name-defined]

    @property
    def UpdateType(self) -> Type[Value]:
        """The type of the update received by the channel."""
        return self.typ

    @contextmanager
    def empty(self, checkpoint: Optional[str] = None) -> Generator[Self, None, None]:
        empty = self.__class__(self.typ)
        if checkpoint is not None:
            empty.set = json.loads(checkpoint)
        try:
            yield empty
        finally:
            pass

    def update(self, values: Sequence[Value]) -> None:
        self.set.extend(values)

    def get(self) -> Sequence[Value]:
        try:
            return tuple(self.set)
        except AttributeError:
            raise EmptyChannelError()

    def checkpoint(self) -> str:
        return json.dumps(self.set)
