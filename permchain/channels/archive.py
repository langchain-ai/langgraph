import json
from contextlib import contextmanager
from typing import Any, FrozenSet, Generator, Generic, Optional, Sequence, Type

from typing_extensions import Self

from permchain.channels.base import BaseChannel, EmptyChannelError, Value
from permchain.channels.inbox import flatten


class Archive(Generic[Value], BaseChannel[Sequence[Value], Value | list[Value]]):
    """Stores all unique values received, persists across steps."""

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

    def update(self, values: Sequence[Value | list[Value]]) -> None:
        self.set.extend(flatten(values))

    def get(self) -> Sequence[Value]:
        try:
            return tuple(self.set)
        except AttributeError:
            raise EmptyChannelError()

    def checkpoint(self) -> str:
        try:
            return json.dumps(self.set)
        except AttributeError:
            raise EmptyChannelError()


class UniqueArchive(Generic[Value], BaseChannel[FrozenSet[Value], Value]):
    """Stores all unique values received, persists across steps."""

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

    def update(self, values: Sequence[Value | list[Value]]) -> None:
        self.set.update(flatten(values))

    def get(self) -> FrozenSet[Value]:
        try:
            return frozenset(self.set)
        except AttributeError:
            raise EmptyChannelError()

    def checkpoint(self) -> str:
        try:
            return json.dumps(list(self.set))
        except AttributeError:
            raise EmptyChannelError()
