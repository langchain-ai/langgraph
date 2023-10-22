import json
from contextlib import contextmanager
from typing import Any, Generator, Generic, Optional, Sequence, Type, Union, cast

from typing_extensions import Self

from permchain.channels.base import (
    Channel,
    EmptyChannelError,
    Value,
)


class Inbox(Generic[Value], Channel[Sequence[Value], Value | Sequence[Value]]):
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

    def update(self, values: Sequence[Value | Sequence[Value]]) -> None:
        self.queue = tuple(
            cast(Value, v)
            for value in values
            for v in (
                (value,)
                if isinstance(value, self.typ)
                else cast(Sequence[Value], value)
            )
        )

    def get(self) -> Sequence[Value]:
        try:
            return self.queue
        except AttributeError:
            raise EmptyChannelError()

    def checkpoint(self) -> str:
        return json.dumps(self.queue)


class UniqueInbox(Generic[Value], Channel[Sequence[Value], Value | Sequence[Value]]):
    """Stores all unique values received, resets in each step."""

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

    def update(self, values: Sequence[Value | Sequence[Value]]) -> None:
        self.queue = tuple(
            set(
                cast(Value, v)
                for value in values
                for v in (
                    (value,)
                    if isinstance(value, self.typ)
                    else cast(Sequence[Value], value)
                )
            )
        )

    def get(self) -> Sequence[Value]:
        try:
            return self.queue
        except AttributeError:
            raise EmptyChannelError()

    def checkpoint(self) -> str:
        return json.dumps(self.queue)
