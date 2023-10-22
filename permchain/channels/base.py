from abc import ABC, abstractmethod
from contextlib import asynccontextmanager, contextmanager
from typing import (
    Any,
    AsyncGenerator,
    Generator,
    Generic,
    Optional,
    Sequence,
    TypeVar,
)

from typing_extensions import Self

Value = TypeVar("Value")
Update = TypeVar("Update")


class EmptyChannelError(Exception):
    pass


class InvalidUpdateError(Exception):
    pass


class Channel(Generic[Value, Update], ABC):
    @property
    @abstractmethod
    def ValueType(self) -> Any:
        """The type of the value stored in the channel."""

    @property
    @abstractmethod
    def UpdateType(self) -> Any:
        """The type of the update received by the channel."""

    @contextmanager
    @abstractmethod
    def empty(self, checkpoint: Optional[str] = None) -> Generator[Self, None, None]:
        """Return a new identical channel, optionally initialized from a checkpoint."""

    @asynccontextmanager
    async def aempty(
        self, checkpoint: Optional[str] = None
    ) -> AsyncGenerator[Self, None]:
        """Return a new identical channel, optionally initialized from a checkpoint."""
        with self.empty(checkpoint) as value:
            yield value

    @abstractmethod
    def update(self, values: Sequence[Update]) -> None:
        ...

    @abstractmethod
    def get(self) -> Value:
        ...

    @abstractmethod
    def checkpoint(self) -> str | None:
        ...
