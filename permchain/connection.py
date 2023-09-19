import asyncio
from abc import ABC, abstractmethod
from typing import Any, Callable, Iterator, TypedDict

PubSubListener = Callable[[Any], None]


class PubSubLog(TypedDict):
    topic: str
    value: Any
    published_at: str
    correlation_id: str


class PubSubConnection(ABC):
    def full_name(self, prefix: str, *parts: str) -> str:
        """Return the full topic name for a given prefix and topic name."""
        return ":".join(map(str, [prefix, *parts]))

    @abstractmethod
    def observe(self, prefix: str) -> Iterator[PubSubLog]:
        """Iterate over messages for all topics under this prefix,
        without affecting listeners/iterators on each topic.
        This method waits for new messages to arrive."""
        ...

    @abstractmethod
    def iterate(self, prefix: str, topic: str, *, wait: bool) -> Iterator[Any]:
        """Iterate over all currently queued messages for a topic, consuming them.
        Optionally wait for new messages to arrive."""
        ...

    # TODO add aiterate() method

    @abstractmethod
    def listen(self, prefix: str, topic: str, listeners: list[PubSubListener]) -> None:
        ...

    async def alisten(
        self, prefix: str, topic: str, listeners: list[PubSubListener]
    ) -> None:
        return await asyncio.get_event_loop().run_in_executor(
            None, self.listen, prefix, topic, listeners
        )

    @abstractmethod
    def send(self, prefix: str, topic: str, message: Any) -> None:
        ...

    async def asend(self, prefix: str, topic: str, message: Any) -> None:
        return await asyncio.get_event_loop().run_in_executor(
            None, self.send, prefix, topic, message
        )

    @abstractmethod
    def disconnect(self, name: str) -> None:
        ...

    async def adisconnect(self, name: str) -> None:
        return await asyncio.get_event_loop().run_in_executor(
            None, self.disconnect, name
        )
