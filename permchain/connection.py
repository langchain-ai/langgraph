import asyncio
from abc import ABC, abstractmethod
from typing import Any, Callable, Iterator, TypedDict

PubSubListener = Callable[[Any], None]


class LogMessage(TypedDict):
    topic: str
    value: Any
    published_at: str
    correlation_id: str


class PubSubConnection(ABC):
    def full_topic_name(self, prefix: str, *parts: str) -> str:
        """Return the full topic name for a given prefix and topic name."""
        return ":".join(map(str, [prefix, *parts]))

    @abstractmethod
    def iterate(self, prefix: str, topic_name: str, wait: bool) -> Iterator[Any]:
        """Iterate over all currently queued messages for a topic, consuming them."""
        ...

    # TODO add aiterate() method

    @abstractmethod
    def listen(
        self, prefix: str, topic_name: str, listeners: list[PubSubListener]
    ) -> None:
        ...

    async def alisten(
        self, prefix: str, topic_name: str, listeners: list[PubSubListener]
    ) -> None:
        return await asyncio.get_event_loop().run_in_executor(
            None, self.listen, prefix, topic_name, listeners
        )

    @abstractmethod
    def send(self, prefix: str, topic_name: str, message: Any) -> None:
        ...

    async def asend(self, prefix: str, topic_name: str, message: Any) -> None:
        return await asyncio.get_event_loop().run_in_executor(
            None, self.send, prefix, topic_name, message
        )

    @abstractmethod
    def disconnect(self, prefix: str) -> None:
        ...

    async def adisconnect(self, prefix: str) -> None:
        return await asyncio.get_event_loop().run_in_executor(
            None, self.disconnect, prefix
        )

    @abstractmethod
    def peek(self, prefix: str) -> Iterator[LogMessage]:
        """Iterate over all previously published messages for all topics,
        without consuming them."""
        ...
