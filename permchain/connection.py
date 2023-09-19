import asyncio
from abc import ABC, abstractmethod
from typing import Any, Callable, Iterator, TypedDict


class PubSubMessage(TypedDict):
    value: Any
    topic: str
    namespace: str
    published_at: str
    correlation_ids: list[str]


PubSubListener = Callable[[PubSubMessage], None]


class PubSubConnection(ABC):
    def full_name(self, prefix: str, *parts: str) -> str:
        """Return the full topic name for a given prefix and topic name."""
        return ":".join(map(str, [prefix, *parts]))

    @abstractmethod
    def observe(self, prefix: str) -> Iterator[PubSubMessage]:
        """Iterate over messages for all topics under this prefix,
        without affecting listeners/iterators on each topic.
        This method waits for new messages to arrive."""
        ...

    @abstractmethod
    def iterate(
        self, prefix: str, topic: str, *, wait: bool
    ) -> Iterator[PubSubMessage]:
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
    def send(
        self, prefix: str, topic: str, value: Any, correlation_ids: list[str]
    ) -> None:
        ...

    async def asend(
        self, prefix: str, topic: str, value: Any, correlation_ids: list[str]
    ) -> None:
        return await asyncio.get_event_loop().run_in_executor(
            None, self.send, correlation_ids, prefix, topic, value
        )

    @abstractmethod
    def disconnect(self, name: str) -> None:
        ...

    async def adisconnect(self, name: str) -> None:
        return await asyncio.get_event_loop().run_in_executor(
            None, self.disconnect, name
        )
