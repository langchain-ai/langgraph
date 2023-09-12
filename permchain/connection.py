import asyncio
from abc import ABC, abstractmethod
from typing import Any, Callable, Iterator, TypedDict

PubSubListener = Callable[[Any], None]


class LogMessage(TypedDict):
    message: Any
    topic_name: str
    started_at: str


class PubSubConnection(ABC):
    def full_topic_name(self, prefix: str, topic_name: str) -> str:
        return f"{prefix}:{topic_name}"

    @abstractmethod
    def iterate(self, prefix: str, topic_name: str) -> Iterator[Any]:
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
        ...
