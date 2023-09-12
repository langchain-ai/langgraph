import asyncio
from abc import ABC, abstractmethod
from typing import Any, Callable, Iterator

PubSubListener = Callable[[Any], None]


class PubSubConnection(ABC):
    @abstractmethod
    def iterate(self, topic_name: str) -> Iterator[Any]:
        ...

    @abstractmethod
    def listen(self, topic_name: str, listener: PubSubListener) -> None:
        ...

    async def alisten(self, topic_name: str, listener: PubSubListener) -> None:
        return await asyncio.get_event_loop().run_in_executor(
            None, self.listen, topic_name, listener
        )

    @abstractmethod
    def send(self, topic_name: str, message: Any) -> None:
        ...

    async def asend(self, topic_name: str, message: Any) -> None:
        return await asyncio.get_event_loop().run_in_executor(
            None, self.send, topic_name, message
        )

    @abstractmethod
    def disconnect(self, topic_name: str) -> None:
        ...

    async def adisconnect(self, topic_name: str) -> None:
        return await asyncio.get_event_loop().run_in_executor(
            None, self.disconnect, topic_name
        )
