import asyncio
import concurrent.futures
from typing import Any, NamedTuple, Optional, Protocol, Sequence, TypedDict, Union

from langchain_core.runnables import RunnableConfig


class Topics(NamedTuple):
    orchestrator: str
    executor: str
    error: str


class MessageToOrchestrator(TypedDict):
    input: Optional[dict[str, Any]]
    config: RunnableConfig
    finally_executor: Optional[Sequence["MessageToExecutor"]]


class ExecutorTask(TypedDict):
    id: str
    path: tuple[str, ...]


class MessageToExecutor(TypedDict):
    config: RunnableConfig
    task: ExecutorTask
    finally_executor: Optional[Sequence["MessageToExecutor"]]


class ErrorMessage(TypedDict):
    topic: str
    error: str
    msg: Union[MessageToExecutor, MessageToOrchestrator]


class Consumer(Protocol):
    def getmany(
        self, timeout_ms: int, max_records: int
    ) -> dict[str, Sequence[dict[str, Any]]]: ...

    def commit(self) -> None: ...


class AsyncConsumer(Protocol):
    async def getmany(
        self, timeout_ms: int, max_records: int
    ) -> dict[str, Sequence[dict[str, Any]]]: ...

    async def commit(self) -> None: ...


class Producer(Protocol):
    def send(
        self,
        topic: str,
        *,
        key: Optional[bytes] = None,
        value: Optional[bytes] = None,
    ) -> concurrent.futures.Future: ...


class AsyncProducer(Protocol):
    async def send(
        self,
        topic: str,
        *,
        key: Optional[bytes] = None,
        value: Optional[bytes] = None,
    ) -> asyncio.Future: ...
