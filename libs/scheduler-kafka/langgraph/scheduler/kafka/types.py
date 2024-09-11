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


class TopicPartition(Protocol):
    topic: str
    partition: int


class ConsumerRecord(Protocol):
    topic: str
    "The topic this record is received from"
    partition: int
    "The partition from which this record is received"
    offset: int
    "The position of this record in the corresponding Kafka partition."
    timestamp: int
    "The timestamp of this record"
    timestamp_type: int
    "The timestamp type of this record"
    key: Optional[bytes]
    "The key (or `None` if no key is specified)"
    value: Optional[bytes]
    "The value"


class Consumer(Protocol):
    def getmany(
        self, timeout_ms: int, max_records: int
    ) -> dict[TopicPartition, Sequence[ConsumerRecord]]: ...

    def commit(self) -> None: ...


class AsyncConsumer(Protocol):
    async def getmany(
        self, timeout_ms: int, max_records: int
    ) -> dict[TopicPartition, Sequence[ConsumerRecord]]: ...

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
