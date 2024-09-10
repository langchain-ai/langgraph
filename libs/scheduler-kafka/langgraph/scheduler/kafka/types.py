from typing import Any, NamedTuple, Optional, Sequence, TypedDict, Union

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
