from typing import Any, NamedTuple, Optional, TypedDict, Union

from langchain_core.runnables import RunnableConfig


class Topics(NamedTuple):
    orchestrator: str
    executor: str
    error: str


class MessageToOrchestrator(TypedDict):
    input: Optional[dict[str, Any]]
    config: RunnableConfig


class ExecutorTask(TypedDict):
    id: str
    path: tuple[str, ...]
    step: int
    resuming: bool


class MessageToExecutor(TypedDict):
    config: RunnableConfig
    task: ExecutorTask


class ErrorMessage(TypedDict):
    topic: str
    error: str
    msg: Union[MessageToExecutor, MessageToOrchestrator]
