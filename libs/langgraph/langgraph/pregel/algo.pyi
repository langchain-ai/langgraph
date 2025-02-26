from _typeshed import Incomplete
from langchain_core.callbacks import Callbacks as Callbacks
from langchain_core.callbacks.manager import AsyncParentRunManager as AsyncParentRunManager, ParentRunManager as ParentRunManager
from langchain_core.runnables.config import RunnableConfig as RunnableConfig
from langgraph.channels.base import BaseChannel
from langgraph.checkpoint.base import BaseCheckpointSaver as BaseCheckpointSaver, Checkpoint as Checkpoint, PendingWrite as PendingWrite, V
from langgraph.managed.base import ManagedValueMapping as ManagedValueMapping
from langgraph.pregel.read import PregelNode as PregelNode
from langgraph.store.base import BaseStore as BaseStore
from langgraph.types import All as All, PregelExecutableTask, PregelTask, RetryPolicy as RetryPolicy
from typing import Any, Callable, Iterable, Literal, Mapping, NamedTuple, Protocol, Sequence, overload

GetNextVersion = Callable[[V | None, BaseChannel], V]
SUPPORTS_EXC_NOTES: Incomplete

class WritesProtocol(Protocol):
    @property
    def path(self) -> tuple[str | int | tuple, ...]: ...
    @property
    def name(self) -> str: ...
    @property
    def writes(self) -> Sequence[tuple[str, Any]]: ...
    @property
    def triggers(self) -> Sequence[str]: ...

class PregelTaskWrites(NamedTuple):
    path: tuple[str | int | tuple, ...]
    name: str
    writes: Sequence[tuple[str, Any]]
    triggers: Sequence[str]

class Call:
    func: Callable
    input: Any
    retry: RetryPolicy | None
    callbacks: Callbacks
    def __init__(self, func: Callable, input: Any, *, retry: RetryPolicy | None, callbacks: Callbacks) -> None: ...

def should_interrupt(checkpoint: Checkpoint, interrupt_nodes: All | Sequence[str], tasks: Iterable[PregelExecutableTask]) -> list[PregelExecutableTask]: ...
def local_read(step: int, checkpoint: Checkpoint, channels: Mapping[str, BaseChannel], managed: ManagedValueMapping, task: WritesProtocol, config: RunnableConfig, select: list[str] | str, fresh: bool = False) -> dict[str, Any] | Any: ...
def local_write(commit: Callable[[Sequence[tuple[str, Any]]], None], process_keys: Iterable[str], writes: Sequence[tuple[str, Any]]) -> None: ...
def increment(current: int | None, channel: BaseChannel) -> int: ...
def apply_writes(checkpoint: Checkpoint, channels: Mapping[str, BaseChannel], tasks: Iterable[WritesProtocol], get_next_version: GetNextVersion | None) -> dict[str, list[Any]]: ...
@overload
def prepare_next_tasks(checkpoint: Checkpoint, pending_writes: list[PendingWrite], processes: Mapping[str, PregelNode], channels: Mapping[str, BaseChannel], managed: ManagedValueMapping, config: RunnableConfig, step: int, *, for_execution: Literal[False], store: Literal[None] = None, checkpointer: Literal[None] = None, manager: Literal[None] = None) -> dict[str, PregelTask]: ...
@overload
def prepare_next_tasks(checkpoint: Checkpoint, pending_writes: list[PendingWrite], processes: Mapping[str, PregelNode], channels: Mapping[str, BaseChannel], managed: ManagedValueMapping, config: RunnableConfig, step: int, *, for_execution: Literal[True], store: BaseStore | None, checkpointer: BaseCheckpointSaver | None, manager: None | ParentRunManager | AsyncParentRunManager) -> dict[str, PregelExecutableTask]: ...
def prepare_single_task(task_path: tuple[Any, ...], task_id_checksum: str | None, *, checkpoint: Checkpoint, pending_writes: list[PendingWrite], processes: Mapping[str, PregelNode], channels: Mapping[str, BaseChannel], managed: ManagedValueMapping, config: RunnableConfig, step: int, for_execution: bool, store: BaseStore | None = None, checkpointer: BaseCheckpointSaver | None = None, manager: None | ParentRunManager | AsyncParentRunManager = None) -> None | PregelTask | PregelExecutableTask: ...
def task_path_str(tup: str | int | tuple) -> str: ...
