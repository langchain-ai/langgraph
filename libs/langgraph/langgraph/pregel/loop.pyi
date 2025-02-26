from _typeshed import Incomplete
from langchain_core.callbacks import AsyncParentRunManager as AsyncParentRunManager, ParentRunManager as ParentRunManager
from langchain_core.runnables import RunnableConfig as RunnableConfig
from langgraph.channels.base import BaseChannel as BaseChannel
from langgraph.checkpoint.base import BaseCheckpointSaver, ChannelVersions as ChannelVersions, Checkpoint as Checkpoint, CheckpointMetadata as CheckpointMetadata, PendingWrite as PendingWrite
from langgraph.managed.base import ManagedValueMapping as ManagedValueMapping, ManagedValueSpec as ManagedValueSpec
from langgraph.pregel.algo import Call as Call, GetNextVersion as GetNextVersion
from langgraph.pregel.executor import Submit as Submit
from langgraph.pregel.read import PregelNode as PregelNode
from langgraph.store.base import BaseStore as BaseStore
from langgraph.types import All as All, LoopProtocol, PregelExecutableTask, StreamChunk as StreamChunk, StreamProtocol
from types import TracebackType as TracebackType
from typing import Any, AsyncContextManager, Callable, ContextManager, Literal, Mapping, Sequence, TypeVar
from typing_extensions import ParamSpec, Self

V = TypeVar('V')
P = ParamSpec('P')
INPUT_DONE: Incomplete
INPUT_RESUMING: Incomplete
SPECIAL_CHANNELS: Incomplete

def DuplexStream(*streams: StreamProtocol) -> StreamProtocol: ...

class PregelLoop(LoopProtocol):
    input: Any | None
    checkpointer: BaseCheckpointSaver | None
    nodes: Mapping[str, PregelNode]
    specs: Mapping[str, BaseChannel | ManagedValueSpec]
    output_keys: str | Sequence[str]
    stream_keys: str | Sequence[str]
    skip_done_tasks: bool
    is_nested: bool
    manager: None | AsyncParentRunManager | ParentRunManager
    interrupt_after: All | Sequence[str]
    interrupt_before: All | Sequence[str]
    checkpointer_get_next_version: GetNextVersion
    checkpointer_put_writes: Callable[[RunnableConfig, Sequence[tuple[str, Any]], str], Any] | None
    checkpointer_put_writes_accepts_task_path: bool
    submit: Submit
    channels: Mapping[str, BaseChannel]
    managed: ManagedValueMapping
    checkpoint: Checkpoint
    checkpoint_ns: tuple[str, ...]
    checkpoint_config: RunnableConfig
    checkpoint_metadata: CheckpointMetadata
    checkpoint_pending_writes: list[PendingWrite]
    checkpoint_previous_versions: dict[str, str | float | int]
    prev_checkpoint_config: RunnableConfig | None
    status: Literal['pending', 'done', 'interrupt_before', 'interrupt_after', 'out_of_steps']
    tasks: dict[str, PregelExecutableTask]
    to_interrupt: list[PregelExecutableTask]
    output: None | dict[str, Any] | Any
    debug: Incomplete
    stream: Incomplete
    config: Incomplete
    def __init__(self, input: Any | None, *, stream: StreamProtocol | None, config: RunnableConfig, store: BaseStore | None, checkpointer: BaseCheckpointSaver | None, nodes: Mapping[str, PregelNode], specs: Mapping[str, BaseChannel | ManagedValueSpec], output_keys: str | Sequence[str], stream_keys: str | Sequence[str], interrupt_after: All | Sequence[str] = ..., interrupt_before: All | Sequence[str] = ..., manager: None | AsyncParentRunManager | ParentRunManager = None, debug: bool = False) -> None: ...
    def put_writes(self, task_id: str, writes: Sequence[tuple[str, Any]]) -> None: ...
    def accept_push(self, task: PregelExecutableTask, write_idx: int, call: Call | None = None) -> PregelExecutableTask | None: ...
    def tick(self, *, input_keys: str | Sequence[str]) -> bool: ...

class SyncPregelLoop(PregelLoop, ContextManager):
    stack: Incomplete
    checkpointer_get_next_version: Incomplete
    checkpointer_put_writes: Incomplete
    checkpointer_put_writes_accepts_task_path: Incomplete
    def __init__(self, input: Any | None, *, stream: StreamProtocol | None, config: RunnableConfig, store: BaseStore | None, checkpointer: BaseCheckpointSaver | None, nodes: Mapping[str, PregelNode], specs: Mapping[str, BaseChannel | ManagedValueSpec], manager: None | AsyncParentRunManager | ParentRunManager = None, interrupt_after: All | Sequence[str] = ..., interrupt_before: All | Sequence[str] = ..., output_keys: str | Sequence[str] = ..., stream_keys: str | Sequence[str] = ..., debug: bool = False) -> None: ...
    checkpoint_config: Incomplete
    prev_checkpoint_config: Incomplete
    checkpoint: Incomplete
    checkpoint_metadata: Incomplete
    checkpoint_pending_writes: Incomplete
    submit: Incomplete
    status: str
    step: Incomplete
    stop: Incomplete
    checkpoint_previous_versions: Incomplete
    def __enter__(self) -> Self: ...
    def __exit__(self, exc_type: type[BaseException] | None, exc_value: BaseException | None, traceback: TracebackType | None) -> bool | None: ...

class AsyncPregelLoop(PregelLoop, AsyncContextManager):
    stack: Incomplete
    checkpointer_get_next_version: Incomplete
    checkpointer_put_writes: Incomplete
    checkpointer_put_writes_accepts_task_path: Incomplete
    def __init__(self, input: Any | None, *, stream: StreamProtocol | None, config: RunnableConfig, store: BaseStore | None, checkpointer: BaseCheckpointSaver | None, nodes: Mapping[str, PregelNode], specs: Mapping[str, BaseChannel | ManagedValueSpec], interrupt_after: All | Sequence[str] = ..., interrupt_before: All | Sequence[str] = ..., manager: None | AsyncParentRunManager | ParentRunManager = None, output_keys: str | Sequence[str] = ..., stream_keys: str | Sequence[str] = ..., debug: bool = False) -> None: ...
    checkpoint_config: Incomplete
    prev_checkpoint_config: Incomplete
    checkpoint: Incomplete
    checkpoint_metadata: Incomplete
    checkpoint_pending_writes: Incomplete
    submit: Incomplete
    status: str
    step: Incomplete
    stop: Incomplete
    checkpoint_previous_versions: Incomplete
    async def __aenter__(self) -> Self: ...
    async def __aexit__(self, exc_type: type[BaseException] | None, exc_value: BaseException | None, traceback: TracebackType | None) -> bool | None: ...
