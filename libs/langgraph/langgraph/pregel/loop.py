import asyncio
import concurrent.futures
from collections import deque
from contextlib import AsyncExitStack, ExitStack
from types import TracebackType
from typing import (
    Any,
    AsyncContextManager,
    Callable,
    ContextManager,
    List,
    Literal,
    Mapping,
    Optional,
    Sequence,
    Tuple,
    Type,
    TypeVar,
    Union,
    cast,
)

from langchain_core.callbacks import AsyncParentRunManager, ParentRunManager
from langchain_core.runnables import RunnableConfig
from typing_extensions import Self

from langgraph.channels.base import BaseChannel
from langgraph.channels.untracked_value import UntrackedValue
from langgraph.checkpoint.base import (
    BaseCheckpointSaver,
    Checkpoint,
    CheckpointMetadata,
    CheckpointTuple,
    PendingWrite,
    copy_checkpoint,
    create_checkpoint,
    empty_checkpoint,
)
from langgraph.constants import (
    CONFIG_KEY_READ,
    CONFIG_KEY_RESUMING,
    ERROR,
    INPUT,
    INTERRUPT,
)
from langgraph.errors import EmptyInputError, GraphInterrupt
from langgraph.managed.base import (
    ManagedValueMapping,
    ManagedValueSpec,
    WritableManagedValue,
)
from langgraph.pregel.algo import (
    PregelTaskWrites,
    apply_writes,
    increment,
    prepare_next_tasks,
    should_interrupt,
)
from langgraph.pregel.debug import (
    map_debug_checkpoint,
    map_debug_task_results,
    map_debug_tasks,
)
from langgraph.pregel.executor import (
    AsyncBackgroundExecutor,
    BackgroundExecutor,
    Submit,
)
from langgraph.pregel.io import (
    map_input,
    map_output_updates,
    map_output_values,
    read_channels,
    single,
)
from langgraph.pregel.manager import AsyncChannelsManager, ChannelsManager
from langgraph.pregel.read import PregelNode
from langgraph.pregel.types import PregelExecutableTask
from langgraph.pregel.utils import get_new_channel_versions
from langgraph.store.base import BaseStore
from langgraph.store.batch import AsyncBatchedStore

V = TypeVar("V")
INPUT_DONE = object()
INPUT_RESUMING = object()
EMPTY_SEQ = ()


class PregelLoop:
    input: Optional[Any]
    config: RunnableConfig
    store: Optional[BaseStore]
    checkpointer: Optional[BaseCheckpointSaver]
    nodes: Mapping[str, PregelNode]
    specs: Mapping[str, Union[BaseChannel, ManagedValueSpec]]
    output_keys: Union[str, Sequence[str]]
    stream_keys: Union[str, Sequence[str]]
    is_nested: bool

    checkpointer_get_next_version: Callable[[Optional[V]], V]
    checkpointer_put_writes: Optional[
        Callable[[RunnableConfig, Sequence[tuple[str, Any]], str], Any]
    ]
    _checkpointer_put_after_previous: Optional[
        Callable[
            [
                Optional[concurrent.futures.Future],
                RunnableConfig,
                Sequence[tuple[str, Any]],
                str,
            ],
            Any,
        ]
    ]
    submit: Submit
    channels: Mapping[str, BaseChannel]
    managed: ManagedValueMapping
    checkpoint: Checkpoint
    checkpoint_config: RunnableConfig
    checkpoint_metadata: CheckpointMetadata
    checkpoint_pending_writes: List[PendingWrite]
    checkpoint_previous_versions: dict[str, Union[str, float, int]]

    step: int
    stop: int
    status: Literal[
        "pending", "done", "interrupt_before", "interrupt_after", "out_of_steps"
    ]
    tasks: Sequence[PregelExecutableTask]
    stream: deque[Tuple[str, Any]]
    output: Union[None, dict[str, Any], Any] = None

    # public

    def __init__(
        self,
        input: Optional[Any],
        *,
        config: RunnableConfig,
        store: Optional[BaseStore],
        checkpointer: Optional[BaseCheckpointSaver],
        nodes: Mapping[str, PregelNode],
        specs: Mapping[str, Union[BaseChannel, ManagedValueSpec]],
        output_keys: Union[str, Sequence[str]],
        stream_keys: Union[str, Sequence[str]],
    ) -> None:
        self.stream = deque()
        self.input = input
        self.config = config
        self.store = store
        self.checkpointer = checkpointer
        self.nodes = nodes
        self.specs = specs
        self.output_keys = output_keys
        self.stream_keys = stream_keys
        self.is_nested = CONFIG_KEY_READ in self.config.get("configurable", {})

    def put_writes(self, task_id: str, writes: Sequence[tuple[str, Any]]) -> None:
        """Put writes for a task, to be read by the next tick."""
        if not writes:
            return
        # TODO in distributed execution, these UntrackedValue writes would
        # effectively be dropped, so need rethink how to handle them then
        if saveable_writes := [
            w
            for w in writes
            if w[0] not in self.channels
            or not isinstance(self.channels[w[0]], UntrackedValue)
        ]:
            self.checkpoint_pending_writes.extend(
                (task_id, k, v) for k, v in saveable_writes
            )
            if self.checkpointer_put_writes is not None:
                self.submit(
                    self.checkpointer_put_writes,
                    {
                        **self.checkpoint_config,
                        "configurable": {
                            **self.checkpoint_config["configurable"],
                            "checkpoint_ns": self.config["configurable"].get(
                                "checkpoint_ns", ""
                            ),
                            "checkpoint_id": self.checkpoint["id"],
                        },
                    },
                    saveable_writes,
                    task_id,
                )
        if task := next((t for t in self.tasks if t.id == task_id), None):
            self.stream.extend(
                ("updates", v)
                for v in map_output_updates(self.output_keys, [(task, writes)])
            )
            self.stream.extend(
                ("debug", v)
                for v in map_debug_task_results(
                    self.step, [(task, writes)], self.stream_keys
                )
            )

    def tick(
        self,
        *,
        input_keys: Union[str, Sequence[str]],
        interrupt_after: Sequence[str] = EMPTY_SEQ,
        interrupt_before: Sequence[str] = EMPTY_SEQ,
        manager: Union[None, AsyncParentRunManager, ParentRunManager] = None,
    ) -> bool:
        """Execute a single iteration of the Pregel loop.
        Returns True if more iterations are needed."""

        if self.status != "pending":
            raise RuntimeError("Cannot tick when status is no longer 'pending'")

        if self.input not in (INPUT_DONE, INPUT_RESUMING):
            self._first(input_keys=input_keys)
        elif all(task.writes for task in self.tasks):
            writes = [w for t in self.tasks for w in t.writes]
            # all tasks have finished
            mv_writes = apply_writes(
                self.checkpoint,
                self.channels,
                self.tasks,
                self.checkpointer_get_next_version,
            )
            # apply writes to managed values
            for key, values in mv_writes.items():
                self._update_mv(key, values)
            # produce values output
            self.stream.extend(
                ("values", v)
                for v in map_output_values(self.output_keys, writes, self.channels)
            )
            # clear pending writes
            self.checkpoint_pending_writes.clear()
            # save checkpoint
            self._put_checkpoint(
                {
                    "source": "loop",
                    "writes": single(
                        map_output_updates(
                            self.output_keys, [(t, t.writes) for t in self.tasks]
                        )
                    ),
                }
            )
            # after execution, check if we should interrupt
            if should_interrupt(self.checkpoint, interrupt_after, self.tasks):
                self.status = "interrupt_after"
                if self.is_nested:
                    raise GraphInterrupt()
                else:
                    return False
        else:
            return False

        # check if iteration limit is reached
        if self.step > self.stop:
            self.status = "out_of_steps"
            return False

        # prepare next tasks
        self.tasks = prepare_next_tasks(
            self.checkpoint,
            self.nodes,
            self.channels,
            self.managed,
            self.config,
            self.step,
            for_execution=True,
            manager=manager,
            checkpointer=self.checkpointer,
            is_resuming=self.input is INPUT_RESUMING,
        )

        # produce debug output
        if self._checkpointer_put_after_previous is not None:
            self.stream.extend(
                ("debug", v)
                for v in map_debug_checkpoint(
                    self.step - 1,  # printing checkpoint for previous step
                    self.checkpoint_config,
                    self.channels,
                    self.stream_keys,
                    self.checkpoint_metadata,
                    self.checkpoint,
                    self.tasks,
                    self.checkpoint_pending_writes,
                )
            )

        # if no more tasks, we're done
        if not self.tasks:
            self.status = "done"
            return False

        # if there are pending writes from a previous loop, apply them
        if self.checkpoint_pending_writes:
            for tid, k, v in self.checkpoint_pending_writes:
                if k in (ERROR, INTERRUPT):
                    continue
                if task := next((t for t in self.tasks if t.id == tid), None):
                    task.writes.append((k, v))

        # if all tasks have finished, re-tick
        if all(task.writes for task in self.tasks):
            return self.tick(
                input_keys=input_keys,
                interrupt_after=interrupt_after,
                interrupt_before=interrupt_before,
                manager=manager,
            )

        # before execution, check if we should interrupt
        if should_interrupt(self.checkpoint, interrupt_before, self.tasks):
            self.status = "interrupt_before"
            if self.is_nested:
                raise GraphInterrupt()
            else:
                return False

        # produce debug output
        self.stream.extend(("debug", v) for v in map_debug_tasks(self.step, self.tasks))

        return True

    # private

    def _first(self, *, input_keys: Union[str, Sequence[str]]) -> None:
        # resuming from previous checkpoint requires
        # - finding a previous checkpoint
        # - receiving None input (outer graph) or RESUMING flag (subgraph)
        is_resuming = bool(self.checkpoint["channel_versions"]) and bool(
            self.config.get("configurable", {}).get(CONFIG_KEY_RESUMING)
            or self.input is None
        )

        # proceed past previous checkpoint
        if is_resuming:
            self.checkpoint["versions_seen"].setdefault(INTERRUPT, {})
            for k in self.channels:
                if k in self.checkpoint["channel_versions"]:
                    version = self.checkpoint["channel_versions"][k]
                    self.checkpoint["versions_seen"][INTERRUPT][k] = version
        # map inputs to channel updates
        elif input_writes := deque(map_input(input_keys, self.input)):
            # discard any unfinished tasks from previous checkpoint
            discard_tasks = prepare_next_tasks(
                self.checkpoint,
                self.nodes,
                self.channels,
                self.managed,
                self.config,
                self.step,
                for_execution=True,
                manager=None,
            )
            # apply input writes
            assert not apply_writes(
                self.checkpoint,
                self.channels,
                discard_tasks + [PregelTaskWrites(INPUT, input_writes, [])],
                self.checkpointer_get_next_version,
            ), "Can't write to SharedValues in graph input"
            # save input checkpoint
            self._put_checkpoint({"source": "input", "writes": self.input})
        else:
            raise EmptyInputError(f"Received no input for {input_keys}")
        # done with input
        self.input = INPUT_RESUMING if is_resuming else INPUT_DONE

    def _put_checkpoint(self, metadata: CheckpointMetadata) -> None:
        # assign step
        metadata["step"] = self.step
        # bail if no checkpointer
        if self._checkpointer_put_after_previous is not None:
            # create new checkpoint
            self.checkpoint_metadata = metadata
            self.checkpoint = create_checkpoint(
                self.checkpoint,
                self.channels,
                self.step,
                # child graphs keep at most one checkpoint per parent checkpoint
                # this is achieved by writing child checkpoints as progress is made
                # (so that error recovery / resuming from interrupt don't lose work)
                # but doing so always with an id equal to that of the parent checkpoint
                id=self.config["configurable"]["checkpoint_id"]
                if self.is_nested
                else None,
            )

            self.checkpoint_config = {
                **self.checkpoint_config,
                "configurable": {
                    **self.checkpoint_config["configurable"],
                    "checkpoint_ns": self.config["configurable"].get(
                        "checkpoint_ns", ""
                    ),
                },
            }

            channel_versions = self.checkpoint["channel_versions"].copy()
            new_versions = get_new_channel_versions(
                self.checkpoint_previous_versions, channel_versions
            )

            self.checkpoint_previous_versions = channel_versions

            # save it, without blocking
            # if there's a previous checkpoint save in progress, wait for it
            # ensuring checkpointers receive checkpoints in order
            self._put_checkpoint_fut = self.submit(
                self._checkpointer_put_after_previous,
                getattr(self, "_put_checkpoint_fut", None),
                self.checkpoint_config,
                copy_checkpoint(self.checkpoint),
                self.checkpoint_metadata,
                new_versions,
            )
            self.checkpoint_config = {
                **self.checkpoint_config,
                "configurable": {
                    **self.checkpoint_config["configurable"],
                    "checkpoint_id": self.checkpoint["id"],
                },
            }
        # increment step
        self.step += 1

    def _update_mv(self, key: str, values: Sequence[Any]) -> None:
        raise NotImplementedError

    def _suppress_interrupt(
        self,
        exc_type: Optional[Type[BaseException]],
        exc_value: Optional[BaseException],
        traceback: Optional[TracebackType],
    ) -> Optional[bool]:
        suppress = isinstance(exc_value, GraphInterrupt) and not self.is_nested
        if suppress or exc_type is None:
            # save final output
            self.output = read_channels(self.channels, self.output_keys)
        if suppress:
            # suppress interrupt
            return True


class SyncPregelLoop(PregelLoop, ContextManager):
    def __init__(
        self,
        input: Optional[Any],
        *,
        config: RunnableConfig,
        store: Optional[BaseStore],
        checkpointer: Optional[BaseCheckpointSaver],
        nodes: Mapping[str, PregelNode],
        specs: Mapping[str, Union[BaseChannel, ManagedValueSpec]],
        output_keys: Union[str, Sequence[str]] = EMPTY_SEQ,
        stream_keys: Union[str, Sequence[str]] = EMPTY_SEQ,
    ) -> None:
        super().__init__(
            input,
            config=config,
            checkpointer=checkpointer,
            store=store,
            nodes=nodes,
            specs=specs,
            output_keys=output_keys,
            stream_keys=stream_keys,
        )
        self.stack = ExitStack()
        if checkpointer:
            self.checkpointer_get_next_version = checkpointer.get_next_version
            self.checkpointer_put_writes = checkpointer.put_writes
        else:
            self.checkpointer_get_next_version = increment
            self._checkpointer_put_after_previous = None
            self.checkpointer_put_writes = None

    def _checkpointer_put_after_previous(
        self,
        prev: Optional[concurrent.futures.Future],
        config: RunnableConfig,
        checkpoint: Checkpoint,
        metadata: CheckpointMetadata,
        new_versions: Optional[dict[str, Union[str, float, int]]],
    ) -> RunnableConfig:
        try:
            if prev is not None:
                prev.result()
        finally:
            self.checkpointer.put(config, checkpoint, metadata, new_versions)

    def _update_mv(self, key: str, values: Sequence[Any]) -> None:
        return self.submit(cast(WritableManagedValue, self.managed[key]).update, values)

    # context manager

    def __enter__(self) -> Self:
        saved = (
            self.checkpointer.get_tuple(self.config) if self.checkpointer else None
        ) or CheckpointTuple(self.config, empty_checkpoint(), {"step": -2}, None, [])
        self.checkpoint_config = {
            **self.config,
            **saved.config,
            "configurable": {
                **self.config.get("configurable", {}),
                **saved.config.get("configurable", {}),
            },
        }
        self.checkpoint = copy_checkpoint(saved.checkpoint)
        self.checkpoint_metadata = saved.metadata
        self.checkpoint_pending_writes = saved.pending_writes or []

        self.submit = self.stack.enter_context(BackgroundExecutor(self.config))
        self.channels, self.managed = self.stack.enter_context(
            ChannelsManager(self.specs, self.checkpoint, self.config, self.store)
        )
        self.stack.push(self._suppress_interrupt)
        self.status = "pending"
        self.step = self.checkpoint_metadata["step"] + 1
        self.stop = self.step + self.config["recursion_limit"] + 1
        self.checkpoint_previous_versions = self.checkpoint["channel_versions"].copy()

        return self

    def __exit__(
        self,
        exc_type: Optional[Type[BaseException]],
        exc_value: Optional[BaseException],
        traceback: Optional[TracebackType],
    ) -> Optional[bool]:
        # unwind stack
        return self.stack.__exit__(exc_type, exc_value, traceback)


class AsyncPregelLoop(PregelLoop, AsyncContextManager):
    def __init__(
        self,
        input: Optional[Any],
        *,
        config: RunnableConfig,
        store: Optional[BaseStore],
        checkpointer: Optional[BaseCheckpointSaver],
        nodes: Mapping[str, PregelNode],
        specs: Mapping[str, Union[BaseChannel, ManagedValueSpec]],
        output_keys: Union[str, Sequence[str]] = EMPTY_SEQ,
        stream_keys: Union[str, Sequence[str]] = EMPTY_SEQ,
    ) -> None:
        super().__init__(
            input,
            config=config,
            checkpointer=checkpointer,
            store=store,
            nodes=nodes,
            specs=specs,
            output_keys=output_keys,
            stream_keys=stream_keys,
        )
        self.store = AsyncBatchedStore(self.store) if self.store else None
        self.stack = AsyncExitStack()
        if checkpointer:
            self.checkpointer_get_next_version = checkpointer.get_next_version
            self.checkpointer_put_writes = checkpointer.aput_writes
        else:
            self.checkpointer_get_next_version = increment
            self._checkpointer_put_after_previous = None
            self.checkpointer_put_writes = None

    async def _checkpointer_put_after_previous(
        self,
        prev: Optional[asyncio.Task],
        config: RunnableConfig,
        checkpoint: Checkpoint,
        metadata: CheckpointMetadata,
        new_versions: Optional[dict[str, Union[str, float, int]]],
    ) -> RunnableConfig:
        try:
            if prev is not None:
                await prev
        finally:
            await self.checkpointer.aput(config, checkpoint, metadata, new_versions)

    def _update_mv(self, key: str, values: Sequence[Any]) -> None:
        return self.submit(
            cast(WritableManagedValue, self.managed[key]).aupdate, values
        )

    # context manager

    async def __aenter__(self) -> Self:
        saved = (
            await self.checkpointer.aget_tuple(self.config)
            if self.checkpointer
            else None
        ) or CheckpointTuple(self.config, empty_checkpoint(), {"step": -2}, None, [])
        self.checkpoint_config = {
            **self.config,
            **saved.config,
            "configurable": {
                **self.config.get("configurable", {}),
                **saved.config.get("configurable", {}),
            },
        }
        self.checkpoint = copy_checkpoint(saved.checkpoint)
        self.checkpoint_metadata = saved.metadata
        self.checkpoint_pending_writes = saved.pending_writes or []

        self.submit = await self.stack.enter_async_context(AsyncBackgroundExecutor())
        self.channels, self.managed = await self.stack.enter_async_context(
            AsyncChannelsManager(self.specs, self.checkpoint, self.config, self.store)
        )
        self.stack.push(self._suppress_interrupt)
        self.status = "pending"
        self.step = self.checkpoint_metadata["step"] + 1
        self.stop = self.step + self.config["recursion_limit"] + 1

        self.checkpoint_previous_versions = self.checkpoint["channel_versions"].copy()

        return self

    async def __aexit__(
        self,
        exc_type: Optional[Type[BaseException]],
        exc_value: Optional[BaseException],
        traceback: Optional[TracebackType],
    ) -> Optional[bool]:
        # unwind stack
        return await asyncio.shield(
            self.stack.__aexit__(exc_type, exc_value, traceback)
        )
