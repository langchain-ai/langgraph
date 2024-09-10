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
    Iterable,
    List,
    Literal,
    Mapping,
    Optional,
    Protocol,
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
    CONFIG_KEY_CHECKPOINT_MAP,
    CONFIG_KEY_DEDUPE_TASKS,
    CONFIG_KEY_ENSURE_LATEST,
    CONFIG_KEY_RESUMING,
    CONFIG_KEY_STREAM,
    CONFIG_KEY_TASK_ID,
    ERROR,
    INPUT,
    INTERRUPT,
    SCHEDULED,
    TAG_HIDDEN,
)
from langgraph.errors import CheckpointNotLatest, EmptyInputError, GraphInterrupt
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
    print_step_checkpoint,
    print_step_tasks,
    print_step_writes,
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
from langgraph.utils.config import patch_configurable

V = TypeVar("V")
INPUT_DONE = object()
INPUT_RESUMING = object()
EMPTY_SEQ = ()


class StreamProtocol(Protocol):
    def __call__(self, values: Iterable[Tuple[str, str, Any]]) -> None: ...


class DuplexStream(StreamProtocol):
    def __init__(self, *queues: StreamProtocol) -> None:
        self.queues = queues

    def __call__(self, value: Tuple[str, str, Any]) -> None:
        for queue in self.queues:
            queue(value)


class PregelLoop:
    input: Optional[Any]
    config: RunnableConfig
    store: Optional[BaseStore]
    checkpointer: Optional[BaseCheckpointSaver]
    nodes: Mapping[str, PregelNode]
    specs: Mapping[str, Union[BaseChannel, ManagedValueSpec]]
    output_keys: Union[str, Sequence[str]]
    stream_keys: Union[str, Sequence[str]]
    stream: Optional[StreamProtocol]
    skip_done_tasks: bool
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
    tasks: dict[str, PregelExecutableTask]
    output: Union[None, dict[str, Any], Any] = None

    # public

    def __init__(
        self,
        input: Optional[Any],
        *,
        stream: Optional[StreamProtocol],
        config: RunnableConfig,
        store: Optional[BaseStore],
        checkpointer: Optional[BaseCheckpointSaver],
        nodes: Mapping[str, PregelNode],
        specs: Mapping[str, Union[BaseChannel, ManagedValueSpec]],
        output_keys: Union[str, Sequence[str]],
        stream_keys: Union[str, Sequence[str]],
        debug: bool = False,
    ) -> None:
        self.stream = stream
        self.input = input
        self.config = config
        self.store = store
        self.checkpointer = checkpointer
        self.nodes = nodes
        self.specs = specs
        self.output_keys = output_keys
        self.stream_keys = stream_keys
        self.is_nested = CONFIG_KEY_TASK_ID in self.config.get("configurable", {})
        self.skip_done_tasks = (
            "checkpoint_id" not in config["configurable"]
            or CONFIG_KEY_DEDUPE_TASKS in config["configurable"]
        )
        self.debug = debug
        if CONFIG_KEY_STREAM in config["configurable"]:
            self.stream = DuplexStream(
                self.stream, config["configurable"][CONFIG_KEY_STREAM]
            )
        if not self.is_nested and config["configurable"].get("checkpoint_ns"):
            self.config = patch_configurable(
                self.config, {"checkpoint_ns": "", "checkpoint_id": None}
            )
        if (
            CONFIG_KEY_CHECKPOINT_MAP in self.config["configurable"]
            and self.config["configurable"].get("checkpoint_ns")
            in self.config["configurable"][CONFIG_KEY_CHECKPOINT_MAP]
        ):
            self.checkpoint_config = patch_configurable(
                self.config,
                {
                    "checkpoint_id": config["configurable"][CONFIG_KEY_CHECKPOINT_MAP][
                        self.config["configurable"]["checkpoint_ns"]
                    ]
                },
            )
        else:
            self.checkpoint_config = config

    def put_writes(self, task_id: str, writes: Sequence[tuple[str, Any]]) -> None:
        """Put writes for a task, to be read by the next tick."""
        if not writes:
            return
        self.checkpoint_pending_writes.extend((task_id, k, v) for k, v in writes)
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
                writes,
                task_id,
            )
        self._output_writes(task_id, writes)

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
        elif all(task.writes for task in self.tasks.values()):
            writes = [w for t in self.tasks.values() for w in t.writes]
            # debug flag
            if self.debug:
                print_step_writes(
                    self.step,
                    writes,
                    [self.stream_keys]
                    if isinstance(self.stream_keys, str)
                    else self.stream_keys,
                )
            # all tasks have finished
            mv_writes = apply_writes(
                self.checkpoint,
                self.channels,
                self.tasks.values(),
                self.checkpointer_get_next_version,
            )
            # apply writes to managed values
            for key, values in mv_writes.items():
                self._update_mv(key, values)
            # produce values output
            self._emit(
                (self.config["configurable"].get("checkpoint_ns", ""), "values", v)
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
                            self.output_keys,
                            [(t, t.writes) for t in self.tasks.values()],
                        )
                    ),
                }
            )
            # after execution, check if we should interrupt
            if should_interrupt(self.checkpoint, interrupt_after, self.tasks.values()):
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
        )

        # produce debug output
        if self._checkpointer_put_after_previous is not None:
            self._emit(
                (self.config["configurable"].get("checkpoint_ns", ""), "debug", v)
                for v in map_debug_checkpoint(
                    self.step - 1,  # printing checkpoint for previous step
                    self.checkpoint_config,
                    self.channels,
                    self.stream_keys,
                    self.checkpoint_metadata,
                    self.checkpoint,
                    self.tasks.values(),
                    self.checkpoint_pending_writes,
                )
            )

        # if no more tasks, we're done
        if not self.tasks:
            self.status = "done"
            return False

        # if there are pending writes from a previous loop, apply them
        if self.skip_done_tasks and self.checkpoint_pending_writes:
            for tid, k, v in self.checkpoint_pending_writes:
                if k in (ERROR, INTERRUPT):
                    continue
                if task := self.tasks.get(tid):
                    if k == SCHEDULED:
                        if v == max(
                            self.checkpoint["versions_seen"]
                            .get(INTERRUPT, {})
                            .values(),
                            default=None,
                        ):
                            self.tasks[tid] = task._replace(scheduled=True)
                    else:
                        task.writes.append((k, v))
            # print output for any tasks we applied previous writes to
            for task in self.tasks.values():
                if task.writes:
                    self._output_writes(task.id, task.writes, cached=True)

        # if all tasks have finished, re-tick
        if all(task.writes for task in self.tasks.values()):
            return self.tick(
                input_keys=input_keys,
                interrupt_after=interrupt_after,
                interrupt_before=interrupt_before,
                manager=manager,
            )

        # before execution, check if we should interrupt
        if should_interrupt(self.checkpoint, interrupt_before, self.tasks.values()):
            self.status = "interrupt_before"
            if self.is_nested:
                raise GraphInterrupt()
            else:
                return False

        # produce debug output
        self._emit(
            (self.config["configurable"].get("checkpoint_ns", ""), "debug", v)
            for v in map_debug_tasks(self.step, self.tasks.values())
        )

        # debug flag
        if self.debug:
            print_step_tasks(self.step, self.tasks.values())

        return True

    # private

    def _first(self, *, input_keys: Union[str, Sequence[str]]) -> None:
        # resuming from previous checkpoint requires
        # - finding a previous checkpoint
        # - receiving None input (outer graph) or RESUMING flag (subgraph)
        configurable = self.config.get("configurable", {})
        is_resuming = bool(self.checkpoint["channel_versions"]) and bool(
            configurable.get(CONFIG_KEY_RESUMING, self.input is None)
        )

        # proceed past previous checkpoint
        if is_resuming:
            self.checkpoint["versions_seen"].setdefault(INTERRUPT, {})
            for k in self.channels:
                if k in self.checkpoint["channel_versions"]:
                    version = self.checkpoint["channel_versions"][k]
                    self.checkpoint["versions_seen"][INTERRUPT][k] = version
            # produce values output
            self._emit(
                (self.config["configurable"].get("checkpoint_ns", ""), "values", v)
                for v in map_output_values(self.output_keys, True, self.channels)
            )
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
                [*discard_tasks.values(), PregelTaskWrites(INPUT, input_writes, [])],
                self.checkpointer_get_next_version,
            ), "Can't write to SharedValues in graph input"
            # save input checkpoint
            self._put_checkpoint({"source": "input", "writes": dict(input_writes)})
        elif CONFIG_KEY_RESUMING not in configurable:
            raise EmptyInputError(f"Received no input for {input_keys}")
        # done with input
        self.input = INPUT_RESUMING if is_resuming else INPUT_DONE
        # update config
        if not self.is_nested:
            self.config = patch_configurable(
                self.config, {CONFIG_KEY_RESUMING: is_resuming}
            )

    def _put_checkpoint(self, metadata: CheckpointMetadata) -> None:
        # assign step
        metadata["step"] = self.step
        metadata["parents"] = self.config["configurable"].get(
            CONFIG_KEY_CHECKPOINT_MAP, {}
        )
        # debug flag
        if self.debug:
            print_step_checkpoint(
                metadata,
                self.channels,
                [self.stream_keys]
                if isinstance(self.stream_keys, str)
                else self.stream_keys,
            )
        # create new checkpoint
        self.checkpoint = create_checkpoint(self.checkpoint, self.channels, self.step)
        # bail if no checkpointer
        if self._checkpointer_put_after_previous is not None:
            self.checkpoint_metadata = metadata
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

    def _emit(self, values: Sequence[tuple[str, str, Any]]) -> None:
        if self.stream is None:
            return
        for v in values:
            self.stream(v)

    def _output_writes(
        self, task_id: str, writes: Sequence[tuple[str, Any]], *, cached: bool = False
    ) -> None:
        if task := self.tasks.get(task_id):
            if task.config is not None and TAG_HIDDEN in task.config.get("tags"):
                return
            if writes[0][0] != ERROR and writes[0][0] != INTERRUPT:
                self._emit(
                    (self.config["configurable"].get("checkpoint_ns", ""), "updates", v)
                    for v in map_output_updates(
                        self.output_keys, [(task, writes)], cached
                    )
                )
            if not cached:
                self._emit(
                    (self.config["configurable"].get("checkpoint_ns", ""), "debug", v)
                    for v in map_debug_task_results(
                        self.step, (task, writes), self.stream_keys
                    )
                )


class SyncPregelLoop(PregelLoop, ContextManager):
    def __init__(
        self,
        input: Optional[Any],
        *,
        stream: Optional[StreamProtocol],
        config: RunnableConfig,
        store: Optional[BaseStore],
        checkpointer: Optional[BaseCheckpointSaver],
        nodes: Mapping[str, PregelNode],
        specs: Mapping[str, Union[BaseChannel, ManagedValueSpec]],
        output_keys: Union[str, Sequence[str]] = EMPTY_SEQ,
        stream_keys: Union[str, Sequence[str]] = EMPTY_SEQ,
        debug: bool = False,
    ) -> None:
        super().__init__(
            input,
            stream=stream,
            config=config,
            checkpointer=checkpointer,
            store=store,
            nodes=nodes,
            specs=specs,
            output_keys=output_keys,
            stream_keys=stream_keys,
            debug=debug,
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
        if self.config.get("configurable", {}).get(
            CONFIG_KEY_ENSURE_LATEST
        ) and self.checkpoint_config["configurable"].get("checkpoint_id"):
            saved = self.checkpointer.get_tuple(
                patch_configurable(self.checkpoint_config, {"checkpoint_id": None})
            )
            if (
                saved is None
                or saved.checkpoint["id"]
                != self.checkpoint_config["configurable"]["checkpoint_id"]
            ):
                raise CheckpointNotLatest
        elif self.checkpointer:
            saved = self.checkpointer.get_tuple(self.checkpoint_config)
        else:
            saved = None
        if saved is None:
            saved = CheckpointTuple(
                self.config, empty_checkpoint(), {"step": -2}, None, []
            )
        self.checkpoint_config = {
            **self.config,
            **saved.config,
            "configurable": {
                "checkpoint_ns": "",
                **self.config.get("configurable", {}),
                **saved.config.get("configurable", {}),
            },
        }
        self.checkpoint = copy_checkpoint(saved.checkpoint)
        self.checkpoint_metadata = saved.metadata
        self.checkpoint_pending_writes = (
            [(str(tid), k, v) for tid, k, v in saved.pending_writes]
            if saved.pending_writes is not None
            else []
        )

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
        stream: Optional[StreamProtocol],
        config: RunnableConfig,
        store: Optional[BaseStore],
        checkpointer: Optional[BaseCheckpointSaver],
        nodes: Mapping[str, PregelNode],
        specs: Mapping[str, Union[BaseChannel, ManagedValueSpec]],
        output_keys: Union[str, Sequence[str]] = EMPTY_SEQ,
        stream_keys: Union[str, Sequence[str]] = EMPTY_SEQ,
        debug: bool = False,
    ) -> None:
        super().__init__(
            input,
            stream=stream,
            config=config,
            checkpointer=checkpointer,
            store=store,
            nodes=nodes,
            specs=specs,
            output_keys=output_keys,
            stream_keys=stream_keys,
            debug=debug,
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
        if self.config.get("configurable", {}).get(
            CONFIG_KEY_ENSURE_LATEST
        ) and self.checkpoint_config["configurable"].get("checkpoint_id"):
            saved = await self.checkpointer.aget_tuple(
                patch_configurable(self.checkpoint_config, {"checkpoint_id": None})
            )
            if (
                saved is None
                or saved.checkpoint["id"]
                != self.checkpoint_config["configurable"]["checkpoint_id"]
            ):
                raise CheckpointNotLatest
        elif self.checkpointer:
            saved = await self.checkpointer.aget_tuple(self.checkpoint_config)
        else:
            saved = None
        if saved is None:
            saved = CheckpointTuple(
                self.config, empty_checkpoint(), {"step": -2}, None, []
            )
        self.checkpoint_config = {
            **self.config,
            **saved.config,
            "configurable": {
                "checkpoint_ns": "",
                **self.config.get("configurable", {}),
                **saved.config.get("configurable", {}),
            },
        }
        self.checkpoint = copy_checkpoint(saved.checkpoint)
        self.checkpoint_metadata = saved.metadata
        self.checkpoint_pending_writes = (
            [(str(tid), k, v) for tid, k, v in saved.pending_writes]
            if saved.pending_writes is not None
            else []
        )

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
