import asyncio
import concurrent.futures
from collections import deque
from contextlib import AsyncExitStack, ExitStack
from types import TracebackType
from typing import (
    TYPE_CHECKING,
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
from langchain_core.runnables import RunnableConfig, patch_config
from typing_extensions import Self

from langgraph.channels.base import BaseChannel
from langgraph.channels.manager import (
    AsyncChannelsManager,
    ChannelsManager,
)
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
    CONFIG_KEY_KV,
    CONFIG_KEY_READ,
    CONFIG_KEY_RESUMING,
    ERROR,
    INPUT,
    INTERRUPT,
    Interrupt,
)
from langgraph.errors import EmptyInputError, GraphInterrupt
from langgraph.managed.base import (
    AsyncManagedValuesManager,
    ManagedValueMapping,
    ManagedValuesManager,
    WritableManagedValue,
)
from langgraph.pregel.algo import (
    PregelTaskWrites,
    apply_writes,
    increment,
    prepare_next_tasks,
    should_interrupt,
)
from langgraph.pregel.debug import map_debug_checkpoint, map_debug_tasks
from langgraph.pregel.executor import (
    AsyncBackgroundExecutor,
    BackgroundExecutor,
    Submit,
)
from langgraph.pregel.io import map_input, map_output_updates, map_output_values, single
from langgraph.pregel.types import PregelExecutableTask
from langgraph.pregel.utils import get_new_channel_versions

if TYPE_CHECKING:
    from langgraph.pregel import Pregel


V = TypeVar("V")
INPUT_DONE = object()
INPUT_RESUMING = object()
EMPTY_SEQ = ()


class PregelLoop:
    input: Optional[Any]
    config: RunnableConfig
    checkpointer: Optional[BaseCheckpointSaver]
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
    graph: "Pregel"

    submit: Submit
    channels: Mapping[str, BaseChannel]
    managed: ManagedValueMapping
    checkpoint: Checkpoint
    checkpoint_config: RunnableConfig
    checkpoint_metadata: CheckpointMetadata
    checkpoint_pending_writes: List[PendingWrite]
    # (thread_id, checkpoint_ns -> channel_versions)
    checkpoint_previous_versions: dict[str, Union[str, float, int]]

    step: int
    stop: int
    status: Literal[
        "pending", "done", "interrupt_before", "interrupt_after", "out_of_steps"
    ]
    tasks: Sequence[PregelExecutableTask]
    stream: deque[Tuple[str, Any]]
    is_nested: bool

    # public

    def __init__(
        self,
        input: Optional[Any],
        *,
        config: RunnableConfig,
        checkpointer: Optional[BaseCheckpointSaver],
        graph: "Pregel",
    ) -> None:
        self.stream = deque()
        self.input = input
        self.config = config
        self.checkpointer = checkpointer
        self.graph = graph
        # TODO if managed values no longer needs graph we can replace with
        # managed_specs, channel_specs
        self.is_nested = CONFIG_KEY_READ in self.config.get("configurable", {})

    def mark_tasks_scheduled(self, tasks: Sequence[PregelExecutableTask]) -> None:
        """Mark tasks as scheduled, to be used by queue-based executors."""
        raise NotImplementedError

    def put_writes(self, task_id: str, writes: Sequence[tuple[str, Any]]) -> None:
        """Put writes for a task, to be read by the next tick."""
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

    def tick(
        self,
        *,
        output_keys: Union[str, Sequence[str]] = EMPTY_SEQ,
        interrupt_after: Sequence[str] = EMPTY_SEQ,
        interrupt_before: Sequence[str] = EMPTY_SEQ,
        manager: Union[None, AsyncParentRunManager, ParentRunManager] = None,
    ) -> bool:
        """Execute a single iteration of the Pregel loop.
        Returns True if more iterations are needed."""

        if self.status != "pending":
            raise RuntimeError("Cannot tick when status is no longer 'pending'")

        if self.input not in (INPUT_DONE, INPUT_RESUMING):
            self._first()
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
                for v in map_output_values(output_keys, writes, self.channels)
            )
            # clear pending writes
            self.checkpoint_pending_writes.clear()
            # save checkpoint
            self._put_checkpoint(
                {
                    "source": "loop",
                    "writes": single(
                        map_output_updates(output_keys, self.tasks)
                        if self.graph.stream_mode == "updates"
                        else map_output_values(output_keys, writes, self.channels)
                    ),
                }
            )
            # after execution, check if we should interrupt
            if tasks := should_interrupt(self.checkpoint, interrupt_after, self.tasks):
                self.status = "interrupt_after"
                interrupts = [(t.id, Interrupt("after")) for t in tasks]
                for tid, interrupt in interrupts:
                    self.put_writes(tid, [(INTERRUPT, interrupt)])
                if self.is_nested:
                    raise GraphInterrupt([i[1] for i in interrupts])
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
            self.graph.nodes,
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
                    self.graph.stream_channels_asis,
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
                output_keys=output_keys,
                interrupt_after=interrupt_after,
                interrupt_before=interrupt_before,
                manager=manager,
            )

        # before execution, check if we should interrupt
        if tasks := should_interrupt(self.checkpoint, interrupt_before, self.tasks):
            self.status = "interrupt_before"
            interrupts = [(t.id, Interrupt("before")) for t in tasks]
            for tid, interrupt in interrupts:
                self.put_writes(tid, [(INTERRUPT, interrupt)])
            if self.is_nested:
                raise GraphInterrupt([i[1] for i in interrupts])
            else:
                return False

        # produce debug output
        self.stream.extend(("debug", v) for v in map_debug_tasks(self.step, self.tasks))

        return True

    # private

    def _first(self) -> None:
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
        elif input_writes := deque(map_input(self.graph.input_channels, self.input)):
            # discard any unfinished tasks from previous checkpoint
            discard_tasks = prepare_next_tasks(
                self.checkpoint,
                self.graph.nodes,
                self.channels,
                self.managed,
                self.config,
                self.step,
                for_execution=True,
                manager=None,
            )
            # apply input writes
            mv_writes = apply_writes(
                self.checkpoint,
                self.channels,
                discard_tasks + [PregelTaskWrites(INPUT, input_writes, [])],
                self.checkpointer_get_next_version,
            )
            assert not mv_writes
            # save input checkpoint
            self._put_checkpoint({"source": "input", "writes": self.input})
        else:
            raise EmptyInputError(f"Received no input for {self.graph.input_channels}")
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
        if isinstance(exc_value, GraphInterrupt) and not self.is_nested:
            return True


class SyncPregelLoop(PregelLoop, ContextManager):
    def __init__(
        self,
        input: Optional[Any],
        *,
        config: RunnableConfig,
        checkpointer: Optional[BaseCheckpointSaver],
        graph: "Pregel",
    ) -> None:
        super().__init__(input, config=config, checkpointer=checkpointer, graph=graph)
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
        self.channels = self.stack.enter_context(
            ChannelsManager(self.graph.channels, self.checkpoint, self.config)
        )
        self.managed = self.stack.enter_context(
            ManagedValuesManager(
                self.graph.managed_values_dict,
                patch_config(self.config, configurable={CONFIG_KEY_KV: self.graph.kv}),
            )
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
        del self.graph
        return self.stack.__exit__(exc_type, exc_value, traceback)


class AsyncPregelLoop(PregelLoop, AsyncContextManager):
    def __init__(
        self,
        input: Optional[Any],
        *,
        config: RunnableConfig,
        checkpointer: Optional[BaseCheckpointSaver],
        graph: "Pregel",
    ) -> None:
        super().__init__(input, config=config, checkpointer=checkpointer, graph=graph)
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
        self.channels = await self.stack.enter_async_context(
            AsyncChannelsManager(self.graph.channels, self.checkpoint, self.config)
        )
        self.managed = await self.stack.enter_async_context(
            AsyncManagedValuesManager(
                self.graph.managed_values_dict,
                patch_config(self.config, configurable={CONFIG_KEY_KV: self.graph.kv}),
            )
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
        del self.graph
        return await asyncio.shield(
            self.stack.__aexit__(exc_type, exc_value, traceback)
        )
