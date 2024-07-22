import asyncio
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
)

from langchain_core.callbacks import AsyncParentRunManager, ParentRunManager
from langchain_core.runnables import RunnableConfig
from typing_extensions import Self

from langgraph.channels.base import BaseChannel
from langgraph.channels.manager import (
    AsyncChannelsManager,
    ChannelsManager,
    create_checkpoint,
)
from langgraph.checkpoint.base import (
    BaseCheckpointSaver,
    Checkpoint,
    CheckpointMetadata,
    CheckpointTuple,
    PendingWrite,
    copy_checkpoint,
    empty_checkpoint,
)
from langgraph.constants import CONFIG_KEY_READ, CONFIG_KEY_RESUMING, INPUT, INTERRUPT
from langgraph.errors import EmptyInputError, GraphInterrupt
from langgraph.managed.base import (
    AsyncManagedValuesManager,
    ManagedValueMapping,
    ManagedValuesManager,
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

if TYPE_CHECKING:
    from langgraph.pregel import Pregel


V = TypeVar("V")
INPUT_DONE = object()
INPUT_RESUMING = object()
EMPTY_LIST = []


class PregelLoop:
    input: Optional[Any]
    config: RunnableConfig
    checkpointer: Optional[BaseCheckpointSaver]
    checkpointer_get_next_version: Callable[[Optional[V]], V]
    checkpointer_put_writes: Optional[
        Callable[[RunnableConfig, Sequence[tuple[str, Any]], str], Any]
    ]
    checkpointer_put: Optional[
        Callable[[RunnableConfig, Checkpoint, CheckpointMetadata], Any]
    ]
    graph: "Pregel"

    submit: Submit
    channels: Mapping[str, BaseChannel]
    managed: ManagedValueMapping
    checkpoint: Checkpoint
    checkpoint_config: RunnableConfig
    checkpoint_metadata: CheckpointMetadata
    checkpoint_pending_writes: Optional[List[PendingWrite]]

    step: int
    status: Literal[
        "pending", "done", "interrupt_before", "interrupt_after", "out_of_steps"
    ]
    tasks: Sequence[PregelExecutableTask]
    stream: deque[Tuple[str, Any]]
    is_nested: bool

    # public

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
                        "thread_ts": self.checkpoint["id"],
                    },
                },
                writes,
                task_id,
            )

    def tick(
        self,
        *,
        output_keys: Union[str, Sequence[str]] = EMPTY_LIST,
        interrupt_after: Sequence[str] = EMPTY_LIST,
        interrupt_before: Sequence[str] = EMPTY_LIST,
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
            apply_writes(
                self.checkpoint,
                self.channels,
                self.tasks,
                self.checkpointer_get_next_version,
            )
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
            if should_interrupt(self.checkpoint, interrupt_after, self.tasks):
                self.status = "interrupt_after"
                if self.is_nested:
                    raise GraphInterrupt(self)
                else:
                    return False
        else:
            return False

        # check if iteration limit is reached
        if self.step > self.config["recursion_limit"]:
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

        # if no more tasks, we're done
        if not self.tasks:
            self.status = "done"
            return False

        # if there are pending writes from a previous loop, apply them
        if self.checkpoint_pending_writes:
            for tid, k, v in self.checkpoint_pending_writes:
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
        if should_interrupt(self.checkpoint, interrupt_before, self.tasks):
            self.status = "interrupt_before"
            if self.is_nested:
                raise GraphInterrupt(self)
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
            apply_writes(
                self.checkpoint,
                self.channels,
                discard_tasks + [PregelTaskWrites(INPUT, input_writes, [])],
                self.checkpointer_get_next_version,
            )
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
        if self.checkpointer_put is not None:
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
                id=self.config["configurable"]["thread_ts"] if self.is_nested else None,
            )
            # save it, without blocking
            self.submit(
                self.checkpointer_put,
                self.checkpoint_config,
                copy_checkpoint(self.checkpoint),
                self.checkpoint_metadata,
            )
            self.checkpoint_config = {
                **self.checkpoint_config,
                "configurable": {
                    **self.checkpoint_config["configurable"],
                    "thread_ts": self.checkpoint["id"],
                },
            }
            # produce debug output
            self.stream.extend(
                ("debug", v)
                for v in map_debug_checkpoint(
                    self.step,
                    self.checkpoint_config,
                    self.channels,
                    self.graph.stream_channels_asis,
                    self.checkpoint_metadata,
                )
            )
        # increment step
        self.step += 1


class SyncPregelLoop(PregelLoop, ContextManager):
    def __init__(
        self,
        input: Optional[Any],
        *,
        config: RunnableConfig,
        checkpointer: Optional[BaseCheckpointSaver],
        graph: "Pregel",
    ) -> None:
        self.stream = deque()
        self.stack = ExitStack()
        self.input = input
        self.config = config
        self.checkpointer = checkpointer
        self.checkpointer_get_next_version = (
            checkpointer.get_next_version if checkpointer else increment
        )
        self.checkpointer_put_writes = checkpointer.put_writes if checkpointer else None
        self.checkpointer_put = checkpointer.put if checkpointer else None
        self.graph = graph
        # TODO if managed values no longer needs graph we can replace with
        # managed_specs, channel_specs

    # context manager

    def __enter__(self) -> Self:
        self.is_nested = CONFIG_KEY_READ in self.config.get("configurable", {})
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
        self.checkpoint_pending_writes = saved.pending_writes

        self.submit = self.stack.enter_context(BackgroundExecutor(self.config))
        self.channels = self.stack.enter_context(
            ChannelsManager(self.graph.channels, self.checkpoint, self.config)
        )
        self.managed = self.stack.enter_context(
            ManagedValuesManager(
                self.graph.managed_values_dict, self.config, self.graph
            )
        )
        self.status = "pending"
        self.step = self.checkpoint_metadata["step"] + 1

        return self

    def __exit__(
        self,
        exc_type: Optional[Type[BaseException]],
        exc_value: Optional[BaseException],
        traceback: Optional[TracebackType],
    ) -> Optional[bool]:
        # handle interrupt
        if exc_type is GraphInterrupt:
            if exc_value.args[0] is self:
                # interrupt raised by this loop
                exc_value.args = (object(),)
            if not self.is_nested:
                # in outer graph, catch interrupt
                del self.graph
                return True or self.stack.__exit__(None, None, None)

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
        self.stream = deque()
        self.stack = AsyncExitStack()
        self.input = input
        self.config = config
        self.checkpointer = checkpointer
        self.checkpointer_get_next_version = (
            checkpointer.get_next_version if checkpointer else increment
        )
        self.checkpointer_put_writes = (
            checkpointer.aput_writes if checkpointer else None
        )
        self.checkpointer_put = checkpointer.aput if checkpointer else None
        self.graph = graph
        # TODO if managed values no longer needs graph we can replace with
        # managed_specs, channel_specs

    # context manager

    async def __aenter__(self) -> Self:
        self.is_nested = CONFIG_KEY_READ in self.config.get("configurable", {})
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
        self.checkpoint_pending_writes = saved.pending_writes

        self.submit = await self.stack.enter_async_context(AsyncBackgroundExecutor())
        self.channels = await self.stack.enter_async_context(
            AsyncChannelsManager(self.graph.channels, self.checkpoint, self.config)
        )
        self.managed = await self.stack.enter_async_context(
            AsyncManagedValuesManager(
                self.graph.managed_values_dict, self.config, self.graph
            )
        )
        self.status = "pending"
        self.step = self.checkpoint_metadata["step"] + 1

        return self

    async def __aexit__(
        self,
        exc_type: Optional[Type[BaseException]],
        exc_value: Optional[BaseException],
        traceback: Optional[TracebackType],
    ) -> Optional[bool]:
        # handle interrupt
        if exc_type is GraphInterrupt:
            if exc_value.args[0] is self:
                # interrupt raised by this loop
                exc_value.args = (object(),)
            if not self.is_nested:
                # in outer graph, catch interrupt
                del self.graph
                return True or await asyncio.shield(
                    self.stack.__aexit__(None, None, None)
                )

        # unwind stack
        del self.graph
        return await asyncio.shield(
            self.stack.__aexit__(exc_type, exc_value, traceback)
        )
