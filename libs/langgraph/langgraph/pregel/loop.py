import asyncio
import concurrent.futures
from collections import defaultdict, deque
from contextlib import AsyncExitStack, ExitStack
from inspect import signature
from types import TracebackType
from typing import (
    Any,
    AsyncContextManager,
    Callable,
    ContextManager,
    Iterator,
    List,
    Literal,
    Mapping,
    Optional,
    Sequence,
    Type,
    TypeVar,
    Union,
    cast,
)

from langchain_core.callbacks import AsyncParentRunManager, ParentRunManager
from langchain_core.runnables import RunnableConfig
from pydantic import BaseModel
from typing_extensions import ParamSpec, Self

from langgraph.channels.base import BaseChannel
from langgraph.checkpoint.base import (
    WRITES_IDX_MAP,
    BaseCheckpointSaver,
    ChannelVersions,
    Checkpoint,
    CheckpointMetadata,
    CheckpointTuple,
    PendingWrite,
    copy_checkpoint,
    create_checkpoint,
    empty_checkpoint,
)
from langgraph.constants import (
    CONF,
    CONFIG_KEY_CHECKPOINT_ID,
    CONFIG_KEY_CHECKPOINT_MAP,
    CONFIG_KEY_CHECKPOINT_NS,
    CONFIG_KEY_DEDUPE_TASKS,
    CONFIG_KEY_DELEGATE,
    CONFIG_KEY_ENSURE_LATEST,
    CONFIG_KEY_RESUMING,
    CONFIG_KEY_SCRATCHPAD,
    CONFIG_KEY_STREAM,
    CONFIG_KEY_TASK_ID,
    EMPTY_SEQ,
    ERROR,
    INPUT,
    INTERRUPT,
    MISSING,
    NS_SEP,
    NULL_TASK_ID,
    PUSH,
    RESUME,
    SCHEDULED,
    TAG_HIDDEN,
)
from langgraph.errors import (
    CheckpointNotLatest,
    EmptyInputError,
    GraphDelegate,
    GraphInterrupt,
)
from langgraph.managed.base import (
    ManagedValueMapping,
    ManagedValueSpec,
    WritableManagedValue,
)
from langgraph.pregel.algo import (
    Call,
    GetNextVersion,
    PregelTaskWrites,
    apply_writes,
    increment,
    prepare_next_tasks,
    prepare_single_task,
    should_interrupt,
    task_path_str,
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
    map_command,
    map_input,
    map_output_updates,
    map_output_values,
    read_channels,
    single,
)
from langgraph.pregel.manager import AsyncChannelsManager, ChannelsManager
from langgraph.pregel.read import PregelNode
from langgraph.pregel.utils import get_new_channel_versions
from langgraph.store.base import BaseStore
from langgraph.types import (
    All,
    Command,
    LoopProtocol,
    PregelExecutableTask,
    PregelScratchpad,
    StreamChunk,
    StreamProtocol,
)
from langgraph.utils.config import patch_configurable

V = TypeVar("V")
P = ParamSpec("P")

INPUT_DONE = object()
INPUT_RESUMING = object()
INPUT_SHOULD_VALIDATE = object()
SPECIAL_CHANNELS = (ERROR, INTERRUPT, SCHEDULED)


def DuplexStream(*streams: StreamProtocol) -> StreamProtocol:
    def __call__(value: StreamChunk) -> None:
        for stream in streams:
            if value[1] in stream.modes:
                stream(value)

    return StreamProtocol(__call__, {mode for s in streams for mode in s.modes})


class PregelLoop(LoopProtocol):
    input: Optional[Any]
    input_model: Optional[Type[BaseModel]]
    checkpointer: Optional[BaseCheckpointSaver]
    nodes: Mapping[str, PregelNode]
    specs: Mapping[str, Union[BaseChannel, ManagedValueSpec]]
    output_keys: Union[str, Sequence[str]]
    stream_keys: Union[str, Sequence[str]]
    skip_done_tasks: bool
    is_nested: bool
    manager: Union[None, AsyncParentRunManager, ParentRunManager]
    interrupt_after: Union[All, Sequence[str]]
    interrupt_before: Union[All, Sequence[str]]

    checkpointer_get_next_version: GetNextVersion
    checkpointer_put_writes: Optional[
        Callable[[RunnableConfig, Sequence[tuple[str, Any]], str], Any]
    ]
    checkpointer_put_writes_accepts_task_path: bool
    _checkpointer_put_after_previous: Optional[
        Callable[
            [
                Optional[concurrent.futures.Future],
                RunnableConfig,
                Sequence[tuple[str, Any]],
                str,
                ChannelVersions,
            ],
            Any,
        ]
    ]
    submit: Submit
    channels: Mapping[str, BaseChannel]
    managed: ManagedValueMapping
    checkpoint: Checkpoint
    checkpoint_ns: tuple[str, ...]
    checkpoint_config: RunnableConfig
    checkpoint_metadata: CheckpointMetadata
    checkpoint_pending_writes: List[PendingWrite]
    checkpoint_previous_versions: dict[str, Union[str, float, int]]
    prev_checkpoint_config: Optional[RunnableConfig]

    status: Literal[
        "pending", "done", "interrupt_before", "interrupt_after", "out_of_steps"
    ]
    tasks: dict[str, PregelExecutableTask]
    to_interrupt: list[PregelExecutableTask]
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
        interrupt_after: Union[All, Sequence[str]] = EMPTY_SEQ,
        interrupt_before: Union[All, Sequence[str]] = EMPTY_SEQ,
        manager: Union[None, AsyncParentRunManager, ParentRunManager] = None,
        input_model: Optional[Type[BaseModel]] = None,
        debug: bool = False,
    ) -> None:
        super().__init__(
            step=0,
            stop=0,
            config=config,
            stream=stream,
            store=store,
        )
        self.input = input
        self.input_model = input_model
        self.checkpointer = checkpointer
        self.nodes = nodes
        self.specs = specs
        self.output_keys = output_keys
        self.stream_keys = stream_keys
        self.interrupt_after = interrupt_after
        self.interrupt_before = interrupt_before
        self.manager = manager
        self.is_nested = CONFIG_KEY_TASK_ID in self.config.get(CONF, {})
        self.skip_done_tasks = (
            CONFIG_KEY_CHECKPOINT_ID not in config[CONF]
            or CONFIG_KEY_DEDUPE_TASKS in config[CONF]
        )
        self.debug = debug
        if self.stream is not None and CONFIG_KEY_STREAM in config[CONF]:
            self.stream = DuplexStream(self.stream, config[CONF][CONFIG_KEY_STREAM])
        scratchpad: Optional[PregelScratchpad] = config[CONF].get(CONFIG_KEY_SCRATCHPAD)
        if not self.config[CONF].get(CONFIG_KEY_DELEGATE) and isinstance(
            scratchpad, PregelScratchpad
        ):
            # if count is > 0, append to checkpoint_ns
            # if count is 0, leave as is
            if cnt := scratchpad.subgraph_counter():
                self.config = patch_configurable(
                    self.config,
                    {
                        CONFIG_KEY_CHECKPOINT_NS: NS_SEP.join(
                            (
                                config[CONF][CONFIG_KEY_CHECKPOINT_NS],
                                str(cnt),
                            )
                        )
                    },
                )
        if not self.is_nested and config[CONF].get(CONFIG_KEY_CHECKPOINT_NS):
            self.config = patch_configurable(
                self.config,
                {CONFIG_KEY_CHECKPOINT_NS: "", CONFIG_KEY_CHECKPOINT_ID: None},
            )
        if (
            CONFIG_KEY_CHECKPOINT_MAP in self.config[CONF]
            and self.config[CONF].get(CONFIG_KEY_CHECKPOINT_NS)
            in self.config[CONF][CONFIG_KEY_CHECKPOINT_MAP]
        ):
            self.checkpoint_config = patch_configurable(
                self.config,
                {
                    CONFIG_KEY_CHECKPOINT_ID: config[CONF][CONFIG_KEY_CHECKPOINT_MAP][
                        self.config[CONF][CONFIG_KEY_CHECKPOINT_NS]
                    ]
                },
            )
        else:
            self.checkpoint_config = config
        self.checkpoint_ns = (
            tuple(cast(str, self.config[CONF][CONFIG_KEY_CHECKPOINT_NS]).split(NS_SEP))
            if self.config[CONF].get(CONFIG_KEY_CHECKPOINT_NS)
            else ()
        )
        self.prev_checkpoint_config = None

    def put_writes(self, task_id: str, writes: Sequence[tuple[str, Any]]) -> None:
        """Put writes for a task, to be read by the next tick."""
        if not writes:
            return
        # deduplicate writes to special channels, last write wins
        if all(w[0] in WRITES_IDX_MAP for w in writes):
            writes = list({w[0]: w for w in writes}.values())
        # save writes
        for c, v in writes:
            if (
                c in WRITES_IDX_MAP
                and (
                    idx := next(
                        (
                            i
                            for i, w in enumerate(self.checkpoint_pending_writes)
                            if w[0] == task_id and w[1] == c
                        ),
                        None,
                    )
                )
                is not None
            ):
                self.checkpoint_pending_writes[idx] = (task_id, c, v)
            else:
                self.checkpoint_pending_writes.append((task_id, c, v))
        if self.checkpointer_put_writes is not None:
            config = patch_configurable(
                self.checkpoint_config,
                {
                    CONFIG_KEY_CHECKPOINT_NS: self.config[CONF].get(
                        CONFIG_KEY_CHECKPOINT_NS, ""
                    ),
                    CONFIG_KEY_CHECKPOINT_ID: self.checkpoint["id"],
                },
            )
            if self.checkpointer_put_writes_accepts_task_path:
                if hasattr(self, "tasks"):
                    task = self.tasks.get(task_id)
                else:
                    task = None
                self.submit(
                    self.checkpointer_put_writes,
                    config,
                    writes,
                    task_id,
                    task_path_str(task.path) if task else "",
                )
            else:
                self.submit(
                    self.checkpointer_put_writes,
                    config,
                    writes,
                    task_id,
                )
        # output writes
        if hasattr(self, "tasks"):
            self._output_writes(task_id, writes)

    def accept_push(
        self, task: PregelExecutableTask, write_idx: int, call: Optional[Call] = None
    ) -> Optional[PregelExecutableTask]:
        """Accept a PUSH from a task, potentially returning a new task to start."""
        # don't start if we should interrupt *after* the original task
        if self.interrupt_after and should_interrupt(
            self.checkpoint, self.interrupt_after, [task]
        ):
            self.to_interrupt.append(task)
            return
        if pushed := cast(
            Optional[PregelExecutableTask],
            prepare_single_task(
                (PUSH, task.path, write_idx, task.id, call),
                None,
                checkpoint=self.checkpoint,
                pending_writes=self.checkpoint_pending_writes,
                processes=self.nodes,
                channels=self.channels,
                managed=self.managed,
                config=task.config,
                step=self.step,
                for_execution=True,
                store=self.store,
                checkpointer=self.checkpointer,
                manager=self.manager,
            ),
        ):
            # don't start if we should interrupt *before* the new task
            if self.interrupt_before and should_interrupt(
                self.checkpoint, self.interrupt_before, [pushed]
            ):
                self.to_interrupt.append(pushed)
                return
            # produce debug output
            self._emit("debug", map_debug_tasks, self.step, [pushed])
            # debug flag
            if self.debug:
                print_step_tasks(self.step, [pushed])
            # save the new task
            self.tasks[pushed.id] = pushed
            # match any pending writes to the new task
            if self.skip_done_tasks:
                self._match_writes({pushed.id: pushed})
            # return the new task, to be started if not run before
            return pushed

    def tick(
        self,
        *,
        input_keys: Union[str, Sequence[str]],
    ) -> bool:
        """Execute a single iteration of the Pregel loop.

        Args:
            input_keys: The key(s) to read input from.

        Returns:
            True if more iterations are needed.
        """
        if self.status != "pending":
            raise RuntimeError("Cannot tick when status is no longer 'pending'")

        if self.input not in (INPUT_DONE, INPUT_RESUMING, INPUT_SHOULD_VALIDATE):
            self._first(input_keys=input_keys)
        elif self.to_interrupt:
            # if we need to interrupt, do so
            self.status = "interrupt_before"
            raise GraphInterrupt()
        elif all(task.writes for task in self.tasks.values()):
            # finish superstep
            writes = [w for t in self.tasks.values() for w in t.writes]
            # debug flag
            if self.debug:
                print_step_writes(
                    self.step,
                    writes,
                    (
                        [self.stream_keys]
                        if isinstance(self.stream_keys, str)
                        else self.stream_keys
                    ),
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
            # validate input if requested
            if self.input is INPUT_SHOULD_VALIDATE:
                self.input = INPUT_DONE
                # validate
                cast(Type[BaseModel], self.input_model)(
                    **read_channels(self.channels, self.stream_keys)
                )
            # produce values output
            self._emit(
                "values", map_output_values, self.output_keys, writes, self.channels
            )
            # clear pending writes
            self.checkpoint_pending_writes.clear()
            # "not skip_done_tasks" only applies to first tick after resuming
            self.skip_done_tasks = True
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
            if self.interrupt_after and should_interrupt(
                self.checkpoint, self.interrupt_after, self.tasks.values()
            ):
                self.status = "interrupt_after"
                raise GraphInterrupt()

            # unset resuming flag
            self.config[CONF].pop(CONFIG_KEY_RESUMING, None)
        else:
            return False

        # check if iteration limit is reached
        if self.step > self.stop:
            self.status = "out_of_steps"
            return False

        # prepare next tasks
        self.tasks = prepare_next_tasks(
            self.checkpoint,
            self.checkpoint_pending_writes,
            self.nodes,
            self.channels,
            self.managed,
            self.config,
            self.step,
            for_execution=True,
            manager=self.manager,
            store=self.store,
            checkpointer=self.checkpointer,
        )
        self.to_interrupt = []

        # produce debug output
        if self._checkpointer_put_after_previous is not None:
            self._emit(
                "debug",
                map_debug_checkpoint,
                self.step - 1,  # printing checkpoint for previous step
                self.checkpoint_config,
                self.channels,
                self.stream_keys,
                self.checkpoint_metadata,
                self.checkpoint,
                self.tasks.values(),
                self.checkpoint_pending_writes,
                self.prev_checkpoint_config,
                self.output_keys,
            )

        # if no more tasks, we're done
        if not self.tasks:
            self.status = "done"
            return False

        # check if we should delegate (used by subgraphs in distributed mode)
        if self.config[CONF].get(CONFIG_KEY_DELEGATE):
            assert self.input is INPUT_RESUMING
            raise GraphDelegate(
                {
                    "config": patch_configurable(
                        self.config, {CONFIG_KEY_DELEGATE: False}
                    ),
                    "input": None,
                }
            )

        # if there are pending writes from a previous loop, apply them
        if self.skip_done_tasks and self.checkpoint_pending_writes:
            self._match_writes(self.tasks)

        # if all tasks have finished, re-tick
        if all(task.writes for task in self.tasks.values()):
            return self.tick(input_keys=input_keys)

        # before execution, check if we should interrupt
        if self.interrupt_before and should_interrupt(
            self.checkpoint, self.interrupt_before, self.tasks.values()
        ):
            self.status = "interrupt_before"
            raise GraphInterrupt()

        # produce debug output
        self._emit("debug", map_debug_tasks, self.step, self.tasks.values())

        # debug flag
        if self.debug:
            print_step_tasks(self.step, list(self.tasks.values()))

        # print output for any tasks we applied previous writes to
        for task in self.tasks.values():
            if task.writes:
                self._output_writes(task.id, task.writes, cached=True)

        return True

    # private

    def _match_writes(self, tasks: Mapping[str, PregelExecutableTask]) -> None:
        for tid, k, v in self.checkpoint_pending_writes:
            if k in (ERROR, INTERRUPT, RESUME):
                continue
            if task := tasks.get(tid):
                if k == SCHEDULED:
                    if v == max(
                        self.checkpoint["versions_seen"].get(INTERRUPT, {}).values(),
                        default=None,
                    ):
                        self.tasks[tid] = task._replace(scheduled=True)
                else:
                    task.writes.append((k, v))

    def _first(self, *, input_keys: Union[str, Sequence[str]]) -> None:
        # resuming from previous checkpoint requires
        # - finding a previous checkpoint
        # - receiving None input (outer graph) or RESUMING flag (subgraph)
        configurable = self.config.get(CONF, {})
        is_resuming = bool(self.checkpoint["channel_versions"]) and bool(
            configurable.get(
                CONFIG_KEY_RESUMING,
                self.input is None
                or isinstance(self.input, Command)
                or (
                    not self.is_nested
                    and self.config.get("metadata", {}).get("run_id")
                    == self.checkpoint_metadata.get("run_id", MISSING)
                ),
            )
        )

        # map command to writes
        if isinstance(self.input, Command):
            if self.input.resume is not None and not self.checkpointer:
                raise RuntimeError(
                    "Cannot use Command(resume=...) without checkpointer"
                )
            writes: defaultdict[str, list[tuple[str, Any]]] = defaultdict(list)
            # group writes by task ID
            for tid, c, v in map_command(self.input, self.checkpoint_pending_writes):
                writes[tid].append((c, v))
            if not writes:
                raise EmptyInputError("Received empty Command input")
            # save writes
            for tid, ws in writes.items():
                self.put_writes(tid, ws)
        # apply NULL writes
        if null_writes := [
            w[1:] for w in self.checkpoint_pending_writes if w[0] == NULL_TASK_ID
        ]:
            mv_writes = apply_writes(
                self.checkpoint,
                self.channels,
                [PregelTaskWrites((), INPUT, null_writes, [])],
                self.checkpointer_get_next_version,
            )
            for key, values in mv_writes.items():
                self._update_mv(key, values)
        # proceed past previous checkpoint
        if is_resuming:
            self.checkpoint["versions_seen"].setdefault(INTERRUPT, {})
            for k in self.channels:
                if k in self.checkpoint["channel_versions"]:
                    version = self.checkpoint["channel_versions"][k]
                    self.checkpoint["versions_seen"][INTERRUPT][k] = version
            # produce values output
            self._emit(
                "values", map_output_values, self.output_keys, True, self.channels
            )
            # set flag
            self.input = INPUT_RESUMING
        # map inputs to channel updates
        elif input_writes := deque(map_input(input_keys, self.input)):
            # TODO shouldn't these writes be passed to put_writes too?
            # check if we should delegate (used by subgraphs in distributed mode)
            if self.config[CONF].get(CONFIG_KEY_DELEGATE):
                raise GraphDelegate(
                    {
                        "config": patch_configurable(
                            self.config, {CONFIG_KEY_DELEGATE: False}
                        ),
                        "input": self.input,
                    }
                )
            # discard any unfinished tasks from previous checkpoint
            discard_tasks = prepare_next_tasks(
                self.checkpoint,
                self.checkpoint_pending_writes,
                self.nodes,
                self.channels,
                self.managed,
                self.config,
                self.step,
                for_execution=True,
                store=None,
                checkpointer=None,
                manager=None,
            )
            # apply input writes
            mv_writes = apply_writes(
                self.checkpoint,
                self.channels,
                [
                    *discard_tasks.values(),
                    PregelTaskWrites((), INPUT, input_writes, []),
                ],
                self.checkpointer_get_next_version,
            )
            assert not mv_writes, "Can't write to SharedValues in graph input"
            # save input checkpoint
            self._put_checkpoint({"source": "input", "writes": dict(input_writes)})
            # set flag
            if (
                self.input_model is not None
                and not isinstance(self.input, self.input_model)
                and not isinstance(self.stream_keys, str)
            ):
                self.input = INPUT_SHOULD_VALIDATE
            else:
                self.input = INPUT_DONE
        elif CONFIG_KEY_RESUMING not in configurable:
            raise EmptyInputError(f"Received no input for {input_keys}")
        else:
            self.input = INPUT_DONE
        # update config
        if not self.is_nested:
            self.config = patch_configurable(
                self.config, {CONFIG_KEY_RESUMING: is_resuming}
            )

    def _put_checkpoint(self, metadata: CheckpointMetadata) -> None:
        for k, v in self.config["metadata"].items():
            metadata.setdefault(k, v)  # type: ignore
        # assign step and parents
        metadata["step"] = self.step
        metadata["parents"] = self.config[CONF].get(CONFIG_KEY_CHECKPOINT_MAP, {})
        # debug flag
        if self.debug:
            print_step_checkpoint(
                metadata,
                self.channels,
                (
                    [self.stream_keys]
                    if isinstance(self.stream_keys, str)
                    else self.stream_keys
                ),
            )
        # create new checkpoint
        self.checkpoint = create_checkpoint(self.checkpoint, self.channels, self.step)
        # bail if no checkpointer
        if self._checkpointer_put_after_previous is not None:
            self.checkpoint_metadata = metadata

            self.prev_checkpoint_config = (
                self.checkpoint_config
                if CONFIG_KEY_CHECKPOINT_ID in self.checkpoint_config[CONF]
                and self.checkpoint_config[CONF][CONFIG_KEY_CHECKPOINT_ID]
                else None
            )
            self.checkpoint_config = {
                **self.checkpoint_config,
                CONF: {
                    **self.checkpoint_config[CONF],
                    CONFIG_KEY_CHECKPOINT_NS: self.config[CONF].get(
                        CONFIG_KEY_CHECKPOINT_NS, ""
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
                CONF: {
                    **self.checkpoint_config[CONF],
                    CONFIG_KEY_CHECKPOINT_ID: self.checkpoint["id"],
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
        # suppress interrupt
        suppress = isinstance(exc_value, GraphInterrupt) and not self.is_nested
        if suppress:
            # emit one last "values" event, with pending writes applied
            if (
                hasattr(self, "tasks")
                and self.checkpoint_pending_writes
                and any(task.writes for task in self.tasks.values())
            ):
                mv_writes = apply_writes(
                    self.checkpoint,
                    self.channels,
                    self.tasks.values(),
                    self.checkpointer_get_next_version,
                )
                for key, values in mv_writes.items():
                    self._update_mv(key, values)
                self._emit(
                    "values",
                    map_output_values,
                    self.output_keys,
                    [w for t in self.tasks.values() for w in t.writes],
                    self.channels,
                )
            # emit INTERRUPT if exception is empty (otherwise emitted by put_writes)
            if exc_value is not None and (not exc_value.args or not exc_value.args[0]):
                self._emit(
                    "updates",
                    lambda: iter(
                        [{INTERRUPT: cast(GraphInterrupt, exc_value).args[0]}]
                    ),
                )
            # save final output
            self.output = read_channels(self.channels, self.output_keys)
            # suppress interrupt
            return True
        elif exc_type is None:
            # save final output
            self.output = read_channels(self.channels, self.output_keys)

    def _emit(
        self,
        mode: str,
        values: Callable[P, Iterator[Any]],
        *args: P.args,
        **kwargs: P.kwargs,
    ) -> None:
        if self.stream is None:
            return
        if mode not in self.stream.modes:
            return
        for v in values(*args, **kwargs):
            self.stream((self.checkpoint_ns, mode, v))

    def _output_writes(
        self, task_id: str, writes: Sequence[tuple[str, Any]], *, cached: bool = False
    ) -> None:
        if task := self.tasks.get(task_id):
            if task.config is not None and TAG_HIDDEN in task.config.get(
                "tags", EMPTY_SEQ
            ):
                return
            if writes[0][0] == INTERRUPT:
                self._emit(
                    "updates",
                    lambda: iter(
                        [
                            {
                                INTERRUPT: tuple(
                                    v
                                    for w in writes
                                    if w[0] == INTERRUPT
                                    for v in (
                                        w[1] if isinstance(w[1], Sequence) else (w[1],)
                                    )
                                )
                            }
                        ]
                    ),
                )
            elif writes[0][0] != ERROR:
                self._emit(
                    "updates",
                    map_output_updates,
                    self.output_keys,
                    [(task, writes)],
                    cached,
                )
            if not cached:
                self._emit(
                    "debug",
                    map_debug_task_results,
                    self.step,
                    (task, writes),
                    self.stream_keys,
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
        manager: Union[None, AsyncParentRunManager, ParentRunManager] = None,
        interrupt_after: Union[All, Sequence[str]] = EMPTY_SEQ,
        interrupt_before: Union[All, Sequence[str]] = EMPTY_SEQ,
        output_keys: Union[str, Sequence[str]] = EMPTY_SEQ,
        stream_keys: Union[str, Sequence[str]] = EMPTY_SEQ,
        input_model: Optional[Type[BaseModel]] = None,
        debug: bool = False,
    ) -> None:
        super().__init__(
            input,
            input_model=input_model,
            stream=stream,
            config=config,
            checkpointer=checkpointer,
            store=store,
            nodes=nodes,
            specs=specs,
            output_keys=output_keys,
            stream_keys=stream_keys,
            interrupt_after=interrupt_after,
            interrupt_before=interrupt_before,
            manager=manager,
            debug=debug,
        )
        self.stack = ExitStack()
        if checkpointer:
            self.checkpointer_get_next_version = checkpointer.get_next_version
            self.checkpointer_put_writes = checkpointer.put_writes
            self.checkpointer_put_writes_accepts_task_path = (
                signature(checkpointer.put_writes).parameters.get("task_path")
                is not None
            )
        else:
            self.checkpointer_get_next_version = increment
            self._checkpointer_put_after_previous = None  # type: ignore[assignment]
            self.checkpointer_put_writes = None
            self.checkpointer_put_writes_accepts_task_path = False

    def _checkpointer_put_after_previous(
        self,
        prev: Optional[concurrent.futures.Future],
        config: RunnableConfig,
        checkpoint: Checkpoint,
        metadata: CheckpointMetadata,
        new_versions: ChannelVersions,
    ) -> RunnableConfig:
        try:
            if prev is not None:
                prev.result()
        finally:
            cast(BaseCheckpointSaver, self.checkpointer).put(
                config, checkpoint, metadata, new_versions
            )

    def _update_mv(self, key: str, values: Sequence[Any]) -> None:
        managed_value = self.managed.get(key)
        if managed_value is None:
            return

        return self.submit(cast(WritableManagedValue, managed_value).update, values)

    # context manager

    def __enter__(self) -> Self:
        if self.config.get(CONF, {}).get(
            CONFIG_KEY_ENSURE_LATEST
        ) and self.checkpoint_config[CONF].get(CONFIG_KEY_CHECKPOINT_ID):
            if self.checkpointer is None:
                raise RuntimeError(
                    "Cannot ensure latest checkpoint without checkpointer"
                )
            saved = self.checkpointer.get_tuple(
                patch_configurable(
                    self.checkpoint_config, {CONFIG_KEY_CHECKPOINT_ID: None}
                )
            )
            if (
                saved is None
                or saved.checkpoint["id"]
                != self.checkpoint_config[CONF][CONFIG_KEY_CHECKPOINT_ID]
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
            CONF: {
                CONFIG_KEY_CHECKPOINT_NS: "",
                **self.config.get(CONF, {}),
                **saved.config.get(CONF, {}),
            },
        }
        self.prev_checkpoint_config = saved.parent_config
        self.checkpoint = saved.checkpoint
        self.checkpoint_metadata = saved.metadata
        self.checkpoint_pending_writes = (
            [(str(tid), k, v) for tid, k, v in saved.pending_writes]
            if saved.pending_writes is not None
            else []
        )

        self.submit = self.stack.enter_context(BackgroundExecutor(self.config))
        self.channels, self.managed = self.stack.enter_context(
            ChannelsManager(self.specs, self.checkpoint, self)
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
        interrupt_after: Union[All, Sequence[str]] = EMPTY_SEQ,
        interrupt_before: Union[All, Sequence[str]] = EMPTY_SEQ,
        manager: Union[None, AsyncParentRunManager, ParentRunManager] = None,
        output_keys: Union[str, Sequence[str]] = EMPTY_SEQ,
        stream_keys: Union[str, Sequence[str]] = EMPTY_SEQ,
        input_model: Optional[Type[BaseModel]] = None,
        debug: bool = False,
    ) -> None:
        super().__init__(
            input,
            input_model=input_model,
            stream=stream,
            config=config,
            checkpointer=checkpointer,
            store=store,
            nodes=nodes,
            specs=specs,
            output_keys=output_keys,
            stream_keys=stream_keys,
            interrupt_after=interrupt_after,
            interrupt_before=interrupt_before,
            manager=manager,
            debug=debug,
        )
        self.stack = AsyncExitStack()
        if checkpointer:
            self.checkpointer_get_next_version = checkpointer.get_next_version
            self.checkpointer_put_writes = checkpointer.aput_writes
            self.checkpointer_put_writes_accepts_task_path = (
                signature(checkpointer.aput_writes).parameters.get("task_path")
                is not None
            )
        else:
            self.checkpointer_get_next_version = increment
            self._checkpointer_put_after_previous = None  # type: ignore[assignment]
            self.checkpointer_put_writes = None
            self.checkpointer_put_writes_accepts_task_path = False

    async def _checkpointer_put_after_previous(
        self,
        prev: Optional[asyncio.Task],
        config: RunnableConfig,
        checkpoint: Checkpoint,
        metadata: CheckpointMetadata,
        new_versions: ChannelVersions,
    ) -> RunnableConfig:
        try:
            if prev is not None:
                await prev
        finally:
            await cast(BaseCheckpointSaver, self.checkpointer).aput(
                config, checkpoint, metadata, new_versions
            )

    def _update_mv(self, key: str, values: Sequence[Any]) -> None:
        managed_value = self.managed.get(key)
        if managed_value is None:
            return

        return self.submit(cast(WritableManagedValue, managed_value).aupdate, values)

    # context manager

    async def __aenter__(self) -> Self:
        if self.config.get(CONF, {}).get(
            CONFIG_KEY_ENSURE_LATEST
        ) and self.checkpoint_config[CONF].get(CONFIG_KEY_CHECKPOINT_ID):
            if self.checkpointer is None:
                raise RuntimeError(
                    "Cannot ensure latest checkpoint without checkpointer"
                )
            saved = await self.checkpointer.aget_tuple(
                patch_configurable(
                    self.checkpoint_config, {CONFIG_KEY_CHECKPOINT_ID: None}
                )
            )
            if (
                saved is None
                or saved.checkpoint["id"]
                != self.checkpoint_config[CONF][CONFIG_KEY_CHECKPOINT_ID]
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
            CONF: {
                CONFIG_KEY_CHECKPOINT_NS: "",
                **self.config.get(CONF, {}),
                **saved.config.get(CONF, {}),
            },
        }
        self.prev_checkpoint_config = saved.parent_config
        self.checkpoint = saved.checkpoint
        self.checkpoint_metadata = saved.metadata
        self.checkpoint_pending_writes = (
            [(str(tid), k, v) for tid, k, v in saved.pending_writes]
            if saved.pending_writes is not None
            else []
        )

        self.submit = await self.stack.enter_async_context(
            AsyncBackgroundExecutor(self.config)
        )
        self.channels, self.managed = await self.stack.enter_async_context(
            AsyncChannelsManager(self.specs, self.checkpoint, self)
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
        exit_task = asyncio.create_task(
            self.stack.__aexit__(exc_type, exc_value, traceback)
        )
        try:
            return await exit_task
        except asyncio.CancelledError as e:
            # Bubble up the exit task upon cancellation to permit the API
            # consumer to await it before e.g., reusing the DB connection.
            e.args = (*e.args, exit_task)
            raise
