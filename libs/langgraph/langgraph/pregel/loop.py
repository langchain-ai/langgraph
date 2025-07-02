from __future__ import annotations

import asyncio
import binascii
import concurrent.futures
from collections import defaultdict, deque
from collections.abc import Iterator, Mapping, Sequence
from contextlib import (
    AbstractAsyncContextManager,
    AbstractContextManager,
    AsyncExitStack,
    ExitStack,
)
from datetime import datetime, timezone
from inspect import signature
from types import TracebackType
from typing import (
    Any,
    Callable,
    Literal,
    Optional,
    TypeVar,
    cast,
)

from langchain_core.callbacks import AsyncParentRunManager, ParentRunManager
from langchain_core.runnables import RunnableConfig
from typing_extensions import ParamSpec, Self

from langgraph.cache.base import BaseCache
from langgraph.channels.base import BaseChannel
from langgraph.checkpoint.base import (
    WRITES_IDX_MAP,
    BaseCheckpointSaver,
    ChannelVersions,
    Checkpoint,
    CheckpointMetadata,
    CheckpointTuple,
    PendingWrite,
)
from langgraph.constants import (
    CONF,
    CONFIG_KEY_CHECKPOINT_ID,
    CONFIG_KEY_CHECKPOINT_MAP,
    CONFIG_KEY_CHECKPOINT_NS,
    CONFIG_KEY_RESUME_MAP,
    CONFIG_KEY_RESUMING,
    CONFIG_KEY_SCRATCHPAD,
    CONFIG_KEY_STREAM,
    CONFIG_KEY_TASK_ID,
    CONFIG_KEY_THREAD_ID,
    EMPTY_SEQ,
    ERROR,
    INPUT,
    INTERRUPT,
    MISSING,
    NS_END,
    NS_SEP,
    NULL_TASK_ID,
    PUSH,
    RESUME,
    TAG_HIDDEN,
)
from langgraph.errors import (
    EmptyInputError,
    GraphInterrupt,
)
from langgraph.managed.base import (
    ManagedValueMapping,
    ManagedValueSpec,
)
from langgraph.pregel.algo import (
    Call,
    GetNextVersion,
    PregelTaskWrites,
    apply_writes,
    checkpoint_null_version,
    increment,
    prepare_next_tasks,
    prepare_single_task,
    should_interrupt,
    task_path_str,
)
from langgraph.pregel.checkpoint import (
    channels_from_checkpoint,
    copy_checkpoint,
    create_checkpoint,
    empty_checkpoint,
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
    map_command,
    map_input,
    map_output_updates,
    map_output_values,
    read_channels,
)
from langgraph.pregel.read import PregelNode
from langgraph.pregel.utils import get_new_channel_versions, is_xxh3_128_hexdigest
from langgraph.store.base import BaseStore
from langgraph.types import (
    All,
    CachePolicy,
    Command,
    PregelExecutableTask,
    PregelScratchpad,
    RetryPolicy,
    StreamChunk,
    StreamMode,
    StreamProtocol,
)
from langgraph.utils.config import patch_configurable

V = TypeVar("V")
P = ParamSpec("P")


WritesT = Sequence[tuple[str, Any]]


def DuplexStream(*streams: StreamProtocol) -> StreamProtocol:
    def __call__(value: StreamChunk) -> None:
        for stream in streams:
            if value[1] in stream.modes:
                stream(value)

    return StreamProtocol(__call__, {mode for s in streams for mode in s.modes})


class PregelLoop:
    config: RunnableConfig
    store: BaseStore | None
    stream: StreamProtocol | None
    step: int
    stop: int

    input: Any | None
    cache: BaseCache[WritesT] | None
    checkpointer: BaseCheckpointSaver | None
    nodes: Mapping[str, PregelNode]
    specs: Mapping[str, BaseChannel | ManagedValueSpec]
    input_keys: str | Sequence[str]
    output_keys: str | Sequence[str]
    stream_keys: str | Sequence[str]
    skip_done_tasks: bool
    is_nested: bool
    manager: None | AsyncParentRunManager | ParentRunManager
    interrupt_after: All | Sequence[str]
    interrupt_before: All | Sequence[str]
    checkpoint_during: bool
    retry_policy: Sequence[RetryPolicy]
    cache_policy: CachePolicy | None

    checkpointer_get_next_version: GetNextVersion
    checkpointer_put_writes: Callable[[RunnableConfig, WritesT, str], Any] | None
    checkpointer_put_writes_accepts_task_path: bool
    _checkpointer_put_after_previous: (
        Callable[
            [
                concurrent.futures.Future | None,
                RunnableConfig,
                Checkpoint,
                str,
                ChannelVersions,
            ],
            Any,
        ]
        | None
    )
    _migrate_checkpoint: Callable[[Checkpoint], None] | None
    submit: Submit
    channels: Mapping[str, BaseChannel]
    managed: ManagedValueMapping
    checkpoint: Checkpoint
    checkpoint_id_saved: str
    checkpoint_ns: tuple[str, ...]
    checkpoint_config: RunnableConfig
    checkpoint_metadata: CheckpointMetadata
    checkpoint_pending_writes: list[PendingWrite]
    checkpoint_previous_versions: dict[str, str | float | int]
    prev_checkpoint_config: RunnableConfig | None

    status: Literal[
        "input",
        "pending",
        "done",
        "interrupt_before",
        "interrupt_after",
        "out_of_steps",
    ]
    tasks: dict[str, PregelExecutableTask]
    output: None | dict[str, Any] | Any = None
    updated_channels: set[str] | None = None

    # public

    def __init__(
        self,
        input: Any | None,
        *,
        stream: StreamProtocol | None,
        config: RunnableConfig,
        store: BaseStore | None,
        cache: BaseCache | None,
        checkpointer: BaseCheckpointSaver | None,
        nodes: Mapping[str, PregelNode],
        specs: Mapping[str, BaseChannel | ManagedValueSpec],
        input_keys: str | Sequence[str],
        output_keys: str | Sequence[str],
        stream_keys: str | Sequence[str],
        trigger_to_nodes: Mapping[str, Sequence[str]],
        interrupt_after: All | Sequence[str] = EMPTY_SEQ,
        interrupt_before: All | Sequence[str] = EMPTY_SEQ,
        manager: None | AsyncParentRunManager | ParentRunManager = None,
        migrate_checkpoint: Callable[[Checkpoint], None] | None = None,
        retry_policy: Sequence[RetryPolicy] = (),
        cache_policy: CachePolicy | None = None,
        checkpoint_during: bool = True,
    ) -> None:
        self.stream = stream
        self.config = config
        self.store = store
        self.step = 0
        self.stop = 0
        self.input = input
        self.checkpointer = checkpointer
        self.cache = cache
        self.nodes = nodes
        self.specs = specs
        self.input_keys = input_keys
        self.output_keys = output_keys
        self.stream_keys = stream_keys
        self.interrupt_after = interrupt_after
        self.interrupt_before = interrupt_before
        self.manager = manager
        self.is_nested = CONFIG_KEY_TASK_ID in self.config.get(CONF, {})
        self.skip_done_tasks = CONFIG_KEY_CHECKPOINT_ID not in config[CONF]
        self._migrate_checkpoint = migrate_checkpoint
        self.trigger_to_nodes = trigger_to_nodes
        self.retry_policy = retry_policy
        self.cache_policy = cache_policy
        self.checkpoint_during = checkpoint_during
        if self.stream is not None and CONFIG_KEY_STREAM in config[CONF]:
            self.stream = DuplexStream(self.stream, config[CONF][CONFIG_KEY_STREAM])
        scratchpad: PregelScratchpad | None = config[CONF].get(CONFIG_KEY_SCRATCHPAD)
        if isinstance(scratchpad, PregelScratchpad):
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
                    CONFIG_KEY_CHECKPOINT_ID: self.config[CONF][
                        CONFIG_KEY_CHECKPOINT_MAP
                    ][self.config[CONF][CONFIG_KEY_CHECKPOINT_NS]]
                },
            )
        else:
            self.checkpoint_config = self.config
        if thread_id := self.checkpoint_config[CONF].get(CONFIG_KEY_THREAD_ID):
            if not isinstance(thread_id, str):
                self.checkpoint_config = patch_configurable(
                    self.checkpoint_config,
                    {CONFIG_KEY_THREAD_ID: str(thread_id)},
                )
        self.checkpoint_ns = (
            tuple(cast(str, self.config[CONF][CONFIG_KEY_CHECKPOINT_NS]).split(NS_SEP))
            if self.config[CONF].get(CONFIG_KEY_CHECKPOINT_NS)
            else ()
        )
        self.prev_checkpoint_config = None

    def put_writes(self, task_id: str, writes: WritesT) -> None:
        """Put writes for a task, to be read by the next tick."""
        if not writes:
            return
        # deduplicate writes to special channels, last write wins
        if all(w[0] in WRITES_IDX_MAP for w in writes):
            writes = list({w[0]: w for w in writes}.values())
        if task_id == NULL_TASK_ID:
            # writes for the null task are accumulated
            self.checkpoint_pending_writes = [
                w
                for w in self.checkpoint_pending_writes
                if w[0] != task_id or w[1] not in WRITES_IDX_MAP
            ]
            writes_to_save: WritesT = [
                w[1:] for w in self.checkpoint_pending_writes if w[0] == task_id
            ] + list(writes)
        else:
            # remove existing writes for this task
            self.checkpoint_pending_writes = [
                w for w in self.checkpoint_pending_writes if w[0] != task_id
            ]
            writes_to_save = writes
        # save writes
        self.checkpoint_pending_writes.extend((task_id, c, v) for c, v in writes)
        if self.checkpoint_during and self.checkpointer_put_writes is not None:
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
                    writes_to_save,
                    task_id,
                    task_path_str(task.path) if task else "",
                )
            else:
                self.submit(
                    self.checkpointer_put_writes,
                    config,
                    writes_to_save,
                    task_id,
                )
        # output writes
        if hasattr(self, "tasks"):
            self.output_writes(task_id, writes)

    def _put_pending_writes(self) -> None:
        if self.checkpointer_put_writes is None:
            return
        if not self.checkpoint_pending_writes:
            return
        # patch config
        config = patch_configurable(
            self.checkpoint_config,
            {
                CONFIG_KEY_CHECKPOINT_NS: self.config[CONF].get(
                    CONFIG_KEY_CHECKPOINT_NS, ""
                ),
                CONFIG_KEY_CHECKPOINT_ID: self.checkpoint["id"],
            },
        )
        # group by task id
        by_task = defaultdict(list)
        for task_id, channel, value in self.checkpoint_pending_writes:
            by_task[task_id].append((channel, value))
        # submit writes to checkpointer
        for task_id, writes in by_task.items():
            if self.checkpointer_put_writes_accepts_task_path and hasattr(
                self, "tasks"
            ):
                task = self.tasks.get(task_id)
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

    def accept_push(
        self, task: PregelExecutableTask, write_idx: int, call: Call | None = None
    ) -> PregelExecutableTask | None:
        """Accept a PUSH from a task, potentially returning a new task to start."""
        checkpoint_id_bytes = binascii.unhexlify(self.checkpoint["id"].replace("-", ""))
        null_version = checkpoint_null_version(self.checkpoint)
        if pushed := cast(
            Optional[PregelExecutableTask],
            prepare_single_task(
                (PUSH, task.path, write_idx, task.id, call),
                None,
                checkpoint=self.checkpoint,
                checkpoint_id_bytes=checkpoint_id_bytes,
                checkpoint_null_version=null_version,
                pending_writes=self.checkpoint_pending_writes,
                processes=self.nodes,
                channels=self.channels,
                managed=self.managed,
                config=task.config,
                step=self.step,
                stop=self.stop,
                for_execution=True,
                store=self.store,
                checkpointer=self.checkpointer,
                manager=self.manager,
                retry_policy=self.retry_policy,
                cache_policy=self.cache_policy,
            ),
        ):
            # produce debug output
            self._emit("tasks", map_debug_tasks, [pushed])
            # save the new task
            self.tasks[pushed.id] = pushed
            # match any pending writes to the new task
            if self.skip_done_tasks:
                self._match_writes({pushed.id: pushed})
            # return the new task, to be started if not run before
            return pushed

    def tick(self) -> bool:
        """Execute a single iteration of the Pregel loop.

        Args:
            input_keys: The key(s) to read input from.

        Returns:
            True if more iterations are needed.
        """

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
            self.stop,
            for_execution=True,
            manager=self.manager,
            store=self.store,
            checkpointer=self.checkpointer,
            trigger_to_nodes=self.trigger_to_nodes,
            updated_channels=self.updated_channels,
            retry_policy=self.retry_policy,
            cache_policy=self.cache_policy,
        )

        # produce debug output
        if self._checkpointer_put_after_previous is not None:
            self._emit(
                "checkpoints",
                map_debug_checkpoint,
                {
                    **self.checkpoint_config,
                    CONF: {
                        **self.checkpoint_config[CONF],
                        CONFIG_KEY_CHECKPOINT_ID: self.checkpoint["id"],
                    },
                },
                self.channels,
                self.stream_keys,
                self.checkpoint_metadata,
                self.tasks.values(),
                self.checkpoint_pending_writes,
                self.prev_checkpoint_config,
                self.output_keys,
            )

        # if no more tasks, we're done
        if not self.tasks:
            self.status = "done"
            return False

        # if there are pending writes from a previous loop, apply them
        if self.skip_done_tasks and self.checkpoint_pending_writes:
            self._match_writes(self.tasks)

        # before execution, check if we should interrupt
        if self.interrupt_before and should_interrupt(
            self.checkpoint, self.interrupt_before, self.tasks.values()
        ):
            self.status = "interrupt_before"
            raise GraphInterrupt()

        # produce debug output
        self._emit("tasks", map_debug_tasks, self.tasks.values())

        # print output for any tasks we applied previous writes to
        for task in self.tasks.values():
            if task.writes:
                self.output_writes(task.id, task.writes, cached=True)

        return True

    def after_tick(self) -> None:
        # finish superstep
        writes = [w for t in self.tasks.values() for w in t.writes]
        # all tasks have finished
        self.updated_channels = apply_writes(
            self.checkpoint,
            self.channels,
            self.tasks.values(),
            self.checkpointer_get_next_version,
            self.trigger_to_nodes,
        )
        # produce values output
        if not self.updated_channels.isdisjoint(
            (self.output_keys,)
            if isinstance(self.output_keys, str)
            else self.output_keys
        ):
            self._emit(
                "values", map_output_values, self.output_keys, writes, self.channels
            )
        # clear pending writes
        self.checkpoint_pending_writes.clear()
        # "not skip_done_tasks" only applies to first tick after resuming
        self.skip_done_tasks = True
        # save checkpoint
        self._put_checkpoint({"source": "loop"})
        # after execution, check if we should interrupt
        if self.interrupt_after and should_interrupt(
            self.checkpoint, self.interrupt_after, self.tasks.values()
        ):
            self.status = "interrupt_after"
            raise GraphInterrupt()
        # unset resuming flag
        self.config[CONF].pop(CONFIG_KEY_RESUMING, None)

    def match_cached_writes(self) -> Sequence[PregelExecutableTask]:
        raise NotImplementedError

    async def amatch_cached_writes(self) -> Sequence[PregelExecutableTask]:
        raise NotImplementedError

    # private

    def _match_writes(self, tasks: Mapping[str, PregelExecutableTask]) -> None:
        for tid, k, v in self.checkpoint_pending_writes:
            if k in (ERROR, INTERRUPT, RESUME):
                continue
            if task := tasks.get(tid):
                task.writes.append((k, v))

    def _first(self, *, input_keys: str | Sequence[str]) -> set[str] | None:
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
        # this can be set only when there are input_writes
        updated_channels: set[str] | None = None

        # map command to writes
        if isinstance(self.input, Command):
            if resume_is_map := (
                (resume := self.input.resume) is not None
                and isinstance(resume, dict)
                and all(is_xxh3_128_hexdigest(k) for k in resume)
            ):
                self.config[CONF][CONFIG_KEY_RESUME_MAP] = self.input.resume
            if resume is not None and not self.checkpointer:
                raise RuntimeError(
                    "Cannot use Command(resume=...) without checkpointer"
                )
            writes: defaultdict[str, list[tuple[str, Any]]] = defaultdict(list)
            # group writes by task ID
            for tid, c, v in map_command(cmd=self.input):
                if not (c == RESUME and resume_is_map):
                    writes[tid].append((c, v))
            if not writes and not resume_is_map:
                raise EmptyInputError("Received empty Command input")
            # save writes
            for tid, ws in writes.items():
                self.put_writes(tid, ws)
        # apply NULL writes
        if null_writes := [
            w[1:] for w in self.checkpoint_pending_writes if w[0] == NULL_TASK_ID
        ]:
            apply_writes(
                self.checkpoint,
                self.channels,
                [PregelTaskWrites((), INPUT, null_writes, [])],
                self.checkpointer_get_next_version,
                self.trigger_to_nodes,
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
                "values", map_output_values, self.output_keys, True, self.channels
            )
        # map inputs to channel updates
        elif input_writes := deque(map_input(input_keys, self.input)):
            # discard any unfinished tasks from previous checkpoint
            discard_tasks = prepare_next_tasks(
                self.checkpoint,
                self.checkpoint_pending_writes,
                self.nodes,
                self.channels,
                self.managed,
                self.config,
                self.step,
                self.stop,
                for_execution=True,
                store=None,
                checkpointer=None,
                manager=None,
            )
            # apply input writes
            updated_channels = apply_writes(
                self.checkpoint,
                self.channels,
                [
                    *discard_tasks.values(),
                    PregelTaskWrites((), INPUT, input_writes, []),
                ],
                self.checkpointer_get_next_version,
                self.trigger_to_nodes,
            )
            # save input checkpoint
            self._put_checkpoint({"source": "input"})
        elif CONFIG_KEY_RESUMING not in configurable:
            raise EmptyInputError(f"Received no input for {input_keys}")
        # update config
        if not self.is_nested:
            self.config = patch_configurable(
                self.config, {CONFIG_KEY_RESUMING: is_resuming}
            )
        # set flag
        self.status = "pending"
        return updated_channels

    def _put_checkpoint(self, metadata: CheckpointMetadata) -> None:
        # assign step and parents
        exiting = metadata is self.checkpoint_metadata
        if exiting and self.checkpoint["id"] == self.checkpoint_id_saved:
            # checkpoint already saved
            return
        if not exiting:
            metadata["step"] = self.step
            metadata["parents"] = self.config[CONF].get(CONFIG_KEY_CHECKPOINT_MAP, {})
            self.checkpoint_metadata = metadata
        # do checkpoint?
        do_checkpoint = self._checkpointer_put_after_previous is not None and (
            exiting or self.checkpoint_during
        )
        # create new checkpoint
        self.checkpoint = create_checkpoint(
            self.checkpoint,
            self.channels if do_checkpoint else None,
            self.step,
            id=self.checkpoint["id"] if exiting else None,
        )
        # bail if no checkpointer
        if do_checkpoint and self._checkpointer_put_after_previous is not None:
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
        if not exiting:
            # increment step
            self.step += 1

    def _suppress_interrupt(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> bool | None:
        # persist current checkpoint and writes
        if not self.checkpoint_during and (
            # if it's a top graph
            not self.is_nested
            # or a nested graph with error or interrupt
            or exc_value is not None
            # or a nested graph with checkpointer=True
            or all(NS_END not in part for part in self.checkpoint_ns)
        ):
            self._put_checkpoint(self.checkpoint_metadata)
            self._put_pending_writes()
        # suppress interrupt
        suppress = isinstance(exc_value, GraphInterrupt) and not self.is_nested
        if suppress:
            # emit one last "values" event, with pending writes applied
            if (
                hasattr(self, "tasks")
                and self.checkpoint_pending_writes
                and any(task.writes for task in self.tasks.values())
            ):
                updated_channels = apply_writes(
                    self.checkpoint,
                    self.channels,
                    self.tasks.values(),
                    self.checkpointer_get_next_version,
                    self.trigger_to_nodes,
                )
                if not updated_channels.isdisjoint(
                    (self.output_keys,)
                    if isinstance(self.output_keys, str)
                    else self.output_keys
                ):
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
        mode: StreamMode,
        values: Callable[P, Iterator[Any]],
        *args: P.args,
        **kwargs: P.kwargs,
    ) -> None:
        if self.stream is None:
            return
        debug_remap = mode in ("checkpoints", "tasks") and "debug" in self.stream.modes
        if mode not in self.stream.modes and not debug_remap:
            return
        for v in values(*args, **kwargs):
            if mode in self.stream.modes:
                self.stream((self.checkpoint_ns, mode, v))
            # "debug" mode is "checkpoints" or "tasks" with a wrapper dict
            if debug_remap:
                self.stream(
                    (
                        self.checkpoint_ns,
                        "debug",
                        {
                            "step": self.step - 1
                            if mode == "checkpoints"
                            else self.step,
                            "timestamp": datetime.now(timezone.utc).isoformat(),
                            "type": "checkpoint"
                            if mode == "checkpoints"
                            else "task_result"
                            if "result" in v
                            else "task",
                            "payload": v,
                        },
                    )
                )

    def output_writes(
        self, task_id: str, writes: WritesT, *, cached: bool = False
    ) -> None:
        if task := self.tasks.get(task_id):
            if task.config is not None and TAG_HIDDEN in task.config.get(
                "tags", EMPTY_SEQ
            ):
                return
            if writes[0][0] == INTERRUPT:
                # in loop.py we append a bool to the PUSH task paths to indicate
                # whether or not a call was present. If so,
                # we don't emit the interrupt as it'll be emitted by the parent
                if task.path[0] == PUSH and task.path[-1] is True:
                    return
                interrupts = [
                    {
                        INTERRUPT: tuple(
                            v
                            for w in writes
                            if w[0] == INTERRUPT
                            for v in (w[1] if isinstance(w[1], Sequence) else (w[1],))
                        )
                    }
                ]
                self._emit("updates", lambda: iter(interrupts))
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
                    "tasks",
                    map_debug_task_results,
                    (task, writes),
                    self.stream_keys,
                )


class SyncPregelLoop(PregelLoop, AbstractContextManager):
    def __init__(
        self,
        input: Any | None,
        *,
        stream: StreamProtocol | None,
        config: RunnableConfig,
        store: BaseStore | None,
        cache: BaseCache | None,
        checkpointer: BaseCheckpointSaver | None,
        nodes: Mapping[str, PregelNode],
        specs: Mapping[str, BaseChannel | ManagedValueSpec],
        trigger_to_nodes: Mapping[str, Sequence[str]],
        manager: None | AsyncParentRunManager | ParentRunManager = None,
        interrupt_after: All | Sequence[str] = EMPTY_SEQ,
        interrupt_before: All | Sequence[str] = EMPTY_SEQ,
        input_keys: str | Sequence[str] = EMPTY_SEQ,
        output_keys: str | Sequence[str] = EMPTY_SEQ,
        stream_keys: str | Sequence[str] = EMPTY_SEQ,
        migrate_checkpoint: Callable[[Checkpoint], None] | None = None,
        retry_policy: Sequence[RetryPolicy] = (),
        cache_policy: CachePolicy | None = None,
        checkpoint_during: bool = True,
    ) -> None:
        super().__init__(
            input,
            stream=stream,
            config=config,
            checkpointer=checkpointer,
            cache=cache,
            store=store,
            nodes=nodes,
            specs=specs,
            input_keys=input_keys,
            output_keys=output_keys,
            stream_keys=stream_keys,
            interrupt_after=interrupt_after,
            interrupt_before=interrupt_before,
            manager=manager,
            migrate_checkpoint=migrate_checkpoint,
            trigger_to_nodes=trigger_to_nodes,
            retry_policy=retry_policy,
            cache_policy=cache_policy,
            checkpoint_during=checkpoint_during,
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
        prev: concurrent.futures.Future | None,
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

    def match_cached_writes(self) -> Sequence[PregelExecutableTask]:
        if self.cache is None:
            return ()
        matched: list[PregelExecutableTask] = []
        if cached := {
            (t.cache_key.ns, t.cache_key.key): t
            for t in self.tasks.values()
            if t.cache_key and not t.writes
        }:
            for key, values in self.cache.get(tuple(cached)).items():
                task = cached[key]
                task.writes.extend(values)
                matched.append(task)
        return matched

    def accept_push(
        self, task: PregelExecutableTask, write_idx: int, call: Call | None = None
    ) -> PregelExecutableTask | None:
        if pushed := super().accept_push(task, write_idx, call):
            for task in self.match_cached_writes():
                self.output_writes(task.id, task.writes, cached=True)
        return pushed

    def put_writes(self, task_id: str, writes: WritesT) -> None:
        """Put writes for a task, to be read by the next tick."""
        super().put_writes(task_id, writes)
        if not writes or self.cache is None or not hasattr(self, "tasks"):
            return
        task = self.tasks.get(task_id)
        if task is None or task.cache_key is None:
            return
        self.submit(
            self.cache.set,
            {
                (task.cache_key.ns, task.cache_key.key): (
                    task.writes,
                    task.cache_key.ttl,
                )
            },
        )

    # context manager

    def __enter__(self) -> Self:
        if self.checkpointer:
            saved = self.checkpointer.get_tuple(self.checkpoint_config)
        else:
            saved = None
        if saved is None:
            saved = CheckpointTuple(
                self.checkpoint_config, empty_checkpoint(), {"step": -2}, None, []
            )
        elif self._migrate_checkpoint is not None:
            self._migrate_checkpoint(saved.checkpoint)
        self.checkpoint_config = {
            **self.checkpoint_config,
            **saved.config,
            CONF: {
                CONFIG_KEY_CHECKPOINT_NS: "",
                **self.checkpoint_config.get(CONF, {}),
                **saved.config.get(CONF, {}),
            },
        }
        self.prev_checkpoint_config = saved.parent_config
        self.checkpoint_id_saved = saved.checkpoint["id"]
        self.checkpoint = saved.checkpoint
        self.checkpoint_metadata = saved.metadata
        self.checkpoint_pending_writes = (
            [(str(tid), k, v) for tid, k, v in saved.pending_writes]
            if saved.pending_writes is not None
            else []
        )

        self.submit = self.stack.enter_context(BackgroundExecutor(self.config))
        self.channels, self.managed = channels_from_checkpoint(
            self.specs, self.checkpoint
        )
        self.stack.push(self._suppress_interrupt)
        self.status = "input"
        self.step = self.checkpoint_metadata["step"] + 1
        self.stop = self.step + self.config["recursion_limit"] + 1
        self.checkpoint_previous_versions = self.checkpoint["channel_versions"].copy()
        self.updated_channels = self._first(input_keys=self.input_keys)

        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> bool | None:
        # unwind stack
        return self.stack.__exit__(exc_type, exc_value, traceback)


class AsyncPregelLoop(PregelLoop, AbstractAsyncContextManager):
    def __init__(
        self,
        input: Any | None,
        *,
        stream: StreamProtocol | None,
        config: RunnableConfig,
        store: BaseStore | None,
        cache: BaseCache | None,
        checkpointer: BaseCheckpointSaver | None,
        nodes: Mapping[str, PregelNode],
        specs: Mapping[str, BaseChannel | ManagedValueSpec],
        trigger_to_nodes: Mapping[str, Sequence[str]],
        interrupt_after: All | Sequence[str] = EMPTY_SEQ,
        interrupt_before: All | Sequence[str] = EMPTY_SEQ,
        manager: None | AsyncParentRunManager | ParentRunManager = None,
        input_keys: str | Sequence[str] = EMPTY_SEQ,
        output_keys: str | Sequence[str] = EMPTY_SEQ,
        stream_keys: str | Sequence[str] = EMPTY_SEQ,
        migrate_checkpoint: Callable[[Checkpoint], None] | None = None,
        retry_policy: Sequence[RetryPolicy] = (),
        cache_policy: CachePolicy | None = None,
        checkpoint_during: bool = True,
    ) -> None:
        super().__init__(
            input,
            stream=stream,
            config=config,
            checkpointer=checkpointer,
            cache=cache,
            store=store,
            nodes=nodes,
            specs=specs,
            input_keys=input_keys,
            output_keys=output_keys,
            stream_keys=stream_keys,
            interrupt_after=interrupt_after,
            interrupt_before=interrupt_before,
            manager=manager,
            migrate_checkpoint=migrate_checkpoint,
            trigger_to_nodes=trigger_to_nodes,
            retry_policy=retry_policy,
            cache_policy=cache_policy,
            checkpoint_during=checkpoint_during,
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
        prev: asyncio.Task | None,
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

    async def amatch_cached_writes(self) -> Sequence[PregelExecutableTask]:
        if self.cache is None:
            return []
        matched: list[PregelExecutableTask] = []
        if cached := {
            (t.cache_key.ns, t.cache_key.key): t
            for t in self.tasks.values()
            if t.cache_key and not t.writes
        }:
            for key, values in (await self.cache.aget(tuple(cached))).items():
                task = cached[key]
                task.writes.extend(values)
                matched.append(task)
        return matched

    async def aaccept_push(
        self, task: PregelExecutableTask, write_idx: int, call: Call | None = None
    ) -> PregelExecutableTask | None:
        if pushed := super().accept_push(task, write_idx, call):
            for task in await self.amatch_cached_writes():
                self.output_writes(task.id, task.writes, cached=True)
        return pushed

    def put_writes(self, task_id: str, writes: WritesT) -> None:
        """Put writes for a task, to be read by the next tick."""
        super().put_writes(task_id, writes)
        if not writes or self.cache is None or not hasattr(self, "tasks"):
            return
        task = self.tasks.get(task_id)
        if task is None or task.cache_key is None:
            return
        if writes[0][0] in (INTERRUPT, ERROR):
            # only cache successful tasks
            return
        self.submit(
            self.cache.aset,
            {
                (task.cache_key.ns, task.cache_key.key): (
                    task.writes,
                    task.cache_key.ttl,
                )
            },
        )

    # context manager

    async def __aenter__(self) -> Self:
        if self.checkpointer:
            saved = await self.checkpointer.aget_tuple(self.checkpoint_config)
        else:
            saved = None
        if saved is None:
            saved = CheckpointTuple(
                self.checkpoint_config, empty_checkpoint(), {"step": -2}, None, []
            )
        elif self._migrate_checkpoint is not None:
            self._migrate_checkpoint(saved.checkpoint)
        self.checkpoint_config = {
            **self.checkpoint_config,
            **saved.config,
            CONF: {
                CONFIG_KEY_CHECKPOINT_NS: "",
                **self.checkpoint_config.get(CONF, {}),
                **saved.config.get(CONF, {}),
            },
        }
        self.prev_checkpoint_config = saved.parent_config
        self.checkpoint_id_saved = saved.checkpoint["id"]
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
        self.channels, self.managed = channels_from_checkpoint(
            self.specs, self.checkpoint
        )
        self.stack.push(self._suppress_interrupt)
        self.status = "input"
        self.step = self.checkpoint_metadata["step"] + 1
        self.stop = self.step + self.config["recursion_limit"] + 1
        self.checkpoint_previous_versions = self.checkpoint["channel_versions"].copy()
        self.updated_channels = self._first(input_keys=self.input_keys)

        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> bool | None:
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
