from __future__ import annotations

import asyncio
import binascii
import concurrent.futures
from collections import defaultdict, deque
from collections.abc import Callable, Iterator, Mapping, Sequence
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
    Literal,
    TypeVar,
    cast,
)

from langchain_core.callbacks import AsyncParentRunManager, ParentRunManager
from langchain_core.runnables import RunnableConfig
from langgraph.cache.base import BaseCache
from langgraph.checkpoint.base import (
    WRITES_IDX_MAP,
    BaseCheckpointSaver,
    ChannelVersions,
    Checkpoint,
    CheckpointMetadata,
    CheckpointTuple,
    PendingWrite,
)
from langgraph.store.base import BaseStore
from typing_extensions import ParamSpec, Self

from langgraph._internal._config import patch_configurable
from langgraph._internal._constants import (
    CONF,
    CONFIG_KEY_CHECKPOINT_ID,
    CONFIG_KEY_CHECKPOINT_MAP,
    CONFIG_KEY_CHECKPOINT_NS,
    CONFIG_KEY_REPLAY_STATE,
    CONFIG_KEY_RESUME_MAP,
    CONFIG_KEY_RESUMING,
    CONFIG_KEY_RUNTIME,
    CONFIG_KEY_SCRATCHPAD,
    CONFIG_KEY_STREAM,
    CONFIG_KEY_TASK_ID,
    CONFIG_KEY_THREAD_ID,
    ERROR,
    ERROR_SOURCE_NODE,
    INPUT,
    INTERRUPT,
    NS_END,
    NS_SEP,
    NULL_TASK_ID,
    PUSH,
    RESUME,
    TASKS,
)
from langgraph._internal._replay import ReplayState
from langgraph._internal._scratchpad import PregelScratchpad
from langgraph._internal._typing import EMPTY_SEQ, MISSING
from langgraph.callbacks import (
    GraphInterruptEvent,
    GraphLifecycleEvent,
    GraphResumeEvent,
)
from langgraph.channels.base import BaseChannel
from langgraph.channels.delta import DeltaChannel
from langgraph.channels.untracked_value import UntrackedValue
from langgraph.constants import TAG_HIDDEN
from langgraph.errors import (
    EmptyInputError,
    GraphInterrupt,
)
from langgraph.managed.base import (
    ManagedValueMapping,
    ManagedValueSpec,
)
from langgraph.pregel._algo import (
    Call,
    GetNextVersion,
    PregelTaskWrites,
    apply_writes,
    checkpoint_null_version,
    increment,
    prepare_next_tasks,
    prepare_node_error_handler_task,
    prepare_single_task,
    sanitize_untracked_values_in_send,
    should_interrupt,
    task_path_str,
)
from langgraph.pregel._checkpoint import (
    achannels_from_checkpoint,
    channels_from_checkpoint,
    copy_checkpoint,
    create_checkpoint,
    delta_channels_to_snapshot,
    empty_checkpoint,
)
from langgraph.pregel._executor import (
    AsyncBackgroundExecutor,
    BackgroundExecutor,
    Submit,
)
from langgraph.pregel._io import (
    map_command,
    map_input,
    map_output_updates,
    map_output_values,
    read_channels,
)
from langgraph.pregel._read import PregelNode
from langgraph.pregel._utils import get_new_channel_versions, is_xxh3_128_hexdigest
from langgraph.pregel.debug import (
    map_debug_checkpoint,
    map_debug_task_results,
    map_debug_tasks,
)
from langgraph.pregel.protocol import StreamChunk, StreamProtocol
from langgraph.runtime import RunControl, Runtime
from langgraph.types import (
    All,
    CachePolicy,
    Command,
    Durability,
    Interrupt,
    PregelExecutableTask,
    RetryPolicy,
    Send,
    StreamMode,
)

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
    is_replaying: bool
    is_nested: bool
    manager: None | AsyncParentRunManager | ParentRunManager
    interrupt_after: All | Sequence[str]
    interrupt_before: All | Sequence[str]
    durability: Durability
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
    # Futures from `checkpointer.put_writes` calls that produced delta-channel
    # writes. `_checkpointer_put_after_previous` drains this list (swap to a
    # local `futs` then reset to `[]` and wait/gather) before putting the
    # next checkpoint, so a checkpoint never becomes durable before the
    # writes that produced it. Initialised to `[]` in both sync and async
    # `__enter__`; stays `None` only when no checkpointer.
    _delta_write_futs: list[Any] | None = None

    # Same pattern as `_delta_write_futs` but for error-handler writes.
    # When `put_writes` persists an ERROR_SOURCE_NODE marker, the future is
    # appended here.  `schedule_error_handler` / `aschedule_error_handler`
    # drain this list so the write is durable before the handler starts.
    _error_handler_write_futs: list[Any] | None = None

    # Exit-mode accumulator: every delta-channel write produced during this
    # run (input writes from `_first` + per-superstep writes captured in
    # `after_tick`). At exit, `_put_exit_delta_writes` filters out channels
    # that will snapshot, then persists the rest under an anchor parent.
    # `None` when not in exit mode (so the capture sites are no-ops).
    # Each tuple is `(step, task_id, channel, value)` — `step` drives the
    # synthetic step-prefixed task_id used to preserve chronological order
    # under the saver's `ORDER BY task_id, idx` sorting.
    _exit_delta_writes: list[tuple[int, str, str, Any]] | None = None

    # The checkpoint_config that points at the parent loaded at `__enter__`
    # (or the synthetic-empty checkpoint, on first run). We capture it
    # eagerly because every `_put_checkpoint` advances `self.checkpoint_config`
    # to the newly-saved checkpoint's id — by exit time the original parent
    # config would otherwise be lost. `_put_exit_delta_writes` uses this:
    # on resumed runs as the anchor for exit delta writes; on first runs
    # to derive the lazy stub's config (its `checkpoint_id` is the
    # synthetic-empty id we want the stub persisted under).
    _initial_checkpoint_config: RunnableConfig

    # True iff the saver actually returned a tuple at `__enter__`. False
    # on the first-ever run for a thread (no parent persisted yet).
    # `_put_exit_delta_writes` uses this to decide between anchoring on
    # the existing parent (True) or creating a lazy stub (False).
    _has_persisted_parent: bool = False

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
        "draining",
        "interrupt_before",
        "interrupt_after",
        "out_of_steps",
    ]
    control: RunControl | None
    tasks: dict[str, PregelExecutableTask]
    output: None | dict[str, Any] | Any = None
    updated_channels: set[str] | None = None
    _graph_lifecycle_events: deque[GraphLifecycleEvent]
    _has_graph_lifecycle_callbacks: bool

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
        durability: Durability,
        interrupt_after: All | Sequence[str] = EMPTY_SEQ,
        interrupt_before: All | Sequence[str] = EMPTY_SEQ,
        manager: None | AsyncParentRunManager | ParentRunManager = None,
        migrate_checkpoint: Callable[[Checkpoint], None] | None = None,
        retry_policy: Sequence[RetryPolicy] = (),
        cache_policy: CachePolicy | None = None,
        has_graph_lifecycle_callbacks: bool = False,
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
        self.is_replaying = CONFIG_KEY_CHECKPOINT_ID in config[CONF]
        self._migrate_checkpoint = migrate_checkpoint
        self.trigger_to_nodes = trigger_to_nodes
        self.retry_policy = retry_policy
        self.cache_policy = cache_policy
        self.durability = durability
        self._has_graph_lifecycle_callbacks = has_graph_lifecycle_callbacks
        self._graph_lifecycle_events = deque()
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
        runtime = self.config[CONF].get(CONFIG_KEY_RUNTIME)
        self.control = runtime.control if isinstance(runtime, Runtime) else None

    def _push_graph_lifecycle_event(
        self,
        kind: Literal["resume", "interrupt"],
        *,
        interrupts: tuple[Interrupt, ...] = (),
    ) -> None:
        # drain status never reaches lifecycle events: tick() returns False
        # before pushing, and interrupts are raised through GraphInterrupt
        if self.status == "draining":
            raise RuntimeError("Draining status cannot emit lifecycle events")
        status = self.status
        if kind == "resume":
            self._graph_lifecycle_events.append(
                GraphResumeEvent(
                    run_id=None,
                    status=status,
                    checkpoint_id=self.checkpoint["id"],
                    checkpoint_ns=self.checkpoint_ns,
                )
            )
        elif kind == "interrupt":
            self._graph_lifecycle_events.append(
                GraphInterruptEvent(
                    run_id=None,
                    status=status,
                    checkpoint_id=self.checkpoint["id"],
                    checkpoint_ns=self.checkpoint_ns,
                    interrupts=interrupts,
                )
            )
        else:
            msg = f"Unknown graph lifecycle event type: {kind}"
            raise AssertionError(msg)

    def _pop_lifecycle_event(self) -> GraphLifecycleEvent | None:
        if not self._graph_lifecycle_events:
            return None
        return self._graph_lifecycle_events.popleft()

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

        # check if any writes are to an UntrackedValue channel
        if any(
            isinstance(channel, UntrackedValue) for channel in self.channels.values()
        ):
            # we do not persist untracked values in checkpoints
            writes_to_save = [
                # sanitize UntrackedValues that are nested within Send packets
                (
                    (c, sanitize_untracked_values_in_send(v, self.channels))
                    if c == TASKS and isinstance(v, Send)
                    else (c, v)
                )
                for c, v in writes_to_save
                # dont persist UntrackedValue channel writes
                if not isinstance(self.specs.get(c), UntrackedValue)
            ]

        # save writes
        self.checkpoint_pending_writes.extend((task_id, c, v) for c, v in writes)
        if self.durability != "exit" and self.checkpointer_put_writes is not None:
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
                fut = self.submit(
                    self.checkpointer_put_writes,
                    config,
                    writes_to_save,
                    task_id,
                    task_path_str(task.path) if task else "",
                )
            else:
                fut = self.submit(
                    self.checkpointer_put_writes,
                    config,
                    writes_to_save,
                    task_id,
                )
            if self._delta_write_futs is not None and any(
                isinstance(self.specs.get(c), DeltaChannel) for c, _ in writes_to_save
            ):
                self._delta_write_futs.append(fut)
            # ERROR_SOURCE_NODE is only appended by commit() when the task
            # has an error handler (_should_route_to_error_handler), so this
            # check naturally limits future collection to those tasks.
            if self._error_handler_write_futs is not None and any(
                c == ERROR_SOURCE_NODE for c, _ in writes
            ):
                self._error_handler_write_futs.append(fut)
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
            PregelExecutableTask | None,
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
            if not self.is_replaying:
                self._reapply_writes_to_succeeded_nodes({pushed.id: pushed})
            # return the new task, to be started if not run before
            return pushed

    def schedule_error_handler(
        self, failed_task: PregelExecutableTask, error: BaseException
    ) -> PregelExecutableTask | None:
        raise NotImplementedError

    async def aschedule_error_handler(
        self, failed_task: PregelExecutableTask, error: BaseException
    ) -> PregelExecutableTask | None:
        raise NotImplementedError

    def tick(self) -> bool:
        """Execute a single iteration of the Pregel loop.

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

        if self.control is not None and self.control.drain_requested:
            self.status = "draining"
            return False

        # if there are pending writes from a previous loop, apply them
        if not self.is_replaying and self.checkpoint_pending_writes:
            self._reapply_writes_to_succeeded_nodes(self.tasks)
            self._resume_error_handlers_if_applicable()

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
        # capture delta-channel writes for exit-mode accumulator before clearing
        if self._exit_delta_writes is not None:
            for tid, ch, v in self.checkpoint_pending_writes:
                if isinstance(self.specs.get(ch), DeltaChannel):
                    self._exit_delta_writes.append((self.step, tid, ch, v))
        # clear pending writes
        self.checkpoint_pending_writes.clear()
        # only replay (re-execute) done tasks on the first tick
        self.is_replaying = False
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

    def _reapply_writes_to_succeeded_nodes(
        self, tasks: Mapping[str, PregelExecutableTask]
    ) -> None:
        """Restore successful channel writes from checkpoint to in-memory tasks.

        Skips control signals (ERROR, ERROR_SOURCE_NODE, INTERRUPT, RESUME)
        so that failed/interrupted tasks remain with empty writes and will be
        re-executed (or routed to error handlers) by the runner.
        """
        for tid, k, v in self.checkpoint_pending_writes:
            if k in (ERROR, ERROR_SOURCE_NODE, INTERRUPT, RESUME):
                continue
            if task := tasks.get(tid):
                task.writes.append((k, v))

    def _resume_error_handlers_if_applicable(self) -> None:
        """On resume, schedule error handlers for tasks that failed in a prior run.

        Called right after ``_reapply_writes_to_succeeded_nodes`` during ``tick()``.
        At that point, ``_reapply_writes_to_succeeded_nodes`` has already skipped
        ERROR / ERROR_SOURCE_NODE writes, so a previously-failed task still has
        empty ``writes``.  Without intervention the runner (which executes only
        tasks where ``not t.writes``) would re-run the original node.

        This method prevents that re-execution for nodes that have an error
        handler:

        1. Scan ``checkpoint_pending_writes`` for ERROR_SOURCE_NODE markers
           persisted by a prior ``commit()``.  Each marker means "this task
           already failed and was routed to an error handler".
        2. For each such task, write ``(ERROR, error)`` into ``task.writes``
           so the task is no longer empty — the runner will skip it.
        3. Prepare a fresh error-handler task and add it to ``self.tasks``.
           Because the handler task starts with empty ``writes``, the runner
           will pick it up and execute it.
        """
        # Phase 1: collect task-ids that have ERROR_SOURCE_NODE + ERROR pairs.
        failed: dict[str, BaseException] = {}
        for tid, chan, val in self.checkpoint_pending_writes:
            if chan == ERROR_SOURCE_NODE:
                error = next(
                    (
                        v
                        for t, c, v in self.checkpoint_pending_writes
                        if t == tid and c == ERROR
                    ),
                    None,
                )
                if error is not None:
                    failed[tid] = error
        # Phase 2: mark originals as done, schedule handler tasks.
        for task_id, error in failed.items():
            task = self.tasks.get(task_id)
            if task is None:
                continue
            handler_node = self.nodes[task.name].error_handler_node
            if not handler_node:
                continue
            # Non-empty writes → runner's `not t.writes` filter skips this task.
            task.writes.append((ERROR, error))
            # The handler task starts with empty writes → runner will execute it.
            handler_task = prepare_node_error_handler_task(
                task,
                handler_node_name=handler_node,
                failed_error=error,
                checkpoint=self.checkpoint,
                pending_writes=self.checkpoint_pending_writes,
                processes=self.nodes,
                channels=self.channels,
                managed=self.managed,
                config=task.config,
                step=self.step,
                stop=self.stop,
                store=self.store,
                checkpointer=self.checkpointer,
                manager=self.manager,
                retry_policy=self.retry_policy,
                cache_policy=self.cache_policy,
            )
            if handler_task is not None:
                self.tasks[handler_task.id] = handler_task

    def _pending_interrupts(self) -> set[str]:
        """Return the set of interrupt ids that are pending without corresponding resume values."""
        # mapping of task ids to interrupt ids
        pending_interrupts: dict[str, str] = {}

        # set of resume task ids
        pending_resumes: set[str] = set()

        for task_id, write_type, value in self.checkpoint_pending_writes:
            if write_type == INTERRUPT:
                # interrupts is always a list, but there should only be one element
                pending_interrupts[task_id] = value[0].id
            elif write_type == RESUME:
                pending_resumes.add(task_id)

        resumed_interrupt_ids = {
            pending_interrupts[task_id]
            for task_id in pending_resumes
            if task_id in pending_interrupts
        }

        # Keep only interrupts whose interrupt_id is not resumed
        hanging_interrupts: set[str] = {
            interrupt_id
            for interrupt_id in pending_interrupts.values()
            if interrupt_id not in resumed_interrupt_ids
        }

        return hanging_interrupts

    def _first(
        self, *, input_keys: str | Sequence[str], updated_channels: set[str] | None
    ) -> set[str] | None:
        # Resuming from a previous checkpoint requires two things:
        # 1. A prior checkpoint exists (channel_versions is non-empty)
        # 2. The input signals continuation (not a fresh run with new input)
        # For subgraphs, the parent explicitly sets CONFIG_KEY_RESUMING.
        # For the outer graph, we infer from the input:
        #   - None input: resume after interrupt (invoke(None, config))
        #   - Command input: any Command operates on existing state
        #   - Same run_id: re-entry into an ongoing run (e.g. stream reconnect)
        configurable = self.config.get(CONF, {})
        input_is_command = isinstance(self.input, Command)
        is_resuming = bool(self.checkpoint["channel_versions"]) and bool(
            configurable.get(
                CONFIG_KEY_RESUMING,
                self.input is None
                or input_is_command
                or (
                    not self.is_nested
                    and self.config.get("metadata", {}).get("run_id")
                    == self.checkpoint_metadata.get("run_id", MISSING)
                ),
            )
        )

        # When replaying from a specific checkpoint, drop cached RESUME
        # writes so that interrupt() calls re-fire instead of returning
        # stale values. But if we're actively resuming, keep them —
        # multi-interrupt scenarios need previously resolved values preserved.
        is_time_traveling = self.is_replaying and (
            # Time-travel to a subgraph checkpoint: the parent sets
            # RESUMING=True (it can't distinguish time-travel from resume),
            # so we check if this subgraph's own ns is in checkpoint_map.
            # Normally the map only has ancestor entries (_algo.py); the
            # subgraph's own entry only appears via get_state(subgraphs=True).
            (
                self.is_nested
                and configurable.get(CONFIG_KEY_CHECKPOINT_NS, "")
                in configurable.get(CONFIG_KEY_CHECKPOINT_MAP, {})
            )
            or not (
                # Outer graph: resume arrives as Command(resume=...)
                (input_is_command and cast(Command, self.input).resume is not None)
                # Subgraphs: resume arrives via config flag from parent
                # (subgraph input is a Send arg, not a Command)
                or configurable.get(CONFIG_KEY_RESUMING, False)
            )
        )
        if is_time_traveling:
            self.checkpoint_pending_writes = [
                w for w in self.checkpoint_pending_writes if w[1] != RESUME
            ]

        # map command to writes
        if input_is_command:
            if (resume := cast(Command, self.input).resume) is not None:
                if not self.checkpointer:
                    raise RuntimeError(
                        "Cannot use Command(resume=...) without checkpointer"
                    )

                if resume_is_map := (
                    isinstance(resume, dict)
                    and all(is_xxh3_128_hexdigest(k) for k in resume)
                ):
                    self.config[CONF][CONFIG_KEY_RESUME_MAP] = resume
                else:
                    if len(self._pending_interrupts()) > 1:
                        raise RuntimeError(
                            "When there are multiple pending interrupts, you must specify the interrupt id when resuming. "
                            "Docs: https://docs.langchain.com/oss/python/langgraph/add-human-in-the-loop#resume-multiple-interrupts-with-one-invocation."
                        )

            writes: defaultdict[str, list[tuple[str, Any]]] = defaultdict(list)
            # group writes by task ID
            for tid, c, v in map_command(cmd=cast(Command, self.input)):
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
            null_updated_channels = apply_writes(
                self.checkpoint,
                self.channels,
                [PregelTaskWrites((), INPUT, null_writes, [])],
                self.checkpointer_get_next_version,
                self.trigger_to_nodes,
            )
            if updated_channels is not None:
                updated_channels.update(null_updated_channels)
        # proceed past previous checkpoint
        if is_resuming:
            self.checkpoint["versions_seen"].setdefault(INTERRUPT, {})
            for k in self.channels:
                if k in self.checkpoint["channel_versions"]:
                    version = self.checkpoint["channel_versions"][k]
                    self.checkpoint["versions_seen"][INTERRUPT][k] = version
            # When time-traveling (replaying from a specific checkpoint),
            # save a fork checkpoint so the replayed execution creates a
            # new branch. Without this, if the execution hits an interrupt
            # before after_tick() runs, no new checkpoint is created —
            # the parent's latest checkpoint remains the old one and
            # subsequent resumes load the wrong state.
            # Skip for update_state forks (source=update/fork) since they
            # already have their own fork checkpoint.
            if is_time_traveling and self.checkpoint_metadata.get("source") not in (
                "update",
                "fork",
            ):
                # Clear old INTERRUPT writes from the loaded checkpoint.
                # The fork will have a new checkpoint_id which changes
                # task IDs — stale interrupt writes would accumulate and
                # confuse the multiple-interrupt check in future resumes.
                self.checkpoint_pending_writes = [
                    w for w in self.checkpoint_pending_writes if w[1] != INTERRUPT
                ]
                self._put_checkpoint({"source": "fork"})
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
                updated_channels=updated_channels,
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
            # Input writes go through `apply_writes` directly (above) — they
            # never enter `checkpoint_pending_writes`, so the after_tick
            # capture site does not see them. In exit mode, capture them
            # here so `_exit_delta_writes` includes the input's delta writes
            # alongside per-superstep writes; otherwise the input would be
            # lost on read (it's not in final_checkpoint.channel_values for
            # sub-freq channels, and walks ignore target.pending_writes).
            if self._exit_delta_writes is not None:
                for c, v in input_writes:
                    if isinstance(self.specs.get(c), DeltaChannel):
                        self._exit_delta_writes.append((self.step, NULL_TASK_ID, c, v))
            # Persist delta-channel input writes so sub-freq inputs are
            # recoverable via ancestor walk (mirrors the Command input path).
            if self.durability != "exit":
                delta_input = [
                    (c, v)
                    for c, v in input_writes
                    if isinstance(self.specs.get(c), DeltaChannel)
                ]
                if delta_input:
                    self.put_writes(NULL_TASK_ID, delta_input)
            # save input checkpoint
            self.updated_channels = updated_channels
            self._put_checkpoint({"source": "input"})
        elif CONFIG_KEY_RESUMING not in configurable:
            raise EmptyInputError(f"Received no input for {input_keys}")
        # Propagate resuming and replaying flags to subgraphs.
        if not self.is_nested:
            # Pass the resolved before-bound checkpoint ID so subgraphs can
            # find their corresponding checkpoint without re-fetching the
            # parent. For forks (source=update/fork), use the fork's parent
            # checkpoint ID since the fork was created after the subgraph's
            # checkpoints from the original execution.
            #
            # Only gate on is_time_traveling (not is_replaying). When the
            # client resumes with an explicit checkpoint_id that happens to
            # point at the current head (e.g. LangGraph Studio sending
            # `checkpoint: {checkpoint_id}` alongside Command(resume=...)),
            # is_replaying is True but is_time_traveling is False. In that
            # case subgraphs should load their latest checkpoint normally,
            # not go through ReplayState's before-bound lookup which would
            # miss subgraph checkpoints created during processing of the
            # current parent step.
            replay_state: ReplayState | None = None
            if is_time_traveling:
                replay_checkpoint_id = self.checkpoint["id"]
                if (
                    self.checkpoint_metadata.get("source")
                    in (
                        "update",
                        "fork",
                    )
                    and self.prev_checkpoint_config
                ):
                    replay_checkpoint_id = self.prev_checkpoint_config[CONF].get(
                        CONFIG_KEY_CHECKPOINT_ID, replay_checkpoint_id
                    )
                replay_state = ReplayState(replay_checkpoint_id)
            self.config = patch_configurable(
                self.config,
                {
                    CONFIG_KEY_RESUMING: is_resuming,
                    CONFIG_KEY_REPLAY_STATE: replay_state,
                },
            )
        # set flag
        self.status = "pending"
        if is_resuming:
            self._push_graph_lifecycle_event("resume")
        return updated_channels

    def _put_checkpoint(self, metadata: CheckpointMetadata) -> None:
        # `is` (object identity) — not `==`. Three of four call sites pass a
        # fresh dict ({"source":"input"|"loop"|"fork"}); only
        # `_suppress_interrupt`(will rename to _on_loop_exit soon)
        # at exit reuses the existing `self.checkpoint_metadata` instance. So
        # `metadata is self.checkpoint_metadata` is True only on the exit call,
        # which is what we use to gate exit-only behaviour (skip count-bump,
        # don't replace metadata). Could be replaced by an explicit
        # `exiting: bool = False` parameter; left as-is to match the existing
        # idiom in this file.
        # TODO: replace with an explicit `exiting: bool = False` parameter.
        exiting = metadata is self.checkpoint_metadata
        if exiting and self.checkpoint["id"] == self.checkpoint_id_saved:
            # checkpoint already saved
            return
        # Per-delta-channel counter bookkeeping.
        #
        # Each delta channel tracks a (updates, supersteps) tuple:
        # - `updates` increments only when the channel is written this step.
        # - `supersteps` increments every superstep regardless.
        #
        # `_put_checkpoint` is called once per superstep with a fresh
        # metadata dict (source="input"|"loop"|"fork") — those are the
        # intermediate calls that bump counters. In exit mode,
        # `_suppress_interrupt`(will rename to _on_loop_exit soon)
        # additionally calls `_put_checkpoint(self.checkpoint_metadata)` AT
        # EXIT to commit the final checkpoint — this runs *after* the last
        # intermediate call already counted the last superstep. So the
        # exit call must NOT bump again or it would double-count the last
        # superstep.
        if not exiting:
            prev_counters = dict(
                self.checkpoint_metadata.get("counters_since_delta_snapshot") or {}
            )
            new_counters: dict[str, tuple[int, int]] = {}
            updated = self.updated_channels or set()
            for ch_name, ch in self.channels.items():
                if not isinstance(ch, DeltaChannel):
                    continue
                u, s = prev_counters.get(ch_name, (0, 0))
                s += 1
                if ch_name in updated:
                    u += 1
                new_counters[ch_name] = (u, s)
            metadata["step"] = self.step
            metadata["parents"] = self.config[CONF].get(CONFIG_KEY_CHECKPOINT_MAP, {})
            self.checkpoint_metadata = metadata
        else:
            new_counters = dict(
                self.checkpoint_metadata.get("counters_since_delta_snapshot") or {}
            )
        # do checkpoint?
        do_checkpoint = self._checkpointer_put_after_previous is not None and (
            exiting or self.durability != "exit"
        )
        # create new checkpoint
        channels_to_snapshot = (
            delta_channels_to_snapshot(self.channels, new_counters)
            if do_checkpoint
            else set()
        )
        self.checkpoint = create_checkpoint(
            self.checkpoint,
            self.channels if do_checkpoint else None,
            self.step,
            id=self.checkpoint["id"] if exiting else None,
            updated_channels=self.updated_channels,
            get_next_version=self.checkpointer_get_next_version
            if do_checkpoint
            else None,
            channels_to_snapshot=channels_to_snapshot,
        )
        for k in channels_to_snapshot:
            new_counters[k] = (0, 0)
        non_zero = {k: v for k, v in new_counters.items() if v != (0, 0)}
        if non_zero:
            self.checkpoint_metadata["counters_since_delta_snapshot"] = non_zero
        elif "counters_since_delta_snapshot" in self.checkpoint_metadata:
            del self.checkpoint_metadata["counters_since_delta_snapshot"]
        # sanitize TASK channel in the checkpoint before saving (durability=="exit")
        if TASKS in self.checkpoint["channel_values"] and any(
            isinstance(channel, UntrackedValue) for channel in self.channels.values()
        ):
            sanitized_tasks = [
                sanitize_untracked_values_in_send(value, self.channels)
                if isinstance(value, Send)
                else value
                for value in self.checkpoint["channel_values"][TASKS]
            ]
            self.checkpoint["channel_values"][TASKS] = sanitized_tasks
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

    def _put_exit_delta_writes(self) -> None:
        """Stage stub + accumulated delta writes so final_checkpoint's put
        waits on them (visibility invariant: both must be durable before
        final_checkpoint becomes visible to readers).

        Stub is created lazily — only when no persisted parent exists AND at
        least one delta channel has writes that won't be snapshotted.
        """
        if (
            not self._exit_delta_writes
            or self.checkpointer is None
            or self._checkpointer_put_after_previous is None
            or self.checkpointer_put_writes is None
        ):
            return

        counters = dict(
            self.checkpoint_metadata.get("counters_since_delta_snapshot") or {}
        )
        channels_to_snapshot = delta_channels_to_snapshot(self.channels, counters)

        pending = [
            (step, tid, ch, v)
            for (step, tid, ch, v) in self._exit_delta_writes
            if ch not in channels_to_snapshot
        ]
        if not pending:
            return

        if self._has_persisted_parent:
            # _initial_checkpoint_config's checkpoint_id is the saved parent's
            # id (saver returned a real tuple at __enter__).
            anchor_config = self._initial_checkpoint_config
        else:
            stub_cp = empty_checkpoint()
            stub_cp["id"] = self.checkpoint_id_saved
            stub_cp["ts"] = datetime.now(timezone.utc).isoformat()
            # Stub has no parent (checkpoint_id=None in config).
            stub_put_config = patch_configurable(
                self._initial_checkpoint_config,
                {CONFIG_KEY_CHECKPOINT_ID: None},
            )
            # Anchor config for put_writes: checkpoint_id = stub's id.
            anchor_config = patch_configurable(
                self._initial_checkpoint_config,
                {CONFIG_KEY_CHECKPOINT_ID: stub_cp["id"]},
            )
            self._put_checkpoint_fut = self.submit(
                self._checkpointer_put_after_previous,
                getattr(self, "_put_checkpoint_fut", None),
                stub_put_config,
                stub_cp,
                {"step": -2},
                {},
            )
            # Set checkpoint_config so final_checkpoint's _put_checkpoint
            # sees the stub as its parent.
            self.checkpoint_config = anchor_config

        # Step-prefixed synthetic task_id preserves chronological superstep
        # order under the saver's ORDER BY task_id, idx sorting.
        grouped: dict[tuple[int, str], list[tuple[str, Any]]] = {}
        for step, tid, ch, v in pending:
            grouped.setdefault((step, tid), []).append((ch, v))
        anchor_write_config = patch_configurable(
            anchor_config,
            {
                CONFIG_KEY_CHECKPOINT_NS: self.config[CONF].get(
                    CONFIG_KEY_CHECKPOINT_NS, ""
                ),
                CONFIG_KEY_CHECKPOINT_ID: anchor_config[CONF][CONFIG_KEY_CHECKPOINT_ID],
            },
        )
        for (step, tid), entries in grouped.items():
            synth_tid = f"{step:08d}-{tid}"
            if self.checkpointer_put_writes_accepts_task_path:
                fut = self.submit(
                    self.checkpointer_put_writes,
                    anchor_write_config,
                    entries,
                    synth_tid,
                    "",
                )
            else:
                fut = self.submit(
                    self.checkpointer_put_writes,
                    anchor_write_config,
                    entries,
                    synth_tid,
                )
            if self._delta_write_futs is not None:
                self._delta_write_futs.append(fut)

    def _suppress_interrupt(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> bool | None:
        # persist current checkpoint and writes
        if self.durability == "exit" and (
            # if it's a top graph
            not self.is_nested
            # or a nested graph with error or interrupt
            or exc_value is not None
            # or a nested graph with checkpointer=True
            or all(NS_END not in part for part in self.checkpoint_ns)
        ):
            self._put_exit_delta_writes()
            self._put_checkpoint(self.checkpoint_metadata)
            self._put_pending_writes()
        # suppress interrupt
        if isinstance(exc_value, GraphInterrupt) and not self.is_nested:
            interrupt = exc_value
            interrupts = tuple(interrupt.args[0]) if interrupt.args else ()
            self._push_graph_lifecycle_event("interrupt", interrupts=interrupts)
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
            if not interrupt.args or not interrupt.args[0]:
                interrupt_payload = interrupt.args[0] if interrupt.args else ()
                self._emit(
                    "updates",
                    lambda: iter([{INTERRUPT: interrupt_payload}]),
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
                stream_modes = self.stream.modes if self.stream else []
                if "updates" in stream_modes:
                    self._emit("updates", lambda: iter(interrupts))
                if "values" in stream_modes:
                    current_values = read_channels(self.channels, self.output_keys)
                    # self.output_keys is a sequence, stream chunk contains entire state and interrupts
                    if isinstance(current_values, dict):
                        current_values[INTERRUPT] = interrupts[0][INTERRUPT]
                        self._emit("values", lambda: iter([current_values]))
                    # self.output_keys is a string, stream chunk contains only interrupts
                    else:
                        self._emit("values", lambda: iter(interrupts))
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
        durability: Durability,
        manager: None | AsyncParentRunManager | ParentRunManager = None,
        interrupt_after: All | Sequence[str] = EMPTY_SEQ,
        interrupt_before: All | Sequence[str] = EMPTY_SEQ,
        input_keys: str | Sequence[str] = EMPTY_SEQ,
        output_keys: str | Sequence[str] = EMPTY_SEQ,
        stream_keys: str | Sequence[str] = EMPTY_SEQ,
        migrate_checkpoint: Callable[[Checkpoint], None] | None = None,
        retry_policy: Sequence[RetryPolicy] = (),
        cache_policy: CachePolicy | None = None,
        has_graph_lifecycle_callbacks: bool = False,
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
            durability=durability,
            has_graph_lifecycle_callbacks=has_graph_lifecycle_callbacks,
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
        if self._delta_write_futs:
            futs, self._delta_write_futs = self._delta_write_futs, []
            concurrent.futures.wait(futs)
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

    def schedule_error_handler(
        self, failed_task: PregelExecutableTask, error: BaseException
    ) -> PregelExecutableTask | None:
        handler_node = self.nodes[failed_task.name].error_handler_node
        if not handler_node:
            return None
        # ensure error + ERROR_SOURCE_NODE writes are durable before handler runs
        if self._error_handler_write_futs:
            futs, self._error_handler_write_futs = self._error_handler_write_futs, []
            concurrent.futures.wait(futs)
        handler_task = prepare_node_error_handler_task(
            failed_task,
            handler_node_name=handler_node,
            failed_error=error,
            checkpoint=self.checkpoint,
            pending_writes=self.checkpoint_pending_writes,
            processes=self.nodes,
            channels=self.channels,
            managed=self.managed,
            config=failed_task.config,
            step=self.step,
            stop=self.stop,
            store=self.store,
            checkpointer=self.checkpointer,
            manager=self.manager,
            retry_policy=self.retry_policy,
            cache_policy=self.cache_policy,
        )
        if handler_task is None:
            return None
        self.tasks[handler_task.id] = handler_task
        if not self.is_replaying:
            self._reapply_writes_to_succeeded_nodes({handler_task.id: handler_task})
        for task in self.match_cached_writes():
            self.output_writes(task.id, task.writes, cached=True)
        return handler_task

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
        self._graph_lifecycle_events = deque()
        if not self.checkpointer:
            saved = None
        elif self.checkpoint_config[CONF].get(CONFIG_KEY_CHECKPOINT_ID):
            # Explicit checkpoint_id requested — fetch that exact checkpoint.
            # This covers both normal replay and subgraphs resolved via
            # checkpoint_map during time-travel.
            saved = self.checkpointer.get_tuple(self.checkpoint_config)
        elif replay_state := self.config[CONF].get(CONFIG_KEY_REPLAY_STATE):
            # Subgraph replay: the parent is replaying and passed us a
            # replay_state with its checkpoint_id. Look up our checkpoint
            # from the parent's checkpoint_map instead of fetching latest.
            saved = replay_state.get_checkpoint(
                self.config[CONF].get(CONFIG_KEY_CHECKPOINT_NS, ""),
                self.checkpointer,
                self.checkpoint_config,
            )
            # Clear RESUMING so _first re-applies input instead of resuming.
            # This recreates ephemeral routing channels so nodes trigger
            # naturally via version comparison.
            self.config[CONF].pop(CONFIG_KEY_RESUMING, None)
        else:
            # Normal case: fetch the most recent checkpoint for this
            # graph/thread. Returns None on first invocation.
            saved = self.checkpointer.get_tuple(self.checkpoint_config)

        # Capture before the synthetic-empty fallback below overwrites `saved`.
        # `_put_exit_delta_writes` uses this on first run (no persisted parent)
        # to lazy-create a stub instead of anchoring delta writes on a parent.
        self._has_persisted_parent = saved is not None
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
        self._initial_checkpoint_config = self.checkpoint_config
        self.prev_checkpoint_config = saved.parent_config
        self.checkpoint_id_saved = saved.checkpoint["id"]
        self.checkpoint = saved.checkpoint
        self.checkpoint_metadata = saved.metadata
        self.checkpoint_pending_writes = (
            [(str(tid), k, v) for tid, k, v in saved.pending_writes]
            if saved.pending_writes is not None
            else []
        )
        self._delta_write_futs = []
        self._error_handler_write_futs = []
        self._exit_delta_writes = (
            [] if self.durability == "exit" and self.checkpointer is not None else None
        )
        self.submit = self.stack.enter_context(BackgroundExecutor(self.config))
        self.channels, self.managed = channels_from_checkpoint(
            self.specs,
            self.checkpoint,
            saver=self.checkpointer,
            config=self.checkpoint_config,
        )
        self.stack.push(self._suppress_interrupt)
        self.status = "input"
        self.step = self.checkpoint_metadata["step"] + 1
        self.stop = self.step + self.config["recursion_limit"] + 1
        self.checkpoint_previous_versions = self.checkpoint["channel_versions"].copy()
        self.updated_channels = self._first(
            input_keys=self.input_keys,
            updated_channels=set(self.checkpoint.get("updated_channels"))  # type: ignore[arg-type]
            if self.checkpoint.get("updated_channels")
            else None,
        )

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
        durability: Durability,
        interrupt_after: All | Sequence[str] = EMPTY_SEQ,
        interrupt_before: All | Sequence[str] = EMPTY_SEQ,
        manager: None | AsyncParentRunManager | ParentRunManager = None,
        input_keys: str | Sequence[str] = EMPTY_SEQ,
        output_keys: str | Sequence[str] = EMPTY_SEQ,
        stream_keys: str | Sequence[str] = EMPTY_SEQ,
        migrate_checkpoint: Callable[[Checkpoint], None] | None = None,
        retry_policy: Sequence[RetryPolicy] = (),
        cache_policy: CachePolicy | None = None,
        has_graph_lifecycle_callbacks: bool = False,
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
            durability=durability,
            has_graph_lifecycle_callbacks=has_graph_lifecycle_callbacks,
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
        # Drain DeltaChannel write futures before committing the checkpoint so
        # ancestor walks never see a checkpoint without its backing writes.
        if self._delta_write_futs:
            futs, self._delta_write_futs = self._delta_write_futs, []
            await asyncio.gather(*futs)
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

    async def aschedule_error_handler(
        self, failed_task: PregelExecutableTask, error: BaseException
    ) -> PregelExecutableTask | None:
        handler_node = self.nodes[failed_task.name].error_handler_node
        if not handler_node:
            return None
        # ensure error + ERROR_SOURCE_NODE writes are durable before handler runs
        if self._error_handler_write_futs:
            futs, self._error_handler_write_futs = self._error_handler_write_futs, []
            await asyncio.gather(*futs)
        handler_task = prepare_node_error_handler_task(
            failed_task,
            handler_node_name=handler_node,
            failed_error=error,
            checkpoint=self.checkpoint,
            pending_writes=self.checkpoint_pending_writes,
            processes=self.nodes,
            channels=self.channels,
            managed=self.managed,
            config=failed_task.config,
            step=self.step,
            stop=self.stop,
            store=self.store,
            checkpointer=self.checkpointer,
            manager=self.manager,
            retry_policy=self.retry_policy,
            cache_policy=self.cache_policy,
        )
        if handler_task is None:
            return None
        self.tasks[handler_task.id] = handler_task
        if not self.is_replaying:
            self._reapply_writes_to_succeeded_nodes({handler_task.id: handler_task})
        for task in await self.amatch_cached_writes():
            self.output_writes(task.id, task.writes, cached=True)
        return handler_task

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
        self._graph_lifecycle_events = deque()
        if not self.checkpointer:
            saved = None
        elif self.checkpoint_config[CONF].get(CONFIG_KEY_CHECKPOINT_ID):
            # Explicit checkpoint_id requested — fetch that exact checkpoint.
            # This covers both normal replay and subgraphs resolved via
            # checkpoint_map during time-travel.
            saved = await self.checkpointer.aget_tuple(self.checkpoint_config)
        elif replay_state := self.config[CONF].get(CONFIG_KEY_REPLAY_STATE):
            # Subgraph replay: the parent is replaying and passed us a
            # replay_state with its checkpoint_id. Look up our checkpoint
            # from the parent's checkpoint_map instead of fetching latest.
            saved = await replay_state.aget_checkpoint(
                self.config[CONF].get(CONFIG_KEY_CHECKPOINT_NS, ""),
                self.checkpointer,
                self.checkpoint_config,
            )
            # Clear RESUMING so _first re-applies input instead of resuming.
            # This recreates ephemeral routing channels so nodes trigger
            # naturally via version comparison.
            self.config[CONF].pop(CONFIG_KEY_RESUMING, None)
        else:
            # Normal case: fetch the most recent checkpoint for this
            # graph/thread. Returns None on first invocation.
            saved = await self.checkpointer.aget_tuple(self.checkpoint_config)

        # Capture before the synthetic-empty fallback below overwrites `saved`.
        # `_put_exit_delta_writes` uses this on first run (no persisted parent)
        # to lazy-create a stub instead of anchoring delta writes on a parent.
        self._has_persisted_parent = saved is not None
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
        self._initial_checkpoint_config = self.checkpoint_config
        self.prev_checkpoint_config = saved.parent_config
        self.checkpoint_id_saved = saved.checkpoint["id"]
        self.checkpoint = saved.checkpoint
        self.checkpoint_metadata = saved.metadata
        self.checkpoint_pending_writes = (
            [(str(tid), k, v) for tid, k, v in saved.pending_writes]
            if saved.pending_writes is not None
            else []
        )
        self._delta_write_futs = []
        self._error_handler_write_futs = []
        self._exit_delta_writes = (
            [] if self.durability == "exit" and self.checkpointer is not None else None
        )
        self.submit = await self.stack.enter_async_context(
            AsyncBackgroundExecutor(self.config)
        )
        self.channels, self.managed = await achannels_from_checkpoint(
            self.specs,
            self.checkpoint,
            saver=self.checkpointer,
            config=self.checkpoint_config,
        )
        self.stack.push(self._suppress_interrupt)
        self.status = "input"
        self.step = self.checkpoint_metadata["step"] + 1
        self.stop = self.step + self.config["recursion_limit"] + 1
        self.checkpoint_previous_versions = self.checkpoint["channel_versions"].copy()
        self.updated_channels = self._first(
            input_keys=self.input_keys,
            updated_channels=set(self.checkpoint.get("updated_channels"))  # type: ignore[arg-type]
            if self.checkpoint.get("updated_channels")
            else None,
        )

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
