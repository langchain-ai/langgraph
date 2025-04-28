import binascii
import itertools
import sys
import threading
from collections import defaultdict, deque
from collections.abc import Iterable, Mapping, Sequence
from copy import copy
from functools import partial
from hashlib import sha1
from typing import (
    Any,
    Callable,
    Literal,
    NamedTuple,
    Optional,
    Protocol,
    Union,
    cast,
    overload,
)

from langchain_core.callbacks import Callbacks
from langchain_core.callbacks.manager import AsyncParentRunManager, ParentRunManager
from langchain_core.runnables.config import RunnableConfig
from xxhash import xxh3_128_hexdigest

from langgraph.channels.base import BaseChannel
from langgraph.checkpoint.base import (
    BaseCheckpointSaver,
    ChannelVersions,
    Checkpoint,
    PendingWrite,
    V,
)
from langgraph.constants import (
    CONF,
    CONFIG_KEY_CHECKPOINT_ID,
    CONFIG_KEY_CHECKPOINT_MAP,
    CONFIG_KEY_CHECKPOINT_NS,
    CONFIG_KEY_CHECKPOINTER,
    CONFIG_KEY_PREVIOUS,
    CONFIG_KEY_READ,
    CONFIG_KEY_RESUME_MAP,
    CONFIG_KEY_SCRATCHPAD,
    CONFIG_KEY_SEND,
    CONFIG_KEY_STORE,
    CONFIG_KEY_TASK_ID,
    EMPTY_SEQ,
    ERROR,
    INTERRUPT,
    MISSING,
    NO_WRITES,
    NS_END,
    NS_SEP,
    NULL_TASK_ID,
    PREVIOUS,
    PULL,
    PUSH,
    RESERVED,
    RESUME,
    RETURN,
    TAG_HIDDEN,
    TASKS,
    Send,
)
from langgraph.errors import InvalidUpdateError
from langgraph.managed.base import ManagedValueMapping
from langgraph.pregel.call import get_runnable_for_task
from langgraph.pregel.io import read_channels
from langgraph.pregel.log import logger
from langgraph.pregel.read import INPUT_CACHE_KEY_TYPE, PregelNode
from langgraph.store.base import BaseStore
from langgraph.types import (
    All,
    PregelExecutableTask,
    PregelScratchpad,
    PregelTask,
    RetryPolicy,
)
from langgraph.utils.config import merge_configs, patch_config

GetNextVersion = Callable[[Optional[V], BaseChannel], V]
SUPPORTS_EXC_NOTES = sys.version_info >= (3, 11)


class WritesProtocol(Protocol):
    """Protocol for objects containing writes to be applied to checkpoint.
    Implemented by PregelTaskWrites and PregelExecutableTask."""

    @property
    def path(self) -> tuple[Union[str, int, tuple], ...]: ...

    @property
    def name(self) -> str: ...

    @property
    def writes(self) -> Sequence[tuple[str, Any]]: ...

    @property
    def triggers(self) -> Sequence[str]: ...


class PregelTaskWrites(NamedTuple):
    """Simplest implementation of WritesProtocol, for usage with writes that
    don't originate from a runnable task, eg. graph input, update_state, etc."""

    path: tuple[Union[str, int, tuple], ...]
    name: str
    writes: Sequence[tuple[str, Any]]
    triggers: Sequence[str]


class Call:
    __slots__ = ("func", "input", "retry", "callbacks")

    func: Callable
    input: Any
    retry: Optional[Sequence[RetryPolicy]]
    callbacks: Callbacks

    def __init__(
        self,
        func: Callable,
        input: Any,
        *,
        retry: Optional[Sequence[RetryPolicy]],
        callbacks: Callbacks,
    ) -> None:
        self.func = func
        self.input = input
        self.retry = retry
        self.callbacks = callbacks


def should_interrupt(
    checkpoint: Checkpoint,
    interrupt_nodes: Union[All, Sequence[str]],
    tasks: Iterable[PregelExecutableTask],
) -> list[PregelExecutableTask]:
    """Check if the graph should be interrupted based on current state."""
    version_type = type(next(iter(checkpoint["channel_versions"].values()), None))
    null_version = version_type()  # type: ignore[misc]
    seen = checkpoint["versions_seen"].get(INTERRUPT, {})
    # interrupt if any channel has been updated since last interrupt
    any_updates_since_prev_interrupt = any(
        version > seen.get(chan, null_version)  # type: ignore[operator]
        for chan, version in checkpoint["channel_versions"].items()
    )
    # and any triggered node is in interrupt_nodes list
    return (
        [
            task
            for task in tasks
            if (
                (
                    not task.config
                    or TAG_HIDDEN not in task.config.get("tags", EMPTY_SEQ)
                )
                if interrupt_nodes == "*"
                else task.name in interrupt_nodes
            )
        ]
        if any_updates_since_prev_interrupt
        else []
    )


def local_read(
    channels: Mapping[str, BaseChannel],
    managed: ManagedValueMapping,
    task: WritesProtocol,
    select: Union[list[str], str],
    fresh: bool = False,
) -> Union[dict[str, Any], Any]:
    """Function injected under CONFIG_KEY_READ in task config, to read current state.
    Used by conditional edges to read a copy of the state with reflecting the writes
    from that node only."""
    updated: dict[str, list[Any]] = defaultdict(list)
    if isinstance(select, str):
        managed_keys = []
        for c, v in task.writes:
            if c == select:
                updated[c].append(v)
    else:
        managed_keys = [k for k in select if k in managed]
        select = [k for k in select if k not in managed]
        for c, v in task.writes:
            if c in select:
                updated[c].append(v)
    if fresh and updated:
        # apply writes
        local_channels: dict[str, BaseChannel] = {}
        for k in channels:
            if k in updated:
                cc = channels[k].copy()
                cc.update(updated[k])
            else:
                cc = channels[k]
            local_channels[k] = cc
        # read fresh values
        values = read_channels(local_channels, select)
    else:
        values = read_channels(channels, select)
    if managed_keys:
        values.update({k: managed[k]() for k in managed_keys})
    return values


def local_write(
    commit: Callable[[Sequence[tuple[str, Any]]], None],
    process_keys: Iterable[str],
    writes: Sequence[tuple[str, Any]],
) -> None:
    """Function injected under CONFIG_KEY_SEND in task config, to write to channels.
    Validates writes and forwards them to `commit` function."""
    for chan, value in writes:
        if chan in (PUSH, TASKS) and value is not None:
            if not isinstance(value, Send):
                raise InvalidUpdateError(f"Expected Send, got {value}")
            if value.node not in process_keys:
                raise InvalidUpdateError(f"Invalid node name {value.node} in packet")
    commit(writes)


def increment(current: Optional[int], channel: BaseChannel) -> int:
    """Default channel versioning function, increments the current int version."""
    return current + 1 if current is not None else 1


def apply_writes(
    checkpoint: Checkpoint,
    channels: Mapping[str, BaseChannel],
    tasks: Iterable[WritesProtocol],
    get_next_version: Optional[GetNextVersion],
) -> tuple[dict[str, list[Any]], set[str]]:
    """Apply writes from a set of tasks (usually the tasks from a Pregel step)
    to the checkpoint and channels, and return managed values writes to be applied
    externally.

    Args:
        checkpoint: The checkpoint to update.
        channels: The channels to update.
        tasks: The tasks to apply writes from.
        get_next_version: Optional function to determine the next version of a channel.

    Returns:
        A tuple containing the managed values writes to be applied externally, and
        the set of channels that were updated in this step.
    """
    # sort tasks on path, to ensure deterministic order for update application
    # any path parts after the 3rd are ignored for sorting
    # (we use them for eg. task ids which aren't good for sorting)
    tasks = sorted(tasks, key=lambda t: task_path_str(t.path[:3]))
    # if no task has triggers this is applying writes from the null task only
    # so we don't do anything other than update the channels written to
    bump_step = any(t.triggers for t in tasks)

    # update seen versions
    for task in tasks:
        checkpoint["versions_seen"].setdefault(task.name, {}).update(
            {
                chan: checkpoint["channel_versions"][chan]
                for chan in task.triggers
                if chan in checkpoint["channel_versions"]
            }
        )

    # Find the highest version of all channels
    if checkpoint["channel_versions"]:
        max_version = max(checkpoint["channel_versions"].values())
    else:
        max_version = None

    # Consume all channels that were read
    for chan in {
        chan
        for task in tasks
        for chan in task.triggers
        if chan not in RESERVED and chan in channels
    }:
        if channels[chan].consume() and get_next_version is not None:
            checkpoint["channel_versions"][chan] = get_next_version(
                max_version,
                channels[chan],
            )

    # clear pending sends
    if checkpoint["pending_sends"] and bump_step:
        checkpoint["pending_sends"].clear()

    # Group writes by channel
    pending_writes_by_channel: dict[str, list[Any]] = defaultdict(list)
    pending_writes_by_managed: dict[str, list[Any]] = defaultdict(list)
    for task in tasks:
        for chan, val in task.writes:
            if chan in (NO_WRITES, PUSH, RESUME, INTERRUPT, RETURN, ERROR):
                pass
            elif chan == TASKS:
                checkpoint["pending_sends"].append(val)
            elif chan in channels:
                pending_writes_by_channel[chan].append(val)
            else:
                pending_writes_by_managed[chan].append(val)

    # Find the highest version of all channels
    if checkpoint["channel_versions"]:
        max_version = max(checkpoint["channel_versions"].values())
    else:
        max_version = None

    # Apply writes to channels
    updated_channels: set[str] = set()
    for chan, vals in pending_writes_by_channel.items():
        if chan in channels:
            if channels[chan].update(vals) and get_next_version is not None:
                checkpoint["channel_versions"][chan] = get_next_version(
                    max_version,
                    channels[chan],
                )
            updated_channels.add(chan)

    # Channels that weren't updated in this step are notified of a new step
    if bump_step:
        for chan in channels:
            if channels[chan].is_available() and chan not in updated_channels:
                if channels[chan].update(EMPTY_SEQ) and get_next_version is not None:
                    checkpoint["channel_versions"][chan] = get_next_version(
                        max_version,
                        channels[chan],
                    )
    # Return managed values writes to be applied externally
    return pending_writes_by_managed, updated_channels


def has_next_tasks(
    trigger_to_nodes: Mapping[str, Sequence[str]],
    updated_channels: set[str],
    checkpoint: Checkpoint,
) -> bool:
    """Check if there are any tasks that should be run in the next step."""
    return bool(checkpoint["pending_sends"]) or not updated_channels.isdisjoint(
        trigger_to_nodes
    )


@overload
def prepare_next_tasks(
    checkpoint: Checkpoint,
    pending_writes: list[PendingWrite],
    processes: Mapping[str, PregelNode],
    channels: Mapping[str, BaseChannel],
    managed: ManagedValueMapping,
    config: RunnableConfig,
    step: int,
    *,
    for_execution: Literal[False],
    store: Literal[None] = None,
    checkpointer: Literal[None] = None,
    manager: Literal[None] = None,
    trigger_to_nodes: Optional[Mapping[str, Sequence[str]]] = None,
    updated_channels: Optional[set[str]] = None,
) -> dict[str, PregelTask]: ...


@overload
def prepare_next_tasks(
    checkpoint: Checkpoint,
    pending_writes: list[PendingWrite],
    processes: Mapping[str, PregelNode],
    channels: Mapping[str, BaseChannel],
    managed: ManagedValueMapping,
    config: RunnableConfig,
    step: int,
    *,
    for_execution: Literal[True],
    store: Optional[BaseStore],
    checkpointer: Optional[BaseCheckpointSaver],
    manager: Union[None, ParentRunManager, AsyncParentRunManager],
    trigger_to_nodes: Optional[Mapping[str, Sequence[str]]] = None,
    updated_channels: Optional[set[str]] = None,
) -> dict[str, PregelExecutableTask]: ...


def prepare_next_tasks(
    checkpoint: Checkpoint,
    pending_writes: list[PendingWrite],
    processes: Mapping[str, PregelNode],
    channels: Mapping[str, BaseChannel],
    managed: ManagedValueMapping,
    config: RunnableConfig,
    step: int,
    *,
    for_execution: bool,
    store: Optional[BaseStore] = None,
    checkpointer: Optional[BaseCheckpointSaver] = None,
    manager: Union[None, ParentRunManager, AsyncParentRunManager] = None,
    trigger_to_nodes: Optional[Mapping[str, Sequence[str]]] = None,
    updated_channels: Optional[set[str]] = None,
) -> Union[dict[str, PregelTask], dict[str, PregelExecutableTask]]:
    """Prepare the set of tasks that will make up the next Pregel step.

    Args:
        checkpoint: The current checkpoint.
        pending_writes: The list of pending writes.
        processes: The mapping of process names to PregelNode instances.
        channels: The mapping of channel names to BaseChannel instances.
        managed: The mapping of managed value names to functions.
        config: The runnable configuration.
        step: The current step.
        for_execution: Whether the tasks are being prepared for execution.
        store: An instance of BaseStore to make it available for usage within tasks.
        checkpointer: Checkpointer instance used for saving checkpoints.
        manager: The parent run manager to use for the tasks.
        trigger_to_nodes: Optional: Mapping of channel names to the set of nodes
            that are can be triggered by that channel.
        updated_channels: Optional. Set of channel names that have been updated during
            the previous step. Using in conjunction with trigger_to_nodes to speed
            up the process of determining which nodes should be triggered in the next
            step.

    Returns:
        A dictionary of tasks to be executed. The keys are the task ids and the values
        are the tasks themselves. This is the union of all PUSH tasks (Sends)
        and PULL tasks (nodes triggered by edges).
    """
    input_cache: dict[INPUT_CACHE_KEY_TYPE, Any] = {}
    checkpoint_id_bytes = binascii.unhexlify(checkpoint["id"].replace("-", ""))
    null_version = checkpoint_null_version(checkpoint)
    tasks: list[Union[PregelTask, PregelExecutableTask]] = []
    # Consume pending_sends from previous step
    for idx, _ in enumerate(checkpoint["pending_sends"]):
        if task := prepare_single_task(
            (PUSH, idx),
            None,
            checkpoint=checkpoint,
            checkpoint_id_bytes=checkpoint_id_bytes,
            checkpoint_null_version=null_version,
            pending_writes=pending_writes,
            processes=processes,
            channels=channels,
            managed=managed,
            config=config,
            step=step,
            for_execution=for_execution,
            store=store,
            checkpointer=checkpointer,
            manager=manager,
            input_cache=input_cache,
        ):
            tasks.append(task)

    # This section is an optimization that allows which nodes will be active
    # during the next step.
    # When there's information about:
    # 1. Which channels were updated in the previous step
    # 2. Which nodes are triggered by which channels
    # Then we can determine which nodes should be triggered in the next step
    # without having to cycle through all nodes.
    if updated_channels and trigger_to_nodes:
        triggered_nodes: set[str] = set()
        # Get all nodes that have triggers associated with an updated channel
        for channel in updated_channels:
            if node_ids := trigger_to_nodes.get(channel):
                triggered_nodes.update(node_ids)
        # Sort the nodes to ensure deterministic order
        candidate_nodes: Iterable[str] = sorted(triggered_nodes)
    elif not checkpoint["channel_versions"]:
        candidate_nodes = ()
    else:
        candidate_nodes = processes.keys()

    # Check if any processes should be run in next step
    # If so, prepare the values to be passed to them
    for name in candidate_nodes:
        if task := prepare_single_task(
            (PULL, name),
            None,
            checkpoint=checkpoint,
            checkpoint_id_bytes=checkpoint_id_bytes,
            checkpoint_null_version=null_version,
            pending_writes=pending_writes,
            processes=processes,
            channels=channels,
            managed=managed,
            config=config,
            step=step,
            for_execution=for_execution,
            store=store,
            checkpointer=checkpointer,
            manager=manager,
            input_cache=input_cache,
        ):
            tasks.append(task)
    return {t.id: t for t in tasks}


PUSH_TRIGGER = (PUSH,)


def prepare_single_task(
    task_path: tuple[Any, ...],
    task_id_checksum: Optional[str],
    *,
    checkpoint: Checkpoint,
    checkpoint_id_bytes: bytes,
    checkpoint_null_version: Optional[V],
    pending_writes: list[PendingWrite],
    processes: Mapping[str, PregelNode],
    channels: Mapping[str, BaseChannel],
    managed: ManagedValueMapping,
    config: RunnableConfig,
    step: int,
    for_execution: bool,
    store: Optional[BaseStore] = None,
    checkpointer: Optional[BaseCheckpointSaver] = None,
    manager: Union[None, ParentRunManager, AsyncParentRunManager] = None,
    input_cache: Optional[dict[INPUT_CACHE_KEY_TYPE, Any]] = None,
) -> Union[None, PregelTask, PregelExecutableTask]:
    """Prepares a single task for the next Pregel step, given a task path, which
    uniquely identifies a PUSH or PULL task within the graph."""
    configurable = config.get(CONF, {})
    parent_ns = configurable.get(CONFIG_KEY_CHECKPOINT_NS, "")
    task_id_func = _xxhash_str if checkpoint["v"] > 1 else _uuid5_str

    if task_path[0] == PUSH and isinstance(task_path[-1], Call):
        # (PUSH, parent task path, idx of PUSH write, id of parent task, Call)
        task_path_t = cast(tuple[str, tuple, int, str, Call], task_path)
        call = task_path_t[-1]
        proc_ = get_runnable_for_task(call.func)
        name = proc_.name
        if name is None:
            raise ValueError("`call` functions must have a `__name__` attribute")
        # create task id
        triggers: Sequence[str] = PUSH_TRIGGER
        checkpoint_ns = f"{parent_ns}{NS_SEP}{name}" if parent_ns else name
        task_id = task_id_func(
            checkpoint_id_bytes,
            checkpoint_ns,
            str(step),
            name,
            PUSH,
            task_path_str(task_path[1]),
            str(task_path[2]),
        )
        task_checkpoint_ns = f"{checkpoint_ns}:{task_id}"
        # we append True to the task path to indicate that a call is being
        # made, so we should not return interrupts from this task (responsibility lies with the parent)
        task_path = (*task_path[:3], True)
        metadata = {
            "langgraph_step": step,
            "langgraph_node": name,
            "langgraph_triggers": triggers,
            "langgraph_path": task_path,
            "langgraph_checkpoint_ns": task_checkpoint_ns,
        }
        if task_id_checksum is not None:
            assert task_id == task_id_checksum, f"{task_id} != {task_id_checksum}"
        if for_execution:
            writes: deque[tuple[str, Any]] = deque()
            return PregelExecutableTask(
                name,
                call.input,
                proc_,
                writes,
                patch_config(
                    merge_configs(config, {"metadata": metadata}),
                    run_name=name,
                    callbacks=call.callbacks
                    or (manager.get_child(f"graph:step:{step}") if manager else None),
                    configurable={
                        CONFIG_KEY_TASK_ID: task_id,
                        # deque.extend is thread-safe
                        CONFIG_KEY_SEND: partial(
                            local_write,
                            writes.extend,
                            processes.keys(),
                        ),
                        CONFIG_KEY_READ: partial(
                            local_read,
                            channels,
                            managed,
                            PregelTaskWrites(task_path, name, writes, triggers),
                        ),
                        CONFIG_KEY_STORE: (store or configurable.get(CONFIG_KEY_STORE)),
                        CONFIG_KEY_CHECKPOINTER: (
                            checkpointer or configurable.get(CONFIG_KEY_CHECKPOINTER)
                        ),
                        CONFIG_KEY_CHECKPOINT_MAP: {
                            **configurable.get(CONFIG_KEY_CHECKPOINT_MAP, {}),
                            parent_ns: checkpoint["id"],
                        },
                        CONFIG_KEY_CHECKPOINT_ID: None,
                        CONFIG_KEY_CHECKPOINT_NS: task_checkpoint_ns,
                        CONFIG_KEY_SCRATCHPAD: _scratchpad(
                            config[CONF].get(CONFIG_KEY_SCRATCHPAD),
                            pending_writes,
                            task_id,
                            xxh3_128_hexdigest(task_checkpoint_ns.encode()),
                            config[CONF].get(CONFIG_KEY_RESUME_MAP),
                        ),
                    },
                ),
                triggers,
                call.retry,
                None,
                task_id,
                task_path,
            )
        else:
            return PregelTask(task_id, name, task_path)
    elif task_path[0] == PUSH:
        if len(task_path) == 2:
            # SEND tasks, executed in superstep n+1
            # (PUSH, idx of pending send)
            idx = cast(int, task_path[1])
            if idx >= len(checkpoint["pending_sends"]):
                return
            packet = checkpoint["pending_sends"][idx]
            if not isinstance(packet, Send):
                logger.warning(
                    f"Ignoring invalid packet type {type(packet)} in pending sends"
                )
                return
            if packet.node not in processes:
                logger.warning(
                    f"Ignoring unknown node name {packet.node} in pending sends"
                )
                return
            # create task id
            triggers = PUSH_TRIGGER
            checkpoint_ns = (
                f"{parent_ns}{NS_SEP}{packet.node}" if parent_ns else packet.node
            )
            task_id = task_id_func(
                checkpoint_id_bytes,
                checkpoint_ns,
                str(step),
                packet.node,
                PUSH,
                str(idx),
            )
        else:
            logger.warning(f"Ignoring invalid PUSH task path {task_path}")
            return
        task_checkpoint_ns = f"{checkpoint_ns}:{task_id}"
        # we append False to the task path to indicate that a call is not being made
        # so we should return interrupts from this task
        task_path = (*task_path[:3], False)
        metadata = {
            "langgraph_step": step,
            "langgraph_node": packet.node,
            "langgraph_triggers": triggers,
            "langgraph_path": task_path,
            "langgraph_checkpoint_ns": task_checkpoint_ns,
        }
        if task_id_checksum is not None:
            assert task_id == task_id_checksum, f"{task_id} != {task_id_checksum}"
        if for_execution:
            proc = processes[packet.node]
            if node := proc.node:
                if proc.metadata:
                    metadata.update(proc.metadata)
                writes = deque()
                return PregelExecutableTask(
                    packet.node,
                    packet.arg,
                    node,
                    writes,
                    patch_config(
                        merge_configs(
                            config, {"metadata": metadata, "tags": proc.tags}
                        ),
                        run_name=packet.node,
                        callbacks=(
                            manager.get_child(f"graph:step:{step}") if manager else None
                        ),
                        configurable={
                            CONFIG_KEY_TASK_ID: task_id,
                            # deque.extend is thread-safe
                            CONFIG_KEY_SEND: partial(
                                local_write,
                                writes.extend,
                                processes.keys(),
                            ),
                            CONFIG_KEY_READ: partial(
                                local_read,
                                channels,
                                managed,
                                PregelTaskWrites(
                                    task_path, packet.node, writes, triggers
                                ),
                            ),
                            CONFIG_KEY_STORE: (
                                store or configurable.get(CONFIG_KEY_STORE)
                            ),
                            CONFIG_KEY_CHECKPOINTER: (
                                checkpointer
                                or configurable.get(CONFIG_KEY_CHECKPOINTER)
                            ),
                            CONFIG_KEY_CHECKPOINT_MAP: {
                                **configurable.get(CONFIG_KEY_CHECKPOINT_MAP, {}),
                                parent_ns: checkpoint["id"],
                            },
                            CONFIG_KEY_CHECKPOINT_ID: None,
                            CONFIG_KEY_CHECKPOINT_NS: task_checkpoint_ns,
                            CONFIG_KEY_SCRATCHPAD: _scratchpad(
                                config[CONF].get(CONFIG_KEY_SCRATCHPAD),
                                pending_writes,
                                task_id,
                                xxh3_128_hexdigest(task_checkpoint_ns.encode()),
                                config[CONF].get(CONFIG_KEY_RESUME_MAP),
                            ),
                            CONFIG_KEY_PREVIOUS: checkpoint["channel_values"].get(
                                PREVIOUS, None
                            ),
                        },
                    ),
                    triggers,
                    proc.retry_policy,
                    None,
                    task_id,
                    task_path,
                    writers=proc.flat_writers,
                    subgraphs=proc.subgraphs,
                )
        else:
            return PregelTask(task_id, packet.node, task_path)
    elif task_path[0] == PULL:
        # (PULL, node name)
        name = cast(str, task_path[1])
        if name not in processes:
            return
        proc = processes[name]
        if checkpoint_null_version is None:
            return
        # If any of the channels read by this process were updated
        if _triggers(
            channels,
            checkpoint["channel_versions"],
            checkpoint["versions_seen"].get(name),
            checkpoint_null_version,
            proc,
        ):
            triggers = tuple(sorted(proc.triggers))
            try:
                val = _proc_input(
                    proc,
                    managed,
                    channels,
                    for_execution=for_execution,
                    input_cache=input_cache,
                )
                if val is MISSING:
                    return
            except Exception as exc:
                if SUPPORTS_EXC_NOTES:
                    exc.add_note(
                        f"Before task with name '{name}' and path '{task_path[:3]}'"
                    )
                raise

            # create task id
            checkpoint_ns = f"{parent_ns}{NS_SEP}{name}" if parent_ns else name
            task_id = task_id_func(
                checkpoint_id_bytes,
                checkpoint_ns,
                str(step),
                name,
                PULL,
                *triggers,
            )
            task_checkpoint_ns = f"{checkpoint_ns}{NS_END}{task_id}"
            metadata = {
                "langgraph_step": step,
                "langgraph_node": name,
                "langgraph_triggers": triggers,
                "langgraph_path": task_path[:3],
                "langgraph_checkpoint_ns": task_checkpoint_ns,
            }
            if task_id_checksum is not None:
                assert task_id == task_id_checksum, f"{task_id} != {task_id_checksum}"
            if for_execution:
                if node := proc.node:
                    if proc.metadata:
                        metadata.update(proc.metadata)
                    writes = deque()
                    return PregelExecutableTask(
                        name,
                        val,
                        node,
                        writes,
                        patch_config(
                            merge_configs(
                                config, {"metadata": metadata, "tags": proc.tags}
                            ),
                            run_name=name,
                            callbacks=(
                                manager.get_child(f"graph:step:{step}")
                                if manager
                                else None
                            ),
                            configurable={
                                CONFIG_KEY_TASK_ID: task_id,
                                # deque.extend is thread-safe
                                CONFIG_KEY_SEND: partial(
                                    local_write,
                                    writes.extend,
                                    tuple(processes.keys()),
                                ),
                                CONFIG_KEY_READ: partial(
                                    local_read,
                                    channels,
                                    managed,
                                    PregelTaskWrites(
                                        task_path[:3],
                                        name,
                                        writes,
                                        triggers,
                                    ),
                                ),
                                CONFIG_KEY_STORE: (
                                    store or configurable.get(CONFIG_KEY_STORE)
                                ),
                                CONFIG_KEY_CHECKPOINTER: (
                                    checkpointer
                                    or configurable.get(CONFIG_KEY_CHECKPOINTER)
                                ),
                                CONFIG_KEY_CHECKPOINT_MAP: {
                                    **configurable.get(CONFIG_KEY_CHECKPOINT_MAP, {}),
                                    parent_ns: checkpoint["id"],
                                },
                                CONFIG_KEY_CHECKPOINT_ID: None,
                                CONFIG_KEY_CHECKPOINT_NS: task_checkpoint_ns,
                                CONFIG_KEY_SCRATCHPAD: _scratchpad(
                                    config[CONF].get(CONFIG_KEY_SCRATCHPAD),
                                    pending_writes,
                                    task_id,
                                    xxh3_128_hexdigest(task_checkpoint_ns.encode()),
                                    config[CONF].get(CONFIG_KEY_RESUME_MAP),
                                ),
                                CONFIG_KEY_PREVIOUS: checkpoint["channel_values"].get(
                                    PREVIOUS, None
                                ),
                            },
                        ),
                        triggers,
                        proc.retry_policy,
                        None,
                        task_id,
                        task_path[:3],
                        writers=proc.flat_writers,
                        subgraphs=proc.subgraphs,
                    )
            else:
                return PregelTask(task_id, name, task_path[:3])


def checkpoint_null_version(
    checkpoint: Checkpoint,
) -> Optional[V]:
    """Get the null version for the checkpoint, if available."""
    for version in checkpoint["channel_versions"].values():
        return type(version)()
    return None


def _triggers(
    channels: Mapping[str, BaseChannel],
    versions: ChannelVersions,
    seen: Optional[ChannelVersions],
    null_version: V,
    proc: PregelNode,
) -> Sequence[str]:
    if seen is None:
        for chan in proc.triggers:
            if channels[chan].is_available():
                return (chan,)
    else:
        for chan in proc.triggers:
            if channels[chan].is_available() and versions.get(  # type: ignore[operator]
                chan, null_version
            ) > seen.get(chan, null_version):
                return (chan,)
    return EMPTY_SEQ


def _scratchpad(
    parent_scratchpad: Optional[PregelScratchpad],
    pending_writes: list[PendingWrite],
    task_id: str,
    namespace_hash: str,
    resume_map: Optional[dict[str, Any]],
) -> PregelScratchpad:
    if len(pending_writes) > 0:
        # find global resume value
        for w in pending_writes:
            if w[0] == NULL_TASK_ID and w[1] == RESUME:
                null_resume_write = w
                break
        else:
            # None cannot be used as a resume value, because it would be difficult to
            # distinguish from missing when used over http
            null_resume_write = None

        # find task-specific resume value
        for w in pending_writes:
            if w[0] == task_id and w[1] == RESUME:
                task_resume_write = w[2]
                if not isinstance(task_resume_write, list):
                    task_resume_write = [task_resume_write]
                break
        else:
            task_resume_write = []
        del w

        # find namespace and task-specific resume value
        if resume_map and namespace_hash in resume_map:
            mapped_resume_write = resume_map[namespace_hash]
            task_resume_write.append(mapped_resume_write)

    else:
        null_resume_write = None
        task_resume_write = []

    def get_null_resume(consume: bool = False) -> Any:
        if null_resume_write is None:
            if parent_scratchpad is not None:
                return parent_scratchpad.get_null_resume(consume)
            return None
        if consume:
            try:
                pending_writes.remove(null_resume_write)
                return null_resume_write[2]
            except ValueError:
                return None
        return null_resume_write[2]

    # using itertools.count as an atomic counter (+= 1 is not thread-safe)
    return PregelScratchpad(
        # call
        call_counter=LazyAtomicCounter(),
        # interrupt
        interrupt_counter=LazyAtomicCounter(),
        resume=task_resume_write,
        get_null_resume=get_null_resume,
        # subgraph
        subgraph_counter=LazyAtomicCounter(),
    )


def _proc_input(
    proc: PregelNode,
    managed: ManagedValueMapping,
    channels: Mapping[str, BaseChannel],
    *,
    for_execution: bool,
    input_cache: Optional[dict[INPUT_CACHE_KEY_TYPE, Any]],
) -> Any:
    """Prepare input for a PULL task, based on the process's channels and triggers."""
    # if in cache return shallow copy
    if input_cache is not None and proc.input_cache_key in input_cache:
        return copy(input_cache[proc.input_cache_key])
    # If all trigger channels subscribed by this process are not empty
    # then invoke the process with the values of all non-empty channels
    if isinstance(proc.channels, dict):
        val: dict[str, Any] = {}
        for k, chan in proc.channels.items():
            if chan in channels:
                if channels[chan].is_available():
                    val[k] = channels[chan].get()
            else:
                val[k] = managed[k]()
    elif isinstance(proc.channels, list):
        for chan in proc.channels:
            if chan in channels:
                if channels[chan].is_available():
                    val = channels[chan].get()
                    break
            else:
                val[k] = managed[k]()
        else:
            return MISSING
    else:
        raise RuntimeError(
            "Invalid channels type, expected list or dict, got {proc.channels}"
        )

    # If the process has a mapper, apply it to the value
    if for_execution and proc.mapper is not None:
        val = proc.mapper(val)

    # Cache the input value
    if input_cache is not None:
        input_cache[proc.input_cache_key] = val

    return val


def _uuid5_str(namespace: bytes, *parts: str) -> str:
    """Generate a UUID from the SHA-1 hash of a namespace and str parts."""

    sha = sha1(namespace, usedforsecurity=False)
    sha.update(b"".join(p.encode() for p in parts))
    hex = sha.hexdigest()
    return f"{hex[:8]}-{hex[8:12]}-{hex[12:16]}-{hex[16:20]}-{hex[20:32]}"


def _xxhash_str(namespace: bytes, *parts: str) -> str:
    """Generate a UUID from the XXH3 hash of a namespace and str parts."""
    hex = xxh3_128_hexdigest(namespace + b"".join(p.encode() for p in parts))
    return f"{hex[:8]}-{hex[8:12]}-{hex[12:16]}-{hex[16:20]}-{hex[20:32]}"


def task_path_str(tup: Union[str, int, tuple]) -> str:
    """Generate a string representation of the task path."""
    return (
        f"~{', '.join(task_path_str(x) for x in tup)}"
        if isinstance(tup, (tuple, list))
        else f"{tup:010d}"
        if isinstance(tup, int)
        else str(tup)
    )


LAZY_ATOMIC_COUNTER_LOCK = threading.Lock()


class LazyAtomicCounter:
    __slots__ = ("_counter",)

    _counter: Optional[Callable[[], int]]

    def __init__(self) -> None:
        self._counter = None

    def __call__(self) -> int:
        if self._counter is None:
            with LAZY_ATOMIC_COUNTER_LOCK:
                if self._counter is None:
                    self._counter = itertools.count(0).__next__
        return self._counter()
