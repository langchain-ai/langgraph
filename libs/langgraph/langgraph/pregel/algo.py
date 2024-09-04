from collections import defaultdict, deque
from functools import partial
from typing import (
    Any,
    Callable,
    Iterator,
    Literal,
    Mapping,
    NamedTuple,
    Optional,
    Protocol,
    Sequence,
    Union,
    overload,
)
from uuid import UUID, uuid5

from langchain_core.callbacks.manager import AsyncParentRunManager, ParentRunManager
from langchain_core.runnables.config import RunnableConfig

from langgraph.channels.base import BaseChannel
from langgraph.checkpoint.base import (
    BaseCheckpointSaver,
    Checkpoint,
    copy_checkpoint,
    create_checkpoint,
)
from langgraph.constants import (
    CONFIG_KEY_CHECKPOINT_MAP,
    CONFIG_KEY_CHECKPOINTER,
    CONFIG_KEY_READ,
    CONFIG_KEY_RESUMING,
    CONFIG_KEY_SEND,
    CONFIG_KEY_TASK_ID,
    INTERRUPT,
    NS_SEP,
    RESERVED,
    TAG_HIDDEN,
    TASKS,
    Send,
)
from langgraph.errors import EmptyChannelError, InvalidUpdateError
from langgraph.managed.base import ManagedValueMapping
from langgraph.pregel.io import read_channel, read_channels
from langgraph.pregel.log import logger
from langgraph.pregel.manager import ChannelsManager
from langgraph.pregel.read import PregelNode
from langgraph.pregel.types import All, PregelExecutableTask, PregelTask
from langgraph.utils.config import merge_configs, patch_config


class WritesProtocol(Protocol):
    name: str
    writes: Sequence[tuple[str, Any]]
    triggers: Sequence[str]


class PregelTaskWrites(NamedTuple):
    name: str
    writes: Sequence[tuple[str, Any]]
    triggers: Sequence[str]


def should_interrupt(
    checkpoint: Checkpoint,
    interrupt_nodes: Union[All, Sequence[str]],
    tasks: list[PregelExecutableTask],
) -> list[PregelExecutableTask]:
    version_type = type(next(iter(checkpoint["channel_versions"].values()), None))
    null_version = version_type()
    seen = checkpoint["versions_seen"].get(INTERRUPT, {})
    # interrupt if any channel has been updated since last interrupt
    any_updates_since_prev_interrupt = any(
        version > seen.get(chan, null_version)
        for chan, version in checkpoint["channel_versions"].items()
    )
    # and any triggered node is in interrupt_nodes list
    return (
        [
            task
            for task in tasks
            if (
                (not task.config or TAG_HIDDEN not in task.config.get("tags"))
                if interrupt_nodes == "*"
                else task.name in interrupt_nodes
            )
        ]
        if any_updates_since_prev_interrupt
        else []
    )


def local_read(
    step: int,
    checkpoint: Checkpoint,
    channels: Mapping[str, BaseChannel],
    managed: ManagedValueMapping,
    task: WritesProtocol,
    config: RunnableConfig,
    select: Union[list[str], str],
    fresh: bool = False,
) -> Union[dict[str, Any], Any]:
    if isinstance(select, str):
        managed_keys = []
        for c, _ in task.writes:
            if c == select:
                updated = {c}
                break
        else:
            updated = set()
    else:
        managed_keys = [k for k in select if k in managed]
        select = [k for k in select if k not in managed]
        updated = set(select).intersection(c for c, _ in task.writes)
    if fresh and updated:
        with ChannelsManager(
            {k: v for k, v in channels.items() if k in updated},
            checkpoint,
            config,
            skip_context=True,
        ) as (local_channels, _):
            apply_writes(copy_checkpoint(checkpoint), local_channels, [task], None)
            values = read_channels({**channels, **local_channels}, select)
    else:
        values = read_channels(channels, select)
    if managed_keys:
        values.update({k: managed[k](step) for k in managed_keys})
    return values


def local_write(
    step: int,
    commit: Callable[[Sequence[tuple[str, Any]]], None],
    processes: Mapping[str, PregelNode],
    channels: Mapping[str, BaseChannel],
    managed: ManagedValueMapping,
    writes: Sequence[tuple[str, Any]],
) -> None:
    for chan, value in writes:
        if chan == TASKS:
            if not isinstance(value, Send):
                raise InvalidUpdateError(
                    f"Invalid packet type, expected Packet, got {value}"
                )
            if value.node not in processes:
                raise InvalidUpdateError(f"Invalid node name {value.node} in packet")
            # replace any runtime values with placeholders
            managed.replace_runtime_values(step, value.arg)
        elif chan not in channels and chan not in managed:
            logger.warning(f"Skipping write for channel '{chan}' which has no readers")
    commit(writes)


def increment(current: Optional[int], channel: BaseChannel) -> int:
    return current + 1 if current is not None else 1


def apply_writes(
    checkpoint: Checkpoint,
    channels: Mapping[str, BaseChannel],
    tasks: Sequence[WritesProtocol],
    get_next_version: Optional[Callable[[int, BaseChannel], int]],
) -> dict[str, list[Any]]:
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
        if channels[chan].consume():
            if get_next_version is not None:
                checkpoint["channel_versions"][chan] = get_next_version(
                    max_version, channels[chan]
                )

    # clear pending sends
    if checkpoint["pending_sends"]:
        checkpoint["pending_sends"].clear()

    # Group writes by channel
    pending_writes_by_channel: dict[str, list[Any]] = defaultdict(list)
    pending_writes_by_managed: dict[str, list[Any]] = defaultdict(list)
    for task in tasks:
        for chan, val in task.writes:
            if chan == TASKS:
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
            updated = channels[chan].update(vals)
            if updated and get_next_version is not None:
                checkpoint["channel_versions"][chan] = get_next_version(
                    max_version, channels[chan]
                )
            updated_channels.add(chan)

    # Channels that weren't updated in this step are notified of a new step
    for chan in channels:
        if chan not in updated_channels:
            if channels[chan].update([]) and get_next_version is not None:
                checkpoint["channel_versions"][chan] = get_next_version(
                    max_version, channels[chan]
                )

    # Return managed values writes to be applied externally
    return pending_writes_by_managed


@overload
def prepare_next_tasks(
    checkpoint: Checkpoint,
    processes: Mapping[str, PregelNode],
    channels: Mapping[str, BaseChannel],
    managed: ManagedValueMapping,
    config: RunnableConfig,
    step: int,
    *,
    for_execution: Literal[False],
    is_resuming: bool = False,
    checkpointer: Literal[None] = None,
    manager: Literal[None] = None,
) -> list[PregelTask]: ...


@overload
def prepare_next_tasks(
    checkpoint: Checkpoint,
    processes: Mapping[str, PregelNode],
    channels: Mapping[str, BaseChannel],
    managed: ManagedValueMapping,
    config: RunnableConfig,
    step: int,
    *,
    for_execution: Literal[True],
    is_resuming: bool,
    checkpointer: Optional[BaseCheckpointSaver],
    manager: Union[None, ParentRunManager, AsyncParentRunManager],
) -> list[PregelExecutableTask]: ...


def prepare_next_tasks(
    checkpoint: Checkpoint,
    processes: Mapping[str, PregelNode],
    channels: Mapping[str, BaseChannel],
    managed: ManagedValueMapping,
    config: RunnableConfig,
    step: int,
    *,
    for_execution: bool,
    is_resuming: bool = False,
    checkpointer: Optional[BaseCheckpointSaver] = None,
    manager: Union[None, ParentRunManager, AsyncParentRunManager] = None,
) -> Union[list[PregelTask], list[PregelExecutableTask]]:
    checkpoint_id = UUID(checkpoint["id"])
    configurable = config.get("configurable", {})
    parent_ns = configurable.get("checkpoint_ns", "")
    tasks: Union[list[PregelTask], list[PregelExecutableTask]] = []
    # Consume pending packets
    for packet in checkpoint["pending_sends"]:
        if not isinstance(packet, Send):
            logger.warning(
                f"Ignoring invalid packet type {type(packet)} in pending sends"
            )
            continue
        if packet.node not in processes:
            logger.warning(f"Ignoring unknown node name {packet.node} in pending sends")
            continue
        # create task id
        triggers = [TASKS]
        metadata = {
            "langgraph_step": step,
            "langgraph_node": packet.node,
            "langgraph_triggers": triggers,
            "langgraph_task_idx": len(tasks),
        }
        checkpoint_ns = (
            f"{parent_ns}{NS_SEP}{packet.node}" if parent_ns else packet.node
        )
        task_id = str(
            uuid5(
                checkpoint_id,
                "".join(
                    (checkpoint_ns, str(step), packet.node, *triggers, str(len(tasks)))
                ),
            )
        )
        if for_execution:
            proc = processes[packet.node]
            if node := proc.node:
                managed.replace_runtime_placeholders(step, packet.arg)
                writes = deque()
                task_checkpoint_ns = f"{checkpoint_ns}:{task_id}"
                tasks.append(
                    PregelExecutableTask(
                        packet.node,
                        packet.arg,
                        node,
                        writes,
                        patch_config(
                            merge_configs(
                                config,
                                processes[packet.node].config,
                                {"metadata": metadata},
                            ),
                            run_name=packet.node,
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
                                    step,
                                    writes.extend,
                                    processes,
                                    channels,
                                    managed,
                                ),
                                CONFIG_KEY_READ: partial(
                                    local_read,
                                    step,
                                    checkpoint,
                                    channels,
                                    managed,
                                    PregelTaskWrites(packet.node, writes, triggers),
                                    config,
                                ),
                                CONFIG_KEY_CHECKPOINTER: (
                                    checkpointer
                                    or configurable.get(CONFIG_KEY_CHECKPOINTER)
                                ),
                                CONFIG_KEY_CHECKPOINT_MAP: {
                                    **configurable.get(CONFIG_KEY_CHECKPOINT_MAP, {}),
                                    parent_ns: checkpoint["id"],
                                },
                                CONFIG_KEY_RESUMING: is_resuming,
                                "checkpoint_id": None,
                                "checkpoint_ns": task_checkpoint_ns,
                            },
                        ),
                        triggers,
                        proc.retry_policy,
                        None,
                        task_id,
                    )
                )
        else:
            tasks.append(PregelTask(task_id, packet.node))
    # Check if any processes should be run in next step
    # If so, prepare the values to be passed to them
    version_type = type(next(iter(checkpoint["channel_versions"].values()), None))
    null_version = version_type()
    if null_version is None:
        return tasks
    for name, proc in processes.items():
        seen = checkpoint["versions_seen"].get(name, {})
        # If any of the channels read by this process were updated
        if triggers := sorted(
            chan
            for chan in proc.triggers
            if not isinstance(
                read_channel(channels, chan, return_exception=True), EmptyChannelError
            )
            and checkpoint["channel_versions"].get(chan, null_version)
            > seen.get(chan, null_version)
        ):
            try:
                val = next(
                    _proc_input(
                        step, proc, managed, channels, for_execution=for_execution
                    )
                )
            except StopIteration:
                continue

            # create task id
            metadata = {
                "langgraph_step": step,
                "langgraph_node": name,
                "langgraph_triggers": triggers,
                "langgraph_task_idx": len(tasks),
            }
            checkpoint_ns = f"{parent_ns}{NS_SEP}{name}" if parent_ns else name
            task_id = str(
                uuid5(
                    checkpoint_id,
                    "".join(
                        (checkpoint_ns, str(step), name, *triggers, str(len(tasks)))
                    ),
                )
            )

            if for_execution:
                if node := proc.node:
                    writes = deque()
                    task_checkpoint_ns = f"{checkpoint_ns}:{task_id}"
                    tasks.append(
                        PregelExecutableTask(
                            name,
                            val,
                            node,
                            writes,
                            patch_config(
                                merge_configs(
                                    config,
                                    proc.config,
                                    {"metadata": metadata},
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
                                        step,
                                        writes.extend,
                                        processes,
                                        channels,
                                        managed,
                                    ),
                                    CONFIG_KEY_READ: partial(
                                        local_read,
                                        step,
                                        checkpoint,
                                        channels,
                                        managed,
                                        PregelTaskWrites(name, writes, triggers),
                                        config,
                                    ),
                                    CONFIG_KEY_CHECKPOINTER: (
                                        checkpointer
                                        or configurable.get(CONFIG_KEY_CHECKPOINTER)
                                    ),
                                    CONFIG_KEY_CHECKPOINT_MAP: {
                                        **configurable.get(
                                            CONFIG_KEY_CHECKPOINT_MAP, {}
                                        ),
                                        parent_ns: checkpoint["id"],
                                    },
                                    CONFIG_KEY_RESUMING: is_resuming,
                                    "checkpoint_ns": task_checkpoint_ns,
                                },
                            ),
                            triggers,
                            proc.retry_policy,
                            None,
                            task_id,
                        )
                    )
            else:
                tasks.append(PregelTask(task_id, name))
    return tasks


def _proc_input(
    step: int,
    proc: PregelNode,
    managed: ManagedValueMapping,
    channels: Mapping[str, BaseChannel],
    *,
    for_execution: bool,
) -> Iterator[Any]:
    # If all trigger channels subscribed by this process are not empty
    # then invoke the process with the values of all non-empty channels
    if isinstance(proc.channels, dict):
        try:
            val: dict[str, Any] = {}
            for k, chan in proc.channels.items():
                if chan in proc.triggers:
                    val[k] = read_channel(channels, chan, catch=False)
                elif chan in channels:
                    try:
                        val[k] = read_channel(channels, chan, catch=False)
                    except EmptyChannelError:
                        continue
                else:
                    val[k] = managed[k](step)
        except EmptyChannelError:
            return
    elif isinstance(proc.channels, list):
        for chan in proc.channels:
            try:
                val = read_channel(channels, chan, catch=False)
                break
            except EmptyChannelError:
                pass
        else:
            return
    else:
        raise RuntimeError(
            "Invalid channels type, expected list or dict, got {proc.channels}"
        )

    # If the process has a mapper, apply it to the value
    if for_execution and proc.mapper is not None:
        val = proc.mapper(val)

    yield val
