import json
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
from langchain_core.runnables.config import (
    RunnableConfig,
    merge_configs,
    patch_config,
)

from langgraph.channels.base import BaseChannel
from langgraph.channels.context import Context
from langgraph.channels.manager import ChannelsManager, create_checkpoint
from langgraph.checkpoint.base import BaseCheckpointSaver, Checkpoint, copy_checkpoint
from langgraph.constants import (
    CONFIG_KEY_CHECKPOINTER,
    CONFIG_KEY_READ,
    CONFIG_KEY_RESUMING,
    CONFIG_KEY_SEND,
    INTERRUPT,
    RESERVED,
    TAG_HIDDEN,
    TASKS,
    THREAD_ID_SEPARATOR,
    Send,
)
from langgraph.errors import EmptyChannelError, InvalidUpdateError
from langgraph.managed.base import ManagedValueMapping, is_managed_value
from langgraph.pregel.io import read_channel, read_channels
from langgraph.pregel.log import logger
from langgraph.pregel.read import PregelNode
from langgraph.pregel.types import All, PregelExecutableTask, PregelTaskDescription


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
) -> bool:
    version_type = type(next(iter(checkpoint["channel_versions"].values()), None))
    null_version = version_type()
    seen = checkpoint["versions_seen"].get(INTERRUPT, {})
    return (
        # interrupt if any channel has been updated since last interrupt
        any(
            version > seen.get(chan, null_version)
            for chan, version in checkpoint["channel_versions"].items()
        )
        # and any triggered node is in interrupt_nodes list
        and any(
            task.name
            for task in tasks
            if (
                (not task.config or TAG_HIDDEN not in task.config.get("tags"))
                if interrupt_nodes == "*"
                else task.name in interrupt_nodes
            )
        )
    )


def local_read(
    checkpoint: Checkpoint,
    channels: Mapping[str, BaseChannel],
    task: WritesProtocol,
    config: RunnableConfig,
    select: Union[list[str], str],
    fresh: bool = False,
) -> Union[dict[str, Any], Any]:
    if fresh:
        new_checkpoint = create_checkpoint(copy_checkpoint(checkpoint), channels, -1)
        context_channels = {k: v for k, v in channels.items() if isinstance(v, Context)}
        with ChannelsManager(
            {k: v for k, v in channels.items() if k not in context_channels},
            new_checkpoint,
            config,
        ) as channels:
            all_channels = {**channels, **context_channels}
            apply_writes(new_checkpoint, all_channels, [task], None)
            return read_channels(all_channels, select)
    else:
        return read_channels(channels, select)


def local_write(
    commit: Callable[[Sequence[tuple[str, Any]]], None],
    processes: Mapping[str, PregelNode],
    channels: Mapping[str, BaseChannel],
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
        elif chan not in channels:
            logger.warning(f"Skipping write for channel '{chan}' which has no readers")
    commit(writes)


def increment(current: Optional[int], channel: BaseChannel) -> int:
    return current + 1 if current is not None else 1


def apply_writes(
    checkpoint: Checkpoint,
    channels: Mapping[str, BaseChannel],
    tasks: Sequence[WritesProtocol],
    get_next_version: Optional[Callable[[int, BaseChannel], int]],
) -> None:
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
        chan for task in tasks for chan in task.triggers if chan not in RESERVED
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
    for task in tasks:
        for chan, val in task.writes:
            if chan == TASKS:
                checkpoint["pending_sends"].append(val)
            else:
                pending_writes_by_channel[chan].append(val)

    # Find the highest version of all channels
    if checkpoint["channel_versions"]:
        max_version = max(checkpoint["channel_versions"].values())
    else:
        max_version = None

    # Apply writes to channels
    updated_channels: set[str] = set()
    for chan, vals in pending_writes_by_channel.items():
        if chan in channels:
            try:
                updated = channels[chan].update(vals)
            except InvalidUpdateError as e:
                raise InvalidUpdateError(
                    f"Invalid update for channel {chan} with values {vals}"
                ) from e
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


@overload
def prepare_next_tasks(
    checkpoint: Checkpoint,
    processes: Mapping[str, PregelNode],
    channels: Mapping[str, BaseChannel],
    managed: ManagedValueMapping,
    config: RunnableConfig,
    step: int,
    for_execution: Literal[False],
    is_resuming: bool = False,
    checkpointer: Literal[None] = None,
    manager: Literal[None] = None,
) -> list[PregelTaskDescription]:
    ...


@overload
def prepare_next_tasks(
    checkpoint: Checkpoint,
    processes: Mapping[str, PregelNode],
    channels: Mapping[str, BaseChannel],
    managed: ManagedValueMapping,
    config: RunnableConfig,
    step: int,
    for_execution: Literal[True],
    is_resuming: bool,
    checkpointer: Optional[BaseCheckpointSaver],
    manager: Union[None, ParentRunManager, AsyncParentRunManager],
) -> list[PregelExecutableTask]:
    ...


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
) -> Union[list[PregelTaskDescription], list[PregelExecutableTask]]:
    tasks: Union[list[PregelTaskDescription], list[PregelExecutableTask]] = []
    # Consume pending packets
    for packet in checkpoint["pending_sends"]:
        if not isinstance(packet, Send):
            logger.warn(f"Ignoring invalid packet type {type(packet)} in pending sends")
            continue
        if for_execution:
            proc = processes[packet.node]
            if node := proc.get_node():
                triggers = [TASKS]
                metadata = {
                    "langgraph_step": step,
                    "langgraph_node": packet.node,
                    "langgraph_triggers": triggers,
                    "langgraph_task_idx": len(tasks),
                }
                task_id = str(uuid5(UUID(checkpoint["id"]), json.dumps(metadata)))
                writes = deque()
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
                                # deque.extend is thread-safe
                                CONFIG_KEY_SEND: partial(
                                    local_write, writes.extend, processes, channels
                                ),
                                CONFIG_KEY_READ: partial(
                                    local_read,
                                    checkpoint,
                                    channels,
                                    PregelTaskWrites(packet.node, writes, triggers),
                                    config,
                                ),
                                # in Send we can't checkpoint nested graphs
                                # as they could be running in parallel
                            },
                        ),
                        triggers,
                        proc.retry_policy,
                        task_id,
                    )
                )
        else:
            tasks.append(PregelTaskDescription(packet.node, packet.arg))
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
                val = next(_proc_input(step, name, proc, managed, channels))
            except StopIteration:
                continue

            if for_execution:
                if node := proc.get_node():
                    metadata = {
                        "langgraph_step": step,
                        "langgraph_node": name,
                        "langgraph_triggers": triggers,
                        "langgraph_task_idx": len(tasks),
                    }
                    task_id = str(uuid5(UUID(checkpoint["id"]), json.dumps(metadata)))
                    if parent_thread_id := config.get("configurable", {}).get(
                        "thread_id"
                    ):
                        thread_id: Optional[
                            str
                        ] = f"{parent_thread_id}{THREAD_ID_SEPARATOR}{name}"
                    else:
                        thread_id = None
                    writes = deque()
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
                                    # deque.extend is thread-safe
                                    CONFIG_KEY_SEND: partial(
                                        local_write, writes.extend, processes, channels
                                    ),
                                    CONFIG_KEY_READ: partial(
                                        local_read,
                                        checkpoint,
                                        channels,
                                        PregelTaskWrites(name, writes, triggers),
                                        config,
                                    ),
                                    CONFIG_KEY_CHECKPOINTER: checkpointer,
                                    CONFIG_KEY_RESUMING: is_resuming,
                                    "thread_id": thread_id,
                                    "thread_ts": checkpoint["id"],
                                },
                            ),
                            triggers,
                            proc.retry_policy,
                            task_id,
                        )
                    )
            else:
                tasks.append(PregelTaskDescription(name, val))
    return tasks


def _proc_input(
    step: int,
    name: str,
    proc: PregelNode,
    managed: ManagedValueMapping,
    channels: Mapping[str, BaseChannel],
) -> Iterator[Any]:
    # If all trigger channels subscribed by this process are not empty
    # then invoke the process with the values of all non-empty channels
    if isinstance(proc.channels, dict):
        try:
            val: dict = {
                k: read_channel(
                    channels,
                    chan,
                    catch=chan not in proc.triggers,
                )
                for k, chan in proc.channels.items()
                if isinstance(chan, str)
            }

            managed_values = {}
            for key, chan in proc.channels.items():
                if is_managed_value(chan):
                    managed_values[key] = managed[key](
                        step, PregelTaskDescription(name, val)
                    )

            val.update(managed_values)
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
    if proc.mapper is not None:
        val = proc.mapper(val)

    yield val
