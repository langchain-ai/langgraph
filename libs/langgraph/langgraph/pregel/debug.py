from __future__ import annotations

from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import asdict
from typing import Any
from uuid import UUID

from typing_extensions import TypedDict

from langgraph.channels.base import BaseChannel
from langgraph.checkpoint.base import CheckpointMetadata, PendingWrite
from langgraph.constants import (
    CONF,
    CONFIG_KEY_CHECKPOINT_NS,
    ERROR,
    INTERRUPT,
    MISSING,
    NS_END,
    NS_SEP,
    RETURN,
    TAG_HIDDEN,
)
from langgraph.pregel.io import read_channels
from langgraph.types import PregelExecutableTask, PregelTask, StateSnapshot
from langgraph.utils.config import RunnableConfig, patch_checkpoint_map


class TaskPayload(TypedDict):
    id: str
    name: str
    input: Any
    triggers: list[str]


class TaskResultPayload(TypedDict):
    id: str
    name: str
    error: str | None
    interrupts: list[dict]
    result: list[tuple[str, Any]]


class CheckpointTask(TypedDict):
    id: str
    name: str
    error: str | None
    interrupts: list[dict]
    state: RunnableConfig | None


class CheckpointPayload(TypedDict):
    config: RunnableConfig | None
    metadata: CheckpointMetadata
    values: dict[str, Any]
    next: list[str]
    parent_config: RunnableConfig | None
    tasks: list[CheckpointTask]


TASK_NAMESPACE = UUID("6ba7b831-9dad-11d1-80b4-00c04fd430c8")


def map_debug_tasks(tasks: Iterable[PregelExecutableTask]) -> Iterator[TaskPayload]:
    """Produce "task" events for stream_mode=debug."""
    for task in tasks:
        if task.config is not None and TAG_HIDDEN in task.config.get("tags", []):
            continue

        yield {
            "id": task.id,
            "name": task.name,
            "input": task.input,
            "triggers": task.triggers,
        }


def map_debug_task_results(
    task_tup: tuple[PregelExecutableTask, Sequence[tuple[str, Any]]],
    stream_keys: str | Sequence[str],
) -> Iterator[TaskResultPayload]:
    """Produce "task_result" events for stream_mode=debug."""
    stream_channels_list = (
        [stream_keys] if isinstance(stream_keys, str) else stream_keys
    )
    task, writes = task_tup
    yield {
        "id": task.id,
        "name": task.name,
        "error": next((w[1] for w in writes if w[0] == ERROR), None),
        "result": [w for w in writes if w[0] in stream_channels_list or w[0] == RETURN],
        "interrupts": [
            asdict(v)
            for w in writes
            if w[0] == INTERRUPT
            for v in (w[1] if isinstance(w[1], Sequence) else [w[1]])
        ],
    }


def rm_pregel_keys(config: RunnableConfig | None) -> RunnableConfig | None:
    """Remove pregel-specific keys from the config."""
    if config is None:
        return config
    return {
        "configurable": {
            k: v
            for k, v in config.get("configurable", {}).items()
            if not k.startswith("__pregel_")
        }
    }


def map_debug_checkpoint(
    config: RunnableConfig,
    channels: Mapping[str, BaseChannel],
    stream_channels: str | Sequence[str],
    metadata: CheckpointMetadata,
    tasks: Iterable[PregelExecutableTask],
    pending_writes: list[PendingWrite],
    parent_config: RunnableConfig | None,
    output_keys: str | Sequence[str],
) -> Iterator[CheckpointPayload]:
    """Produce "checkpoint" events for stream_mode=debug."""

    parent_ns = config[CONF].get(CONFIG_KEY_CHECKPOINT_NS, "")
    task_states: dict[str, RunnableConfig | StateSnapshot] = {}

    for task in tasks:
        if not task.subgraphs:
            continue

        # assemble checkpoint_ns for this task
        task_ns = f"{task.name}{NS_END}{task.id}"
        if parent_ns:
            task_ns = f"{parent_ns}{NS_SEP}{task_ns}"

        # set config as signal that subgraph checkpoints exist
        task_states[task.id] = {
            CONF: {
                "thread_id": config[CONF]["thread_id"],
                CONFIG_KEY_CHECKPOINT_NS: task_ns,
            }
        }

    yield {
        "config": rm_pregel_keys(patch_checkpoint_map(config, metadata)),
        "parent_config": rm_pregel_keys(patch_checkpoint_map(parent_config, metadata)),
        "values": read_channels(channels, stream_channels),
        "metadata": metadata,
        "next": [t.name for t in tasks],
        "tasks": [
            {
                "id": t.id,
                "name": t.name,
                "error": t.error,
                "state": t.state,
            }
            if t.error
            else {
                "id": t.id,
                "name": t.name,
                "result": t.result,
                "interrupts": tuple(asdict(i) for i in t.interrupts),
                "state": t.state,
            }
            if t.result
            else {
                "id": t.id,
                "name": t.name,
                "interrupts": tuple(asdict(i) for i in t.interrupts),
                "state": t.state,
            }
            for t in tasks_w_writes(tasks, pending_writes, task_states, output_keys)
        ],
    }


def tasks_w_writes(
    tasks: Iterable[PregelTask | PregelExecutableTask],
    pending_writes: list[PendingWrite] | None,
    states: dict[str, RunnableConfig | StateSnapshot] | None,
    output_keys: str | Sequence[str],
) -> tuple[PregelTask, ...]:
    """Apply writes / subgraph states to tasks to be returned in a StateSnapshot."""
    pending_writes = pending_writes or []
    out: list[PregelTask] = []
    for task in tasks:
        rtn = next(
            (
                val
                for tid, chan, val in pending_writes
                if tid == task.id and chan == RETURN
            ),
            MISSING,
        )
        out.append(
            PregelTask(
                task.id,
                task.name,
                task.path,
                next(
                    (
                        exc
                        for tid, n, exc in pending_writes
                        if tid == task.id and n == ERROR
                    ),
                    None,
                ),
                tuple(
                    v
                    for tid, n, vv in pending_writes
                    if tid == task.id and n == INTERRUPT
                    for v in (vv if isinstance(vv, Sequence) else [vv])
                ),
                states.get(task.id) if states else None,
                (
                    rtn
                    if rtn is not MISSING
                    else next(
                        (
                            val
                            for tid, chan, val in pending_writes
                            if tid == task.id and chan == output_keys
                        ),
                        None,
                    )
                    if isinstance(output_keys, str)
                    else {
                        chan: val
                        for tid, chan, val in pending_writes
                        if tid == task.id
                        and (
                            chan == output_keys
                            if isinstance(output_keys, str)
                            else chan in output_keys
                        )
                    }
                )
                if any(
                    w[0] == task.id and w[1] not in (ERROR, INTERRUPT)
                    for w in pending_writes
                )
                else None,
            )
        )
    return tuple(out)


COLOR_MAPPING = {
    "black": "0;30",
    "red": "0;31",
    "green": "0;32",
    "yellow": "0;33",
    "blue": "0;34",
    "magenta": "0;35",
    "cyan": "0;36",
    "white": "0;37",
    "gray": "1;30",
}


def get_colored_text(text: str, color: str) -> str:
    """Get colored text."""
    return f"\033[1;3{COLOR_MAPPING[color]}m{text}\033[0m"


def get_bolded_text(text: str) -> str:
    """Get bolded text."""
    return f"\033[1m{text}\033[0m"
