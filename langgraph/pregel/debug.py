import json
from collections import defaultdict
from datetime import datetime, timezone
from pprint import pformat
from typing import Any, Iterator, Literal, Mapping, Optional, Sequence, TypedDict, Union
from uuid import UUID, uuid5

from langchain_core.runnables.config import RunnableConfig
from langchain_core.utils.input import get_bolded_text, get_colored_text

from langgraph.channels.base import BaseChannel
from langgraph.constants import TAG_HIDDEN
from langgraph.pregel.io import read_channels
from langgraph.pregel.types import PregelExecutableTask


class TaskPayload(TypedDict):
    id: str
    name: str
    input: Any
    triggers: list[str]


class TaskResultPayload(TypedDict):
    id: str
    name: str
    result: list[tuple[str, Any]]


class CheckpointPayload(TypedDict):
    config: Optional[RunnableConfig]
    values: dict[str, Any]


class DebugOutputBase(TypedDict):
    timestamp: str
    step: int
    type: str
    payload: dict[str, Any]


class DebugOutputTask(DebugOutputBase):
    type: Literal["task"]
    payload: TaskPayload


class DebugOutputTaskResult(DebugOutputBase):
    type: Literal["task_result"]
    payload: TaskResultPayload


class DebugOutputCheckpoint(DebugOutputBase):
    type: Literal["checkpoint"]
    payload: CheckpointPayload


DebugOutput = Union[DebugOutputTask, DebugOutputTaskResult, DebugOutputCheckpoint]


TASK_NAMESPACE = UUID("6ba7b831-9dad-11d1-80b4-00c04fd430c8")


def map_debug_tasks(
    step: int, tasks: list[PregelExecutableTask]
) -> Iterator[DebugOutputTask]:
    ts = datetime.now(timezone.utc).isoformat()
    for name, input, _, _, config, triggers in tasks:
        if config is not None and TAG_HIDDEN in config.get("tags", []):
            continue

        yield {
            "type": "task",
            "timestamp": ts,
            "step": step,
            "payload": {
                "id": str(uuid5(TASK_NAMESPACE, json.dumps((name, step)))),
                "name": name,
                "input": input,
                "triggers": triggers,
            },
        }


def map_debug_task_results(
    step: int,
    tasks: list[PregelExecutableTask],
    stream_channels_list: Sequence[str],
) -> Iterator[DebugOutputTaskResult]:
    ts = datetime.now(timezone.utc).isoformat()
    for name, _, _, writes, config, _ in tasks:
        if config is not None and TAG_HIDDEN in config.get("tags", []):
            continue

        yield {
            "type": "task_result",
            "timestamp": ts,
            "step": step,
            "payload": {
                "id": str(uuid5(TASK_NAMESPACE, json.dumps((name, step)))),
                "name": name,
                "result": [w for w in writes if w[0] in stream_channels_list],
            },
        }


def map_debug_checkpoint(
    step: int,
    config: RunnableConfig,
    channels: Mapping[str, BaseChannel],
    stream_channels: Union[str, Sequence[str]],
) -> DebugOutputCheckpoint:
    ts = datetime.now(timezone.utc).isoformat()
    return {
        "type": "checkpoint",
        "timestamp": ts,
        "step": step,
        "payload": {
            "config": config,
            "values": read_channels(channels, stream_channels),
        },
    }


def print_step_tasks(step: int, next_tasks: list[PregelExecutableTask]) -> None:
    n_tasks = len(next_tasks)
    print(
        f"{get_colored_text(f'[{step}:tasks]', color='blue')} "
        + get_bolded_text(
            f"Starting step {step} with {n_tasks} task{'s' if n_tasks > 1 else ''}:\n"
        )
        + "\n".join(
            f"- {get_colored_text(name, 'green')} -> {pformat(val)}"
            for name, val, _, _, _, _ in next_tasks
        )
    )


def print_step_writes(
    step: int, writes: Sequence[tuple[str, Any]], whitelist: Sequence[str]
) -> None:
    by_channel: dict[str, list[Any]] = defaultdict(list)
    for channel, value in writes:
        if channel in whitelist:
            by_channel[channel].append(value)
    print(
        f"{get_colored_text(f'[{step}:writes]', color='blue')} "
        + get_bolded_text(
            f"Finished step {step} with writes to {len(by_channel)} channel{'s' if len(by_channel) > 1 else ''}:\n"
        )
        + "\n".join(
            f"- {get_colored_text(name, 'yellow')} -> {', '.join(pformat(v) for v in vals)}"
            for name, vals in by_channel.items()
        )
    )


def print_step_checkpoint(
    step: int, channels: Mapping[str, BaseChannel], whitelist: Sequence[str]
) -> None:
    print(
        f"{get_colored_text(f'[{step}:checkpoint]', color='blue')} "
        + get_bolded_text(f"State at the end of step {step}:\n")
        + pformat(read_channels(channels, whitelist), depth=3)
    )
