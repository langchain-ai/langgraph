from __future__ import annotations

import hashlib
import json
import logging
import time
from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import asdict
from typing import Any
from uuid import UUID

from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.base import CheckpointMetadata, PendingWrite

from langgraph._internal._config import patch_checkpoint_map
from langgraph._internal._constants import (
    CONF,
    CONFIG_KEY_CHECKPOINT_NS,
    ERROR,
    INTERRUPT,
    NS_END,
    NS_SEP,
    RETURN,
)
from langgraph._internal._typing import MISSING
from langgraph.channels.base import BaseChannel
from langgraph.constants import TAG_HIDDEN
from langgraph.pregel._io import read_channels
from langgraph.types import (
    CheckpointPayload,
    PregelExecutableTask,
    PregelTask,
    StateSnapshot,
    TaskPayload,
    TaskResultPayload,
)

TASK_NAMESPACE = UUID("6ba7b831-9dad-11d1-80b4-00c04fd430c8")

_audit_logger = logging.getLogger("langgraph.audit")

_INPUT_MAX_BYTES = 4096


def _truncate_input(value: Any) -> Any:
    """Truncate task input to enforce output data minimisation."""
    try:
        serialised = json.dumps(value, default=str)
    except Exception:
        serialised = str(value)
    if len(serialised) > _INPUT_MAX_BYTES:
        return serialised[:_INPUT_MAX_BYTES] + "...<truncated>"
    return value


def _hash_value(value: Any) -> str:
    """Return a SHA-256 hex digest of the JSON-serialised value."""
    try:
        raw = json.dumps(value, sort_keys=True, default=str).encode()
    except Exception:
        raw = str(value).encode()
    return hashlib.sha256(raw).hexdigest()


def _redact_config(config: RunnableConfig | None) -> dict[str, Any] | None:
    """Return a redacted view of a config exposing only non-sensitive keys."""
    if config is None:
        return None
    configurable = config.get("configurable", {})
    redacted: dict[str, Any] = {}
    for k, v in configurable.items():
        if k in ("thread_id", CONFIG_KEY_CHECKPOINT_NS):
            redacted[k] = "<redacted>"
        elif not k.startswith("__pregel_"):
            redacted[k] = v
    return {"configurable": redacted}


def _write_audit_record(event: str, record: dict[str, Any]) -> None:
    """Append an audit log entry with a timestamp and event type."""
    entry = {
        "audit_event": event,
        "timestamp": time.time(),
        **record,
    }
    _audit_logger.info(json.dumps(entry, default=str))


def map_debug_tasks(tasks: Iterable[PregelExecutableTask]) -> Iterator[TaskPayload]:
    """Produce "task" events for stream_mode=debug."""
    for task in tasks:
        if task.config is not None and TAG_HIDDEN in task.config.get("tags", []):
            continue

        truncated_input = _truncate_input(task.input)
        input_hash = _hash_value(task.input)

        _write_audit_record(
            "task_dispatched",
            {
                "task_id": task.id,
                "task_name": task.name,
                "triggers": task.triggers,
                "input_hash": input_hash,
            },
        )

        yield {
            "id": task.id,
            "name": task.name,
            "input": truncated_input,
            "triggers": task.triggers,
        }


def is_multiple_channel_write(value: Any) -> bool:
    """Return True if the payload already wraps multiple writes from the same channel."""
    return (
        isinstance(value, dict)
        and "$writes" in value
        and isinstance(value["$writes"], list)
    )


def map_task_result_writes(writes: Sequence[tuple[str, Any]]) -> dict[str, Any]:
    """Folds task writes into a result dict and aggregates multiple writes to the same channel.

    If the channel contains a single write, we record the write in the result dict as `{channel: write}`
    If the channel contains multiple writes, we record the writes in the result dict as `{channel: {'$writes': [write1, write2, ...]}}`"""

    result: dict[str, Any] = {}
    for channel, value in writes:
        existing = result.get(channel)

        if existing is not None:
            channel_writes = (
                existing["$writes"]
                if is_multiple_channel_write(existing)
                else [existing]
            )
            channel_writes.append(value)
            result[channel] = {"$writes": channel_writes}
        else:
            result[channel] = value
    return result


def map_debug_task_results(
    task_tup: tuple[PregelExecutableTask, Sequence[tuple[str, Any]]],
    stream_keys: str | Sequence[str],
) -> Iterator[TaskResultPayload]:
    """Produce "task_result" events for stream_mode=debug."""
    stream_channels_list = (
        [stream_keys] if isinstance(stream_keys, str) else stream_keys
    )
    task, writes = task_tup
    error = next((w[1] for w in writes if w[0] == ERROR), None)
    result = map_task_result_writes(
        [w for w in writes if w[0] in stream_channels_list or w[0] == RETURN]
    )
    interrupts = [
        asdict(v)
        for w in writes
        if w[0] == INTERRUPT
        for v in (w[1] if isinstance(w[1], Sequence) else [w[1]])
    ]

    _write_audit_record(
        "task_result",
        {
            "task_id": task.id,
            "task_name": task.name,
            "has_error": error is not None,
            "result_hash": _hash_value(result),
            "interrupt_count": len(interrupts),
        },
    )

    yield {
        "id": task.id,
        "name": task.name,
        "error": error,
        "result": result,
        "interrupts": interrupts,
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

    tasks = list(tasks)

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

    channel_values = read_channels(channels, stream_channels)
    values_hash = _hash_value(channel_values)

    redacted_config = _redact_config(rm_pregel_keys(patch_checkpoint_map(config, metadata)))
    redacted_parent_config = _redact_config(rm_pregel_keys(patch_checkpoint_map(parent_config, metadata)))

    computed_tasks = list(tasks_w_writes(tasks, pending_writes, task_states, output_keys))

    _write_audit_record(
        "checkpoint",
        {
            "values_hash": values_hash,
            "task_count": len(computed_tasks),
            "task_ids": [t.id for t in computed_tasks],
            "task_names": [t.name for t in computed_tasks],
            "metadata": metadata,
        },
    )

    yield {
        "config": redacted_config,
        "parent_config": redacted_parent_config,
        "values": channel_values,
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
            for t in computed_tasks
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
        task_error = next(
            (exc for tid, n, exc in pending_writes if tid == task.id and n == ERROR),
            None,
        )
        task_interrupts = tuple(
            v
            for tid, n, vv in pending_writes
            if tid == task.id and n == INTERRUPT
            for v in (vv if isinstance(vv, Sequence) else [vv])
        )

        task_writes = [
            (chan, val)
            for tid, chan, val in pending_writes
            if tid == task.id and chan not in (ERROR, INTERRUPT, RETURN)
        ]

        if rtn is not MISSING:
            task_result = rtn
        elif isinstance(output_keys, str):
            # unwrap single channel writes to just the write value
            filtered_writes = [
                (chan, val) for chan, val in task_writes if chan == output_keys
            ]
            mapped_writes = map_task_result_writes(filtered_writes)
            task_result = mapped_writes.get(str(output_keys)) if mapped_writes else None
        else:
            if isinstance(output_keys, str):
                output_keys = [output_keys]
            # map task result writes to the desired output channels
            # repeateed writes to the same channel are aggregated into: {'$writes': [write1, write2, ...]}
            filtered_writes = [
                (chan, val) for chan, val in task_writes if chan in output_keys
            ]
            mapped_writes = map_task_result_writes(filtered_writes)
            task_result = mapped_writes if filtered_writes else {}

        has_writes = rtn is not MISSING or any(
            w[0] == task.id and w[1] not in (ERROR, INTERRUPT) for w in pending_writes
        )

        _write_audit_record(
            "task_write",
            {
                "task_id": task.id,
                "task_name": task.name,
                "has_error": task_error is not None,
                "has_result": has_writes,
                "result_hash": _hash_value(task_result) if has_writes else None,
                "interrupt_count": len(task_interrupts),
            },
        )

        out.append(
            PregelTask(
                task.id,
                task.name,
                task.path,
                task_error,
                task_interrupts,
                states.get(task.id) if states else None,
                task_result if has_writes else None,
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