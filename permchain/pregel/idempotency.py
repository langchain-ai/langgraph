import hashlib
import pickle
from typing import Any, Mapping

from langchain.schema.runnable import Runnable, RunnableConfig

from permchain.channels.base import BaseChannel, EmptyChannelError


def get_channel_values(channels: Mapping[str, BaseChannel]) -> Mapping[str, Any]:
    """Create a checkpoint for the given channels."""
    values = {}
    for k, v in channels.items():
        try:
            values[k] = v.get()
        except EmptyChannelError:
            pass
    return values


def add_idempotency_keys(
    tasks: list[tuple[Runnable, Any, str]],
    channels: Mapping[str, BaseChannel],
    config: RunnableConfig,
) -> list[tuple[Runnable, Any, str, str]]:
    base = hashlib.sha256(
        pickle.dumps(get_channel_values(channels), protocol=pickle.HIGHEST_PROTOCOL)
    )
    base.update(
        pickle.dumps(
            {
                "configurable": {
                    k: v
                    for k, v in config.get("configurable", {}).items()
                    if not callable(v)
                },
                "recursion_limit": config["recursion_limit"],
            },
            protocol=pickle.HIGHEST_PROTOCOL,
        )
    )
    tasks_w_keys: list[tuple[Runnable, Any, str, str]] = []
    for task in tasks:
        h = base.copy()
        h.update(pickle.dumps(task[1:], protocol=pickle.HIGHEST_PROTOCOL))
        tasks_w_keys.append((*task, h.hexdigest()))
    return tasks_w_keys
