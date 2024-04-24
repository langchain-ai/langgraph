from collections import defaultdict
from pprint import pformat
from typing import Any, Iterator, Mapping, Sequence

from langchain_core.utils.input import get_bolded_text, get_colored_text

from langgraph.channels.base import BaseChannel, EmptyChannelError
from langgraph.pregel.types import PregelExecutableTask


def print_step_tasks(step: int, next_tasks: list[PregelExecutableTask]) -> None:
    n_tasks = len(next_tasks)
    print(
        f"{get_colored_text(f'[{step}:tasks]', color='blue')} "
        + get_bolded_text(
            f"Starting step {step} with {n_tasks} task{'s' if n_tasks > 1 else ''}:\n"
        )
        + "\n".join(
            f"- {get_colored_text(name, 'green')} -> {pformat(val)}"
            for name, val, _, _, _ in next_tasks
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
        + pformat(
            {name: val for name, val in _read_channels(channels) if name in whitelist},
            depth=3,
        )
    )


def _read_channels(channels: Mapping[str, BaseChannel]) -> Iterator[tuple[str, Any]]:
    for name, channel in channels.items():
        try:
            yield (name, channel.get())
        except EmptyChannelError:
            pass
