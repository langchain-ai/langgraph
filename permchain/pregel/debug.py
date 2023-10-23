from pprint import pformat
from typing import Any, Iterator, Mapping

from langchain.schema.runnable import Runnable
from langchain.utils.input import get_bolded_text, get_colored_text

from permchain.channels.base import Channel, EmptyChannelError


def print_step_start(step: int, next_tasks: list[tuple[Runnable, Any, str]]) -> None:
    n_tasks = len(next_tasks)
    print(
        f"{get_colored_text('[pregel/step]', color='blue')} "
        + get_bolded_text(
            f"Starting step {step} with {n_tasks} task{'s' if n_tasks > 1 else ''}. Next tasks:\n"
        )
        + "\n".join(f"- {name}({pformat(val)})" for _, val, name in next_tasks)
    )


def print_checkpoint(step: int, channels: Mapping[str, Channel]) -> None:
    print(
        f"{get_colored_text('[pregel/checkpoint]', color='blue')} "
        + get_bolded_text(f"Finishing step {step}. Channel values:\n")
        + pformat({name: val for name, val in _read_channels(channels)}, depth=1)
    )


def _read_channels(channels: Mapping[str, Channel]) -> Iterator[tuple[str, Any]]:
    for name, channel in channels.items():
        try:
            yield (name, channel.get())
        except EmptyChannelError:
            pass
