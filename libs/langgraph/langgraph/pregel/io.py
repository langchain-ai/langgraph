from typing import Any, Iterator, Mapping, Optional, Sequence, TypeVar, Union

from langchain_core.runnables.utils import AddableDict

from langgraph.channels.base import BaseChannel, EmptyChannelError
from langgraph.constants import TAG_HIDDEN
from langgraph.pregel.log import logger
from langgraph.pregel.types import PregelExecutableTask


def read_channel(
    channels: Mapping[str, BaseChannel],
    chan: str,
    *,
    catch: bool = True,
    return_exception: bool = False,
) -> Any:
    try:
        return channels[chan].get()
    except EmptyChannelError as exc:
        if return_exception:
            return exc
        elif catch:
            return None
        else:
            raise


def read_channels(
    channels: Mapping[str, BaseChannel],
    select: Union[list[str], str],
    *,
    skip_empty: bool = True,
) -> Union[dict[str, Any], Any]:
    if isinstance(select, str):
        return read_channel(channels, select)
    else:
        values: dict[str, Any] = {}
        for k in select:
            try:
                values[k] = read_channel(channels, k, catch=not skip_empty)
            except EmptyChannelError:
                pass
        return values


def map_input(
    input_channels: Union[str, Sequence[str]],
    chunk: Optional[Union[dict[str, Any], Any]],
) -> Iterator[tuple[str, Any]]:
    """Map input chunk to a sequence of pending writes in the form (channel, value)."""
    if chunk is None:
        return
    elif isinstance(input_channels, str):
        yield (input_channels, chunk)
    else:
        if not isinstance(chunk, dict):
            raise TypeError(f"Expected chunk to be a dict, got {type(chunk).__name__}")
        for k in chunk:
            if k in input_channels:
                yield (k, chunk[k])
            else:
                logger.warning(f"Input channel {k} not found in {input_channels}")


class AddableValuesDict(AddableDict):
    def __add__(self, other: dict[str, Any]) -> "AddableValuesDict":
        return self | other

    def __radd__(self, other: dict[str, Any]) -> "AddableValuesDict":
        return other | self


def map_output_values(
    output_channels: Union[str, Sequence[str]],
    pending_writes: Sequence[tuple[str, Any]],
    channels: Mapping[str, BaseChannel],
) -> Iterator[Union[dict[str, Any], Any]]:
    """Map pending writes (a sequence of tuples (channel, value)) to output chunk."""
    if isinstance(output_channels, str):
        if any(chan == output_channels for chan, _ in pending_writes):
            yield read_channel(channels, output_channels)
    else:
        if {c for c, _ in pending_writes if c in output_channels}:
            yield AddableValuesDict(read_channels(channels, output_channels))


class AddableUpdatesDict(AddableDict):
    def __add__(self, other: dict[str, Any]) -> "AddableUpdatesDict":
        return [self, other]

    def __radd__(self, other: dict[str, Any]) -> "AddableUpdatesDict":
        raise TypeError("AddableUpdatesDict does not support right-side addition")


def map_output_updates(
    output_channels: Union[str, Sequence[str]],
    tasks: list[PregelExecutableTask],
) -> Iterator[dict[str, Union[Any, dict[str, Any]]]]:
    """Map pending writes (a sequence of tuples (channel, value)) to output chunk."""
    output_tasks = [
        t for t in tasks if not t.config or TAG_HIDDEN not in t.config.get("tags")
    ]
    if not output_tasks:
        return
    if isinstance(output_channels, str):
        updated = [
            (task.name, value)
            for task in output_tasks
            for chan, value in task.writes
            if chan == output_channels
        ]
    else:
        updated = [
            (
                task.name,
                {chan: value for chan, value in task.writes if chan in output_channels},
            )
            for task in output_tasks
            if any(chan in output_channels for chan, _ in task.writes)
        ]
    grouped = {t.name: [] for t in output_tasks}
    for node, value in updated:
        grouped[node].append(value)
    for node, value in grouped.items():
        if len(value) == 0:
            grouped[node] = None
        if len(value) == 1:
            grouped[node] = value[0]
    yield AddableUpdatesDict(grouped)


T = TypeVar("T")


def single(iter: Iterator[T]) -> Optional[T]:
    for item in iter:
        return item
