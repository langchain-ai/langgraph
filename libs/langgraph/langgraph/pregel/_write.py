from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import (
    Any,
    NamedTuple,
    TypeVar,
    cast,
)

from langchain_core.runnables import Runnable, RunnableConfig

from langgraph._internal._constants import CONF, CONFIG_KEY_SEND, TASKS
from langgraph._internal._runnable import RunnableCallable
from langgraph._internal._typing import MISSING
from langgraph.errors import InvalidUpdateError
from langgraph.types import Send

TYPE_SEND = Callable[[Sequence[tuple[str, Any]]], None]
R = TypeVar("R", bound=Runnable)

SKIP_WRITE = object()
PASSTHROUGH = object()


class ChannelWriteEntry(NamedTuple):
    channel: str
    """Channel name to write to."""
    value: Any = PASSTHROUGH
    """Value to write, or PASSTHROUGH to use the input."""
    skip_none: bool = False
    """Whether to skip writing if the value is None."""
    mapper: Callable | None = None
    """Function to transform the value before writing."""


class ChannelWriteTupleEntry(NamedTuple):
    mapper: Callable[[Any], Sequence[tuple[str, Any]] | None]
    """Function to extract tuples from value."""
    value: Any = PASSTHROUGH
    """Value to write, or PASSTHROUGH to use the input."""
    static: Sequence[tuple[str, Any, str | None]] | None = None
    """Optional, declared writes for static analysis."""


class ChannelWrite(RunnableCallable):
    """Implements the logic for sending writes to CONFIG_KEY_SEND.
    Can be used as a runnable or as a static method to call imperatively."""

    writes: list[ChannelWriteEntry | ChannelWriteTupleEntry | Send]
    """Sequence of write entries or Send objects to write."""

    def __init__(
        self,
        writes: Sequence[ChannelWriteEntry | ChannelWriteTupleEntry | Send],
        *,
        tags: Sequence[str] | None = None,
    ):
        super().__init__(
            func=self._write,
            afunc=self._awrite,
            name=None,
            tags=tags,
            trace=False,
        )
        self.writes = cast(
            list[ChannelWriteEntry | ChannelWriteTupleEntry | Send], writes
        )

    def get_name(self, suffix: str | None = None, *, name: str | None = None) -> str:
        if not name:
            name = f"ChannelWrite<{','.join(w.channel if isinstance(w, ChannelWriteEntry) else '...' if isinstance(w, ChannelWriteTupleEntry) else w.node for w in self.writes)}>"
        return super().get_name(suffix, name=name)

    def _write(self, input: Any, config: RunnableConfig) -> None:
        writes = [
            ChannelWriteEntry(write.channel, input, write.skip_none, write.mapper)
            if isinstance(write, ChannelWriteEntry) and write.value is PASSTHROUGH
            else ChannelWriteTupleEntry(write.mapper, input)
            if isinstance(write, ChannelWriteTupleEntry) and write.value is PASSTHROUGH
            else write
            for write in self.writes
        ]
        self.do_write(
            config,
            writes,
        )
        return input

    async def _awrite(self, input: Any, config: RunnableConfig) -> None:
        writes = [
            ChannelWriteEntry(write.channel, input, write.skip_none, write.mapper)
            if isinstance(write, ChannelWriteEntry) and write.value is PASSTHROUGH
            else ChannelWriteTupleEntry(write.mapper, input)
            if isinstance(write, ChannelWriteTupleEntry) and write.value is PASSTHROUGH
            else write
            for write in self.writes
        ]
        self.do_write(
            config,
            writes,
        )
        return input

    @staticmethod
    def do_write(
        config: RunnableConfig,
        writes: Sequence[ChannelWriteEntry | ChannelWriteTupleEntry | Send],
        allow_passthrough: bool = True,
    ) -> None:
        # validate
        for w in writes:
            if isinstance(w, ChannelWriteEntry):
                if w.channel == TASKS:
                    raise InvalidUpdateError(
                        "Cannot write to the reserved channel TASKS"
                    )
                if w.value is PASSTHROUGH and not allow_passthrough:
                    raise InvalidUpdateError("PASSTHROUGH value must be replaced")
            if isinstance(w, ChannelWriteTupleEntry):
                if w.value is PASSTHROUGH and not allow_passthrough:
                    raise InvalidUpdateError("PASSTHROUGH value must be replaced")
        # if we want to persist writes found before hitting a ParentCommand
        # can move this to a finally block
        write: TYPE_SEND = config[CONF][CONFIG_KEY_SEND]
        write(_assemble_writes(writes))

    @staticmethod
    def is_writer(runnable: Runnable) -> bool:
        """Used by PregelNode to distinguish between writers and other runnables."""
        return (
            isinstance(runnable, ChannelWrite)
            or getattr(runnable, "_is_channel_writer", MISSING) is not MISSING
        )

    @staticmethod
    def get_static_writes(
        runnable: Runnable,
    ) -> Sequence[tuple[str, Any, str | None]] | None:
        """Used to get conditional writes a writer declares for static analysis."""
        if isinstance(runnable, ChannelWrite):
            return [
                w
                for entry in runnable.writes
                if isinstance(entry, ChannelWriteTupleEntry) and entry.static
                for w in entry.static
            ] or None
        elif writes := getattr(runnable, "_is_channel_writer", MISSING):
            if writes is not MISSING:
                writes = cast(
                    Sequence[tuple[ChannelWriteEntry | Send, str | None]],
                    writes,
                )
                entries = [e for e, _ in writes]
                labels = [la for _, la in writes]
                return [(*t, la) for t, la in zip(_assemble_writes(entries), labels)]

    @staticmethod
    def register_writer(
        runnable: R,
        static: Sequence[tuple[ChannelWriteEntry | Send, str | None]] | None = None,
    ) -> R:
        """Used to mark a runnable as a writer, so that it can be detected by is_writer.
        Instances of ChannelWrite are automatically marked as writers.
        Optionally, a list of declared writes can be passed for static analysis."""
        # using object.__setattr__ to work around objects that override __setattr__
        # eg. pydantic models and dataclasses
        object.__setattr__(runnable, "_is_channel_writer", static)
        return runnable


def _assemble_writes(
    writes: Sequence[ChannelWriteEntry | ChannelWriteTupleEntry | Send],
) -> list[tuple[str, Any]]:
    """Assembles the writes into a list of tuples."""
    tuples: list[tuple[str, Any]] = []
    for w in writes:
        if isinstance(w, Send):
            tuples.append((TASKS, w))
        elif isinstance(w, ChannelWriteTupleEntry):
            if ww := w.mapper(w.value):
                tuples.extend(ww)
        elif isinstance(w, ChannelWriteEntry):
            value = w.mapper(w.value) if w.mapper is not None else w.value
            if value is SKIP_WRITE:
                continue
            if w.skip_none and value is None:
                continue
            tuples.append((w.channel, value))
        else:
            raise ValueError(f"Invalid write entry: {w}")
    return tuples
