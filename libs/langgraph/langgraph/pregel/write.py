from __future__ import annotations

from typing import (
    Any,
    Callable,
    NamedTuple,
    Optional,
    Sequence,
    TypeVar,
    Union,
    cast,
)

from langchain_core.runnables import Runnable, RunnableConfig
from langchain_core.runnables.utils import ConfigurableFieldSpec

from langgraph.constants import CONF, CONFIG_KEY_SEND, TASKS, Send
from langgraph.errors import InvalidUpdateError
from langgraph.utils.runnable import RunnableCallable

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
    mapper: Optional[Callable] = None
    """Function to transform the value before writing."""


class ChannelWriteTupleEntry(NamedTuple):
    mapper: Callable[[Any], Optional[Sequence[tuple[str, Any]]]]
    """Function to extract tuples from value."""
    value: Any = PASSTHROUGH
    """Value to write, or PASSTHROUGH to use the input."""


class ChannelWrite(RunnableCallable):
    """Implements the logic for sending writes to CONFIG_KEY_SEND.
    Can be used as a runnable or as a static method to call imperatively."""

    writes: list[Union[ChannelWriteEntry, ChannelWriteTupleEntry, Send]]
    """Sequence of write entries or Send objects to write."""

    def __init__(
        self,
        writes: Sequence[Union[ChannelWriteEntry, ChannelWriteTupleEntry, Send]],
        *,
        tags: Optional[Sequence[str]] = None,
        require_at_least_one_of: Optional[Sequence[str]] = None,  # ignored
    ):
        super().__init__(
            func=self._write,
            afunc=self._awrite,
            name=None,
            tags=tags,
            func_accepts_config=True,
        )
        self.writes = cast(
            list[Union[ChannelWriteEntry, ChannelWriteTupleEntry, Send]], writes
        )

    def get_name(
        self, suffix: Optional[str] = None, *, name: Optional[str] = None
    ) -> str:
        if not name:
            name = f"ChannelWrite<{','.join(w.channel if isinstance(w, ChannelWriteEntry) else '...' if isinstance(w, ChannelWriteTupleEntry) else w.node for w in self.writes)}>"
        return super().get_name(suffix, name=name)

    @property
    def config_specs(self) -> list[ConfigurableFieldSpec]:
        return [
            ConfigurableFieldSpec(
                id=CONFIG_KEY_SEND,
                name=CONFIG_KEY_SEND,
                description=None,
                default=None,
                annotation=None,
            ),
        ]

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
        writes: Sequence[Union[ChannelWriteEntry, ChannelWriteTupleEntry, Send]],
        require_at_least_one_of: Optional[Sequence[str]] = None,  # ignored
    ) -> None:
        # validate
        for w in writes:
            if isinstance(w, ChannelWriteEntry):
                if w.channel == TASKS:
                    raise InvalidUpdateError(
                        "Cannot write to the reserved channel TASKS"
                    )
                if w.value is PASSTHROUGH:
                    raise InvalidUpdateError("PASSTHROUGH value must be replaced")
            if isinstance(w, ChannelWriteTupleEntry):
                if w.value is PASSTHROUGH:
                    raise InvalidUpdateError("PASSTHROUGH value must be replaced")
        # assemble writes
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
        write: TYPE_SEND = config[CONF][CONFIG_KEY_SEND]
        write(tuples)

    @staticmethod
    def is_writer(runnable: Runnable) -> bool:
        """Used by PregelNode to distinguish between writers and other runnables."""
        return (
            isinstance(runnable, ChannelWrite)
            or getattr(runnable, "_is_channel_writer", False) is True
        )

    @staticmethod
    def register_writer(runnable: R) -> R:
        """Used to mark a runnable as a writer, so that it can be detected by is_writer.
        Instances of ChannelWrite are automatically marked as writers."""
        # using object.__setattr__ to work around objects that override __setattr__
        # eg. pydantic models and dataclasses
        object.__setattr__(runnable, "_is_channel_writer", True)
        return runnable
