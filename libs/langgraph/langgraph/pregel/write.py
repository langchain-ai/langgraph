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

from langgraph.constants import CONFIG_KEY_SEND, TASKS, Send
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


class ChannelWrite(RunnableCallable):
    """Implements th logic for sending writes to CONFIG_KEY_SEND.
    Can be used as a runnable or as a static method to call imperatively."""

    writes: list[Union[ChannelWriteEntry, Send]]
    """Sequence of write entries or Send objects to write."""
    require_at_least_one_of: Optional[Sequence[str]]
    """If defined, at least one of these channels must be written to."""

    def __init__(
        self,
        writes: Sequence[Union[ChannelWriteEntry, Send]],
        *,
        tags: Optional[Sequence[str]] = None,
        require_at_least_one_of: Optional[Sequence[str]] = None,
    ):
        super().__init__(func=self._write, afunc=self._awrite, name=None, tags=tags)
        self.writes = cast(list[Union[ChannelWriteEntry, Send]], writes)
        self.require_at_least_one_of = require_at_least_one_of

    def get_name(
        self, suffix: Optional[str] = None, *, name: Optional[str] = None
    ) -> str:
        if not name:
            name = f"ChannelWrite<{','.join(w.channel if isinstance(w, ChannelWriteEntry) else w.node for w in self.writes)}>"
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
            else write
            for write in self.writes
        ]
        self.do_write(
            config,
            writes,
            self.require_at_least_one_of if input is not None else None,
        )
        return input

    async def _awrite(self, input: Any, config: RunnableConfig) -> None:
        writes = [
            ChannelWriteEntry(write.channel, input, write.skip_none, write.mapper)
            if isinstance(write, ChannelWriteEntry) and write.value is PASSTHROUGH
            else write
            for write in self.writes
        ]
        self.do_write(
            config,
            writes,
            self.require_at_least_one_of if input is not None else None,
        )
        return input

    @staticmethod
    def do_write(
        config: RunnableConfig,
        writes: Sequence[Union[ChannelWriteEntry, Send]],
        require_at_least_one_of: Optional[Sequence[str]] = None,
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
        # split packets and entries
        sends = [(TASKS, packet) for packet in writes if isinstance(packet, Send)]
        entries = [write for write in writes if isinstance(write, ChannelWriteEntry)]
        # process entries into values
        values = [
            write.mapper(write.value) if write.mapper is not None else write.value
            for write in entries
        ]
        values = [
            (write.channel, val)
            for val, write in zip(values, entries)
            if not write.skip_none or val is not None
        ]
        # filter out SKIP_WRITE values
        filtered = [(chan, val) for chan, val in values if val is not SKIP_WRITE]
        if require_at_least_one_of is not None:
            if not {chan for chan, _ in filtered} & set(require_at_least_one_of):
                raise InvalidUpdateError(
                    f"Must write to at least one of {require_at_least_one_of}"
                )
        write: TYPE_SEND = config["configurable"][CONFIG_KEY_SEND]
        write(sends + filtered)

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
