from __future__ import annotations

import asyncio
from typing import (
    Any,
    Callable,
    List,
    NamedTuple,
    Optional,
    Sequence,
    Tuple,
    TypeVar,
    Union,
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
    value: Any = PASSTHROUGH
    skip_none: bool = False
    mapper: Optional[Runnable] = None


class ChannelWrite(RunnableCallable):
    writes: Sequence[Union[ChannelWriteEntry, Send]]
    """
    Sequence of write entries, each of which is a tuple of:
    - channel name
    - runnable to map input, or None to use the input, or any other value to use instead
    - whether to skip writing if the mapped value is None
    """
    require_at_least_one_of: Optional[Sequence[str]]
    """
    If defined, at least one of these channels must be written to.
    """

    def __init__(
        self,
        writes: Sequence[Union[ChannelWriteEntry, Send]],
        *,
        tags: Optional[list[str]] = None,
        require_at_least_one_of: Optional[Sequence[str]] = None,
    ):
        super().__init__(func=self._write, afunc=self._awrite, name=None, tags=tags)
        self.writes = writes
        self.require_at_least_one_of = require_at_least_one_of

    def __repr_args__(self) -> Any:
        return [("writes", self.writes)]

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
        # split packets and entries
        writes = [(TASKS, packet) for packet in self.writes if isinstance(packet, Send)]
        entries = [
            write for write in self.writes if isinstance(write, ChannelWriteEntry)
        ]
        for entry in entries:
            if entry.channel == TASKS:
                raise InvalidUpdateError("Cannot write to the reserved channel TASKS")
        # process entries into values
        values = [
            input if write.value is PASSTHROUGH else write.value for write in entries
        ]
        values = [
            val if write.mapper is None else write.mapper.invoke(val, config)
            for val, write in zip(values, entries)
        ]
        values = [
            (write.channel, val)
            for val, write in zip(values, entries)
            if not write.skip_none or val is not None
        ]
        # write packets and values
        self.do_write(
            config,
            writes + values,
            self.require_at_least_one_of if input is not None else None,
        )
        return input

    async def _awrite(self, input: Any, config: RunnableConfig) -> None:
        # split packets and entries
        writes = [(TASKS, packet) for packet in self.writes if isinstance(packet, Send)]
        entries = [
            write for write in self.writes if isinstance(write, ChannelWriteEntry)
        ]
        for entry in entries:
            if entry.channel == TASKS:
                raise InvalidUpdateError("Cannot write to the reserved channel TASKS")
        # process entries into values
        values = [
            input if write.value is PASSTHROUGH else write.value for write in entries
        ]
        values = await asyncio.gather(
            *(
                _mk_future(val)
                if write.mapper is None
                else write.mapper.ainvoke(val, config)
                for val, write in zip(values, entries)
            )
        )
        values = [
            (write.channel, val)
            for val, write in zip(values, entries)
            if not write.skip_none or val is not None
        ]
        # write packets and values
        self.do_write(
            config,
            writes + values,
            self.require_at_least_one_of if input is not None else None,
        )
        return input

    @staticmethod
    def do_write(
        config: RunnableConfig,
        values: List[Tuple[str, Any]],
        require_at_least_one_of: Optional[Sequence[str]] = None,
    ) -> None:
        filtered = [(chan, val) for chan, val in values if val is not SKIP_WRITE]
        if require_at_least_one_of is not None:
            if not {chan for chan, _ in filtered} & set(require_at_least_one_of):
                raise InvalidUpdateError(
                    f"Must write to at least one of {require_at_least_one_of}"
                )
        write: TYPE_SEND = config["configurable"][CONFIG_KEY_SEND]
        write(filtered)

    @staticmethod
    def is_writer(runnable: Runnable) -> bool:
        return (
            isinstance(runnable, ChannelWrite)
            or getattr(runnable, "_is_channel_writer", False) is True
        )

    @staticmethod
    def register_writer(runnable: R) -> R:
        # using object.__setattr__ to work around objects that override __setattr__
        # eg. pydantic models and dataclasses
        object.__setattr__(runnable, "_is_channel_writer", True)
        return runnable


def _mk_future(val: Any) -> asyncio.Future:
    fut = asyncio.Future()
    fut.set_result(val)
    return fut
