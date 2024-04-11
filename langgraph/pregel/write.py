from __future__ import annotations

import asyncio
from typing import Any, Callable, NamedTuple, Optional, Sequence, TypeVar, Union

from langchain_core.runnables import Runnable, RunnableConfig
from langchain_core.runnables.utils import ConfigurableFieldSpec

from langgraph.constants import CONFIG_KEY_SEND
from langgraph.utils import RunnableCallable

TYPE_SEND = Callable[[Sequence[tuple[str, Any]]], None]
R = TypeVar("R", bound=Runnable)


SKIP_WRITE = object()


class ChannelWriteEntry(NamedTuple):
    channel: str
    value: Optional[Union[Any, Runnable]] = None
    skip_none: bool = False


class ChannelWrite(RunnableCallable):
    writes: Sequence[ChannelWriteEntry]
    """
    Sequence of write entries, each of which is a tuple of:
    - channel name
    - runnable to map input, or None to use the input, or any other value to use instead
    - whether to skip writing if the mapped value is None
    """

    def __init__(
        self, writes: Sequence[ChannelWriteEntry], *, tags: Optional[list[str]] = None
    ):
        super().__init__(func=self._write, afunc=self._awrite, name=None, tags=tags)
        self.writes = writes

    def __repr_args__(self) -> Any:
        return [("writes", self.writes)]

    def get_name(
        self, suffix: Optional[str] = None, *, name: Optional[str] = None
    ) -> str:
        if not name:
            name = f"ChannelWrite<{','.join(chan for chan, _, _ in self.writes)}>"
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
        values = [
            (
                chan,
                r.invoke(input, config)
                if isinstance(r, Runnable)
                else r
                if r is not None
                else input,
            )
            for chan, r, _ in self.writes
        ]
        values = [
            write
            for write, (_, _, skip_none) in zip(values, self.writes)
            if not skip_none or write[1] is not None
        ]
        self.do_write(config, **dict(values))
        return input

    async def _awrite(self, input: Any, config: RunnableConfig) -> None:
        values = await asyncio.gather(
            *(
                r.ainvoke(input, config)
                if isinstance(r, Runnable)
                else _mk_future(r)
                if r is not None
                else _mk_future(input)
                for _, r, _ in self.writes
            )
        )
        values = [
            (chan, val)
            for val, (chan, _, skip_none) in zip(values, self.writes)
            if not skip_none or val is not None
        ]
        self.do_write(config, **dict(values))
        return input

    @staticmethod
    def do_write(config: RunnableConfig, **values: Any) -> None:
        write: TYPE_SEND = config["configurable"][CONFIG_KEY_SEND]
        write([(chan, val) for chan, val in values.items() if val is not SKIP_WRITE])

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
