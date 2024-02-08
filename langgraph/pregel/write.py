from __future__ import annotations

import asyncio
from typing import Any, Callable, Optional, Sequence

from langchain_core.runnables import (
    Runnable,
    RunnableConfig,
    RunnablePassthrough,
)
from langchain_core.runnables.utils import ConfigurableFieldSpec

from langgraph.constants import CONFIG_KEY_SEND

TYPE_SEND = Callable[[Sequence[tuple[str, Any]]], None]


SKIP_WRITE = object()


class ChannelWrite(RunnablePassthrough):
    channels: Sequence[tuple[str, Optional[Runnable], bool]]
    """
    Mapping of write channels to Runnables that return the value to be written,
    or None to skip writing.
    """

    class Config:
        arbitrary_types_allowed = True

    def __init__(
        self,
        *,
        channels: Sequence[tuple[str, Optional[Runnable], bool]],
    ):
        super().__init__(func=self._write, afunc=self._awrite, channels=channels)
        self.name = f"ChannelWrite<{','.join(chan for chan, _, _ in self.channels)}>"

    def __repr_args__(self) -> Any:
        return [("channels", self.channels)]

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
            (chan, r.invoke(input, config) if r else input)
            for chan, r, _ in self.channels
        ]
        values = [
            write
            for write, (_, _, skip_none) in zip(values, self.channels)
            if not skip_none or write[1] is not None
        ]

        self.do_write(config, **dict(values))

    async def _awrite(self, input: Any, config: RunnableConfig) -> None:
        values = await asyncio.gather(
            *(
                r.ainvoke(input, config) if r else _mk_future(input)
                for _, r, _ in self.channels
            )
        )
        values = [
            (chan, val)
            for val, (chan, _, skip_none) in zip(values, self.channels)
            if not skip_none or val is not None
        ]

        self.do_write(config, **dict(values))

    @staticmethod
    def do_write(config: RunnableConfig, **values: Any) -> None:
        write: TYPE_SEND = config["configurable"][CONFIG_KEY_SEND]
        write([(chan, val) for chan, val in values.items() if val is not SKIP_WRITE])


def _mk_future(val: Any) -> asyncio.Future:
    fut = asyncio.Future()
    fut.set_result(val)
    return fut
