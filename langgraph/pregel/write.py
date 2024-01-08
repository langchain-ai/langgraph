from __future__ import annotations

from typing import Any, Callable, Sequence

from langchain_core.runnables import (
    Runnable,
    RunnableConfig,
    RunnablePassthrough,
)
from langchain_core.runnables.utils import ConfigurableFieldSpec

from langgraph.constants import CONFIG_KEY_SEND

TYPE_SEND = Callable[[Sequence[tuple[str, Any]]], None]


class ChannelWrite(RunnablePassthrough):
    channels: Sequence[tuple[str, Runnable | None]]
    """
    Mapping of write channels to Runnables that return the value to be written,
    or None to skip writing.
    """

    class Config:
        arbitrary_types_allowed = True

    def __init__(
        self,
        *,
        channels: Sequence[tuple[str, Runnable | None]],
    ):
        super().__init__(func=self._write, afunc=self._awrite, channels=channels)
        self.name = f"ChannelWrite<{','.join(chan for chan, _ in self.channels)}>"

    @property
    def config_specs(self) -> list[ConfigurableFieldSpec]:
        return [
            ConfigurableFieldSpec(
                id=CONFIG_KEY_SEND,
                name=CONFIG_KEY_SEND,
                description=None,
                default=None,
                annotation=TYPE_SEND,
            ),
        ]

    def _write(self, input: Any, config: RunnableConfig) -> None:
        values = [
            (chan, r.invoke(input, config) if r else input) for chan, r in self.channels
        ]

        self.do_write(config, **dict(values))

    async def _awrite(self, input: Any, config: RunnableConfig) -> None:
        values = [
            (chan, await r.ainvoke(input, config) if r else input)
            for chan, r in self.channels
        ]

        self.do_write(config, **dict(values))

    @staticmethod
    def do_write(config: RunnableConfig, **values: Any) -> None:
        write: TYPE_SEND = config["configurable"][CONFIG_KEY_SEND]
        write([(chan, val) for chan, val in values.items() if val is not None])
