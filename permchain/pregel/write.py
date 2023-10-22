from __future__ import annotations

from typing import Any, Callable, Optional, Sequence

from langchain.schema.runnable import (
    Runnable,
    RunnableConfig,
    RunnableLambda,
)
from langchain.schema.runnable.utils import ConfigurableFieldSpec

from permchain.pregel.constants import CONFIG_KEY_SEND, CONFIG_KEY_STEP

TYPE_SEND = Callable[[Sequence[tuple[str, Any]]], None]


# TODO switch to RunnablePassthrough after updating langchain
class PregelSink(RunnableLambda):
    channels: Sequence[tuple[str, Runnable]]
    """
    Mapping of write channels to Runnables that return the value to be written,
    or None to skip writing.
    """

    max_steps: Optional[int]

    def __init__(
        self,
        *,
        channels: Sequence[tuple[str, Runnable]],
        max_steps: Optional[int] = None,
    ):
        super().__init__(func=self._write, afunc=self._awrite)  # type: ignore[arg-type]
        self.channels = channels
        self.max_steps = max_steps

    @property
    def config_specs(self) -> Sequence[ConfigurableFieldSpec]:
        return [
            ConfigurableFieldSpec(
                id=CONFIG_KEY_STEP,
                name=CONFIG_KEY_STEP,
                description=None,
                default=None,
                annotation=int,
            ),
            ConfigurableFieldSpec(
                id=CONFIG_KEY_SEND,
                name=CONFIG_KEY_SEND,
                description=None,
                default=None,
                annotation=TYPE_SEND,
            ),
        ]

    def _write(self, input: Any, config: RunnableConfig) -> None:
        step: int = config["configurable"][CONFIG_KEY_STEP]

        if self.max_steps is not None and step >= self.max_steps:
            return

        write: TYPE_SEND = config["configurable"][CONFIG_KEY_SEND]

        values = [(chan, r.invoke(input, config)) for chan, r in self.channels]

        write([(chan, val) for chan, val in values if val is not None])

        return input

    async def _awrite(self, input: Any, config: RunnableConfig) -> None:
        step: int = config["configurable"][CONFIG_KEY_STEP]

        if self.max_steps is not None and step >= self.max_steps:
            return

        write: TYPE_SEND = config["configurable"][CONFIG_KEY_SEND]

        values = [(chan, await r.ainvoke(input, config)) for chan, r in self.channels]

        write([(chan, val) for chan, val in values if val is not None])

        return input
