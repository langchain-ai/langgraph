from __future__ import annotations

from typing import Any, Callable, Mapping, Optional, Sequence

from langchain.pydantic_v1 import Field
from langchain.schema.runnable import (
    Runnable,
    RunnableConfig,
    RunnableLambda,
    RunnablePassthrough,
)
from langchain.schema.runnable.base import (
    Other,
    RunnableBindingBase,
    RunnableEach,
    coerce_to_runnable,
)
from langchain.schema.runnable.utils import ConfigurableFieldSpec

from permchain.channels.base import BaseChannel
from permchain.constants import CONFIG_KEY_READ


class ChannelRead(RunnableLambda):
    channel: str

    @property
    def config_specs(self) -> Sequence[ConfigurableFieldSpec]:
        return [
            ConfigurableFieldSpec(
                id=CONFIG_KEY_READ,
                name=CONFIG_KEY_READ,
                description=None,
                default=None,
                annotation=Callable[[BaseChannel], Any],
            ),
        ]

    def __init__(self, channel: str) -> None:
        super().__init__(func=self._read, afunc=self._aread)
        self.channel = channel

    def _read(self, _: Any, config: RunnableConfig) -> Any:
        try:
            read: Callable[[str], Any] = config["configurable"][CONFIG_KEY_READ]
        except KeyError:
            raise RuntimeError(
                f"Runnable {self} is not configured with a read function"
                "Make sure to call in the context of a Pregel process"
            )
        return read(self.channel)

    async def _aread(self, _: Any, config: RunnableConfig) -> Any:
        try:
            read: Callable[[str], Any] = config["configurable"][CONFIG_KEY_READ]
        except KeyError:
            raise RuntimeError(
                f"Runnable {self} is not configured with a read function"
                "Make sure to call in the context of a Pregel process"
            )
        return read(self.channel)


default_bound = RunnablePassthrough()


class ChannelInvoke(RunnableBindingBase):
    channels: Mapping[None, str] | Mapping[str, str]

    bound: Runnable[Any, Any] = Field(default=default_bound)

    kwargs: Mapping[str, Any] = Field(default_factory=dict)

    def __init__(
        self,
        channels: Mapping[None, str] | Mapping[str, str],
        *,
        bound: Optional[Runnable[Any, Any]] = None,
        kwargs: Optional[Mapping[str, Any]] = None,
        config: Optional[RunnableConfig] = None,
        **other_kwargs: Any,
    ) -> None:
        super().__init__(
            channels=channels,
            bound=bound or default_bound,
            kwargs=kwargs or {},
            config=config,
            **other_kwargs,
        )

    def join(self, channels: Sequence[str]) -> ChannelInvoke:
        assert isinstance(channels, list) or isinstance(
            channels, tuple
        ), "channels must be a list or tuple"
        joiner = RunnablePassthrough.assign(
            **{chan: ChannelRead(chan) for chan in channels}
        )
        if self.bound is default_bound:
            return ChannelInvoke(channels=self.channels, bound=joiner)
        else:
            return ChannelInvoke(channels=self.channels, bound=self.bound | joiner)

    def __or__(
        self,
        other: Runnable[Any, Other]
        | Callable[[Any], Other]
        | Mapping[str, Runnable[Any, Other] | Callable[[Any], Other]],
    ) -> ChannelInvoke:
        if self.bound is default_bound:
            return ChannelInvoke(
                channels=self.channels, bound=coerce_to_runnable(other)
            )
        else:
            # delegate to __or__ in self.bound
            return ChannelInvoke(channels=self.channels, bound=self.bound | other)

    def __ror__(
        self,
        other: Runnable[Other, Any]
        | Callable[[Any], Other]
        | Mapping[str, Runnable[Other, Any] | Callable[[Other], Any]],
    ) -> Runnable:
        raise NotImplementedError()


class ChannelBatch(RunnableEach):
    channel: str

    key: Optional[str]

    bound: Runnable[Any, Any] = Field(default=default_bound)

    def join(self, channels: Sequence[str]) -> ChannelBatch:
        if self.key is None:
            raise ValueError(
                "Cannot join() additional channels without a key."
                " Pass a key arg to Channel.subscribe_to_each()."
            )

        joiner = RunnablePassthrough.assign(
            **{chan: ChannelRead(chan) for chan in channels}
        )
        if self.bound is default_bound:
            return ChannelBatch(channel=self.channel, key=self.key, bound=joiner)
        else:
            return ChannelBatch(
                channel=self.channel, key=self.key, bound=self.bound | joiner
            )

    def __or__(  # type: ignore[override]
        self,
        other: Runnable[Any, Other]
        | Callable[[Any], Other]
        | Mapping[str, Runnable[Any, Other] | Callable[[Any], Other]],
    ) -> ChannelBatch:
        if self.bound is default_bound:
            return ChannelBatch(
                channel=self.channel, key=self.key, bound=coerce_to_runnable(other)
            )
        else:
            # delegate to __or__ in self.bound
            return ChannelBatch(
                channel=self.channel, key=self.key, bound=self.bound | other
            )

    def __ror__(
        self,
        other: Runnable[Other, Any]
        | Callable[[Any], Other]
        | Mapping[str, Runnable[Other, Any] | Callable[[Other], Any]],
    ) -> Runnable:
        raise NotImplementedError()
