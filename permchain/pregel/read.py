from __future__ import annotations

from typing import Any, Callable, List, Mapping, Optional, Sequence

from langchain_core.pydantic_v1 import Field
from langchain_core.runnables import (
    Runnable,
    RunnableConfig,
    RunnableLambda,
    RunnablePassthrough,
    RunnableSerializable,
)
from langchain_core.runnables.base import (
    Other,
    RunnableBindingBase,
    RunnableEach,
    coerce_to_runnable,
)
from langchain_core.runnables.utils import ConfigurableFieldSpec

from permchain.channels.base import BaseChannel
from permchain.constants import CONFIG_KEY_READ


class ChannelRead(RunnableLambda):
    channel: str

    @property
    def config_specs(self) -> list[ConfigurableFieldSpec]:
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
        self.name = f"ChannelRead<{channel}>"

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


default_bound: RunnablePassthrough = RunnablePassthrough()


class ChannelInvoke(RunnableBindingBase):
    channels: Mapping[None, str] | Mapping[str, str]

    triggers: List[str] = Field(default_factory=list)

    when: Optional[Callable[[Any], bool]] = None

    bound: Runnable[Any, Any] = Field(default=default_bound)

    kwargs: Mapping[str, Any] = Field(default_factory=dict)

    def __init__(
        self,
        channels: Mapping[None, str] | Mapping[str, str],
        triggers: Sequence[str],
        when: Optional[Callable[[Any], bool]] = None,
        *,
        bound: Optional[Runnable[Any, Any]] = None,
        kwargs: Optional[Mapping[str, Any]] = None,
        config: Optional[RunnableConfig] = None,
        **other_kwargs: Any,
    ) -> None:
        super().__init__(
            channels=channels,
            triggers=triggers,
            when=when,
            bound=bound or default_bound,
            kwargs=kwargs or {},
            config=config,
            **other_kwargs,
        )

    def join(self, channels: Sequence[str]) -> ChannelInvoke:
        assert isinstance(channels, list) or isinstance(
            channels, tuple
        ), "channels must be a list or tuple"
        assert all(
            k is not None for k in self.channels.keys()
        ), "all channels must be named when using .join()"
        return ChannelInvoke(
            channels={
                **self.channels,
                **{chan: chan for chan in channels},
            },
            triggers=self.triggers,
            when=self.when,
            bound=self.bound,
            kwargs=self.kwargs,
            config=self.config,
        )

    def __or__(
        self,
        other: Runnable[Any, Other]
        | Callable[[Any], Other]
        | Mapping[str, Runnable[Any, Other] | Callable[[Any], Other]],
    ) -> ChannelInvoke:
        if self.bound is default_bound:
            return ChannelInvoke(
                channels=self.channels,
                triggers=self.triggers,
                when=self.when,
                bound=coerce_to_runnable(other),
                kwargs=self.kwargs,
                config=self.config,
            )
        else:
            return ChannelInvoke(
                channels=self.channels,
                triggers=self.triggers,
                when=self.when,
                # delegate to __or__ in self.bound
                bound=self.bound | other,
                kwargs=self.kwargs,
                config=self.config,
            )

    def __ror__(
        self,
        other: Runnable[Other, Any]
        | Callable[[Any], Other]
        | Mapping[str, Runnable[Other, Any] | Callable[[Other], Any]],
    ) -> RunnableSerializable:
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
    ) -> RunnableSerializable:
        raise NotImplementedError()
