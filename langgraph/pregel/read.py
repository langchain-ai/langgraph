from __future__ import annotations

from typing import Any, Callable, Mapping, Optional, Sequence, Type, Union

from langchain_core.pydantic_v1 import Field
from langchain_core.runnables import (
    Runnable,
    RunnableConfig,
    RunnablePassthrough,
    RunnableSequence,
    RunnableSerializable,
)
from langchain_core.runnables.base import Other, RunnableBindingBase, coerce_to_runnable
from langchain_core.runnables.config import merge_configs
from langchain_core.runnables.utils import ConfigurableFieldSpec

from langgraph.constants import CONFIG_KEY_READ
from langgraph.managed.base import ManagedValue
from langgraph.pregel.write import ChannelWrite
from langgraph.utils import RunnableCallable

READ_TYPE = Callable[[str, bool], Union[Any, dict[str, Any]]]


class ChannelRead(RunnableCallable):
    channel: Union[str, list[str]]

    fresh: bool = False

    mapper: Optional[Callable[[Any], Any]] = None

    @property
    def config_specs(self) -> list[ConfigurableFieldSpec]:
        return [
            ConfigurableFieldSpec(
                id=CONFIG_KEY_READ,
                name=CONFIG_KEY_READ,
                description=None,
                default=None,
                annotation=None,
            ),
        ]

    def __init__(
        self,
        channel: Union[str, list[str]],
        *,
        fresh: bool = False,
        mapper: Optional[Callable[[Any], Any]] = None,
        tags: Optional[list[str]] = None,
    ) -> None:
        super().__init__(func=self._read, afunc=self._aread, tags=tags, name=None)
        self.fresh = fresh
        self.mapper = mapper
        self.channel = channel

    def get_name(
        self, suffix: Optional[str] = None, *, name: Optional[str] = None
    ) -> str:
        if name:
            pass
        elif isinstance(self.channel, str):
            name = f"ChannelRead<{self.channel}>"
        else:
            name = f"ChannelRead<{','.join(self.channel)}>"
        return super().get_name(suffix, name=name)

    def _read(self, _: Any, config: RunnableConfig) -> Any:
        return self.do_read(
            config, channel=self.channel, fresh=self.fresh, mapper=self.mapper
        )

    async def _aread(self, _: Any, config: RunnableConfig) -> Any:
        return self.do_read(
            config, channel=self.channel, fresh=self.fresh, mapper=self.mapper
        )

    @staticmethod
    def do_read(
        config: RunnableConfig,
        *,
        channel: Union[str, list[str]],
        fresh: bool = False,
        mapper: Optional[Callable[[Any], Any]] = None,
    ) -> Any:
        try:
            read: READ_TYPE = config["configurable"][CONFIG_KEY_READ]
        except KeyError:
            raise RuntimeError(
                "Not configured with a read function"
                "Make sure to call in the context of a Pregel process"
            )
        if mapper:
            return mapper(read(channel, fresh))
        else:
            return read(channel, fresh)


DEFAULT_BOUND: RunnablePassthrough = RunnablePassthrough()


class PregelNode(RunnableBindingBase):
    channels: Union[list[str], Mapping[str, Union[str, Type[ManagedValue]]]]

    triggers: list[str] = Field(default_factory=list)

    mapper: Optional[Callable[[Any], Any]] = None

    writers: list[Runnable] = Field(default_factory=list)

    bound: Runnable[Any, Any] = Field(default=DEFAULT_BOUND)

    kwargs: Mapping[str, Any] = Field(default_factory=dict)

    def get_writers(self) -> list[Runnable]:
        """Get writers with optimizations applied."""
        writers = self.writers.copy()
        while (
            len(writers) > 1
            and isinstance(writers[-1], ChannelWrite)
            and isinstance(writers[-2], ChannelWrite)
        ):
            # we can combine writes if they are consecutive
            writers[-2].writes += writers[-1].writes
            writers.pop()
        return writers

    def get_node(self) -> Optional[Runnable[Any, Any]]:
        writers = self.get_writers()
        if self.bound is DEFAULT_BOUND and not writers:
            return None
        elif self.bound is DEFAULT_BOUND and len(writers) == 1:
            return writers[0]
        elif self.bound is DEFAULT_BOUND:
            return RunnableSequence(*writers)
        elif writers:
            return RunnableSequence(self.bound, *writers)
        else:
            return self.bound

    def __init__(
        self,
        *,
        channels: Union[list[str], Mapping[str, str]],
        triggers: Sequence[str],
        mapper: Optional[Callable[[Any], Any]] = None,
        writers: Optional[list[Runnable]] = None,
        tags: Optional[list[str]] = None,
        bound: Optional[Runnable[Any, Any]] = None,
        kwargs: Optional[Mapping[str, Any]] = None,
        config: Optional[RunnableConfig] = None,
        **other_kwargs: Any,
    ) -> None:
        super().__init__(
            channels=channels,
            triggers=triggers,
            mapper=mapper,
            writers=writers or [],
            bound=bound or DEFAULT_BOUND,
            kwargs=kwargs or {},
            config=merge_configs(config, {"tags": tags or []}),
            **other_kwargs,
        )

    def __repr_args__(self) -> Any:
        return [(k, v) for k, v in super().__repr_args__() if k != "bound"]

    def join(self, channels: Sequence[str]) -> PregelNode:
        assert isinstance(channels, list) or isinstance(
            channels, tuple
        ), "channels must be a list or tuple"
        assert isinstance(
            self.channels, dict
        ), "all channels must be named when using .join()"
        return PregelNode(
            channels={
                **self.channels,
                **{chan: chan for chan in channels},
            },
            triggers=self.triggers,
            mapper=self.mapper,
            writers=self.writers,
            bound=self.bound,
            kwargs=self.kwargs,
            config=self.config,
        )

    def __or__(
        self,
        other: Union[
            Runnable[Any, Other],
            Callable[[Any], Other],
            Mapping[str, Runnable[Any, Other] | Callable[[Any], Other]],
        ],
    ) -> PregelNode:
        if ChannelWrite.is_writer(other):
            return PregelNode(
                channels=self.channels,
                triggers=self.triggers,
                mapper=self.mapper,
                writers=[*self.writers, other],
                bound=self.bound,
                kwargs=self.kwargs,
                config=self.config,
            )
        elif self.bound is DEFAULT_BOUND:
            return PregelNode(
                channels=self.channels,
                triggers=self.triggers,
                mapper=self.mapper,
                writers=self.writers,
                bound=coerce_to_runnable(other),
                kwargs=self.kwargs,
                config=self.config,
            )
        else:
            return PregelNode(
                channels=self.channels,
                triggers=self.triggers,
                mapper=self.mapper,
                writers=self.writers,
                # delegate to __or__ in self.bound
                bound=self.bound | other,
                kwargs=self.kwargs,
                config=self.config,
            )

    def pipe(
        self,
        *others: Runnable[Any, Other] | Callable[[Any], Other],
        name: Optional[str] = None,
    ) -> RunnableSerializable[Any, Other]:
        for other in others:
            self = self | other
        return self

    def __ror__(
        self,
        other: Union[
            Runnable[Other, Any],
            Callable[[Any], Other],
            Mapping[str, Union[Runnable[Other, Any], Callable[[Other], Any]]],
        ],
    ) -> RunnableSerializable:
        raise NotImplementedError()
