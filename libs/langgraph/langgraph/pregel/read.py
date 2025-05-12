from __future__ import annotations

from collections.abc import AsyncIterator, Iterator, Mapping, Sequence
from functools import cached_property
from typing import (
    Any,
    Callable,
    Union,
)

from langchain_core.runnables import (
    Runnable,
    RunnableConfig,
    RunnablePassthrough,
    RunnableSerializable,
)
from langchain_core.runnables.base import Other, coerce_to_runnable
from langchain_core.runnables.utils import ConfigurableFieldSpec, Input

from langgraph.constants import CONF, CONFIG_KEY_READ
from langgraph.pregel.protocol import PregelProtocol
from langgraph.pregel.retry import RetryPolicy
from langgraph.pregel.utils import find_subgraph_pregel
from langgraph.pregel.write import ChannelWrite
from langgraph.types import CachePolicy
from langgraph.utils.config import merge_configs
from langgraph.utils.runnable import RunnableCallable, RunnableSeq

READ_TYPE = Callable[[Union[str, Sequence[str]], bool], Union[Any, dict[str, Any]]]
INPUT_CACHE_KEY_TYPE = tuple[Callable[..., Any], tuple[str, ...]]


class ChannelRead(RunnableCallable):
    """Implements the logic for reading state from CONFIG_KEY_READ.
    Usable both as a runnable as well as a static method to call imperatively."""

    channel: str | list[str]

    fresh: bool = False

    mapper: Callable[[Any], Any] | None = None

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
        channel: str | list[str],
        *,
        fresh: bool = False,
        mapper: Callable[[Any], Any] | None = None,
        tags: list[str] | None = None,
    ) -> None:
        super().__init__(
            func=self._read,
            afunc=self._aread,
            tags=tags,
            name=None,
            trace=False,
            func_accepts_config=True,
        )
        self.fresh = fresh
        self.mapper = mapper
        self.channel = channel

    def get_name(self, suffix: str | None = None, *, name: str | None = None) -> str:
        if name:
            pass
        elif isinstance(self.channel, str):
            name = f"ChannelRead<{self.channel}>"
        else:
            name = f"ChannelRead<{','.join(self.channel)}>"
        return super().get_name(suffix, name=name)

    def _read(self, _: Any, config: RunnableConfig) -> Any:
        return self.do_read(
            config, select=self.channel, fresh=self.fresh, mapper=self.mapper
        )

    async def _aread(self, _: Any, config: RunnableConfig) -> Any:
        return self.do_read(
            config, select=self.channel, fresh=self.fresh, mapper=self.mapper
        )

    @staticmethod
    def do_read(
        config: RunnableConfig,
        *,
        select: str | list[str],
        fresh: bool = False,
        mapper: Callable[[Any], Any] | None = None,
    ) -> Any:
        try:
            read: READ_TYPE = config[CONF][CONFIG_KEY_READ]
        except KeyError:
            raise RuntimeError(
                "Not configured with a read function"
                "Make sure to call in the context of a Pregel process"
            )
        if mapper:
            return mapper(read(select, fresh))
        else:
            return read(select, fresh)


DEFAULT_BOUND: RunnablePassthrough = RunnablePassthrough()


class PregelNode(Runnable):
    """A node in a Pregel graph. This won't be invoked as a runnable by the graph
    itself, but instead acts as a container for the components necessary to make
    a PregelExecutableTask for a node."""

    channels: list[str] | Mapping[str, str]
    """The channels that will be passed as input to `bound`.
    If a list, the node will be invoked with the first of that isn't empty.
    If a dict, the keys are the names of the channels, and the values are the keys
    to use in the input to `bound`."""

    triggers: list[str]
    """If any of these channels is written to, this node will be triggered in
    the next step."""

    mapper: Callable[[Any], Any] | None
    """A function to transform the input before passing it to `bound`."""

    writers: list[Runnable]
    """A list of writers that will be executed after `bound`, responsible for
    taking the output of `bound` and writing it to the appropriate channels."""

    bound: Runnable[Any, Any]
    """The main logic of the node. This will be invoked with the input from 
    `channels`."""

    retry_policy: Sequence[RetryPolicy] | None
    """The retry policies to use when invoking the node."""

    cache_policy: CachePolicy | None
    """The cache policy to use when invoking the node."""

    tags: Sequence[str] | None
    """Tags to attach to the node for tracing."""

    metadata: Mapping[str, Any] | None
    """Metadata to attach to the node for tracing."""

    subgraphs: Sequence[PregelProtocol]
    """Subgraphs used by the node."""

    def __init__(
        self,
        *,
        channels: list[str] | Mapping[str, str],
        triggers: Sequence[str],
        mapper: Callable[[Any], Any] | None = None,
        writers: list[Runnable] | None = None,
        tags: list[str] | None = None,
        metadata: Mapping[str, Any] | None = None,
        bound: Runnable[Any, Any] | None = None,
        retry_policy: RetryPolicy | Sequence[RetryPolicy] | None = None,
        cache_policy: CachePolicy | None = None,
        subgraphs: Sequence[PregelProtocol] | None = None,
    ) -> None:
        self.channels = channels
        self.triggers = list(triggers)
        self.mapper = mapper
        self.writers = writers or []
        self.bound = bound if bound is not None else DEFAULT_BOUND
        self.cache_policy = cache_policy
        if isinstance(retry_policy, RetryPolicy):
            self.retry_policy = (retry_policy,)
        else:
            self.retry_policy = retry_policy
        self.tags = tags
        self.metadata = metadata
        if subgraphs is not None:
            self.subgraphs = subgraphs
        elif self.bound is not DEFAULT_BOUND:
            try:
                subgraph = find_subgraph_pregel(self.bound)
            except Exception:
                subgraph = None
            if subgraph:
                self.subgraphs = [subgraph]
            else:
                self.subgraphs = []
        else:
            self.subgraphs = []

    def copy(self, update: dict[str, Any]) -> PregelNode:
        attrs = {**self.__dict__, **update}
        # Drop the cached properties
        attrs.pop("flat_writers", None)
        attrs.pop("node", None)
        attrs.pop("input_cache_key", None)
        return PregelNode(**attrs)

    @cached_property
    def flat_writers(self) -> list[Runnable]:
        """Get writers with optimizations applied. Dedupes consecutive ChannelWrites."""
        writers = self.writers.copy()
        while (
            len(writers) > 1
            and isinstance(writers[-1], ChannelWrite)
            and isinstance(writers[-2], ChannelWrite)
        ):
            # we can combine writes if they are consecutive
            # careful to not modify the original writers list or ChannelWrite
            writers[-2] = ChannelWrite(
                writes=writers[-2].writes + writers[-1].writes,
                tags=writers[-2].tags,
            )
            writers.pop()
        return writers

    @cached_property
    def node(self) -> Runnable[Any, Any] | None:
        """Get a runnable that combines `bound` and `writers`."""
        writers = self.flat_writers
        if self.bound is DEFAULT_BOUND and not writers:
            return None
        elif self.bound is DEFAULT_BOUND and len(writers) == 1:
            return writers[0]
        elif self.bound is DEFAULT_BOUND:
            return RunnableSeq(*writers)
        elif writers:
            return RunnableSeq(self.bound, *writers)
        else:
            return self.bound

    @cached_property
    def input_cache_key(self) -> INPUT_CACHE_KEY_TYPE:
        """Get a cache key for the input to the node.
        This is used to avoid calculating the same input multiple times."""
        return (
            self.mapper,
            tuple(f"{key}:{value}" for key, value in self.channels.items())
            if isinstance(self.channels, dict)
            else tuple(self.channels),
        )

    def join(self, channels: Sequence[str]) -> PregelNode:
        assert isinstance(channels, list) or isinstance(
            channels, tuple
        ), "channels must be a list or tuple"
        assert isinstance(
            self.channels, dict
        ), "all channels must be named when using .join()"
        return self.copy(
            update=dict(
                channels={
                    **self.channels,
                    **{chan: chan for chan in channels},
                }
            ),
        )

    def __or__(
        self,
        other: Runnable[Any, Other]
        | Callable[[Any], Other]
        | Mapping[str, Runnable[Any, Other] | Callable[[Any], Other]],
    ) -> PregelNode:
        if isinstance(other, Runnable) and ChannelWrite.is_writer(other):
            return self.copy(update=dict(writers=[*self.writers, other]))
        elif self.bound is DEFAULT_BOUND:
            return self.copy(update=dict(bound=coerce_to_runnable(other)))
        else:
            return self.copy(update=dict(bound=RunnableSeq(self.bound, other)))

    def pipe(
        self,
        *others: Runnable[Any, Other] | Callable[[Any], Other],
        name: str | None = None,
    ) -> RunnableSerializable[Any, Other]:
        for other in others:
            self = self | other
        return self

    def __ror__(
        self,
        other: Runnable[Other, Any]
        | Callable[[Any], Other]
        | Mapping[str, Runnable[Other, Any] | Callable[[Other], Any]],
    ) -> RunnableSerializable:
        raise NotImplementedError()

    def invoke(
        self,
        input: Input,
        config: RunnableConfig | None = None,
        **kwargs: Any | None,
    ) -> Any:
        return self.bound.invoke(
            input,
            merge_configs({"metadata": self.metadata, "tags": self.tags}, config),
            **kwargs,
        )

    async def ainvoke(
        self,
        input: Input,
        config: RunnableConfig | None = None,
        **kwargs: Any | None,
    ) -> Any:
        return await self.bound.ainvoke(
            input,
            merge_configs({"metadata": self.metadata, "tags": self.tags}, config),
            **kwargs,
        )

    def stream(
        self,
        input: Input,
        config: RunnableConfig | None = None,
        **kwargs: Any | None,
    ) -> Iterator[Any]:
        yield from self.bound.stream(
            input,
            merge_configs({"metadata": self.metadata, "tags": self.tags}, config),
            **kwargs,
        )

    async def astream(
        self,
        input: Input,
        config: RunnableConfig | None = None,
        **kwargs: Any | None,
    ) -> AsyncIterator[Any]:
        async for item in self.bound.astream(
            input,
            merge_configs({"metadata": self.metadata, "tags": self.tags}, config),
            **kwargs,
        ):
            yield item
