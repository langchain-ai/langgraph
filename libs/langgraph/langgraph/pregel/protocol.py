from __future__ import annotations

from abc import abstractmethod
from collections.abc import AsyncIterator, Callable, Iterator, Sequence
from typing import Any, Generic, Literal, cast, overload

from langchain_core.runnables import Runnable, RunnableConfig
from langchain_core.runnables.graph import Graph as DrawableGraph
from typing_extensions import Self

from langgraph.types import (
    All,
    Command,
    GraphOutput,
    StateSnapshot,
    StateUpdate,
    StreamMode,
    StreamPart,
)
from langgraph.typing import ContextT, InputT, OutputT, StateT

__all__ = ("PregelProtocol", "StreamProtocol")


class PregelProtocol(Runnable[InputT, Any], Generic[StateT, ContextT, InputT, OutputT]):
    @abstractmethod
    def with_config(
        self, config: RunnableConfig | None = None, **kwargs: Any
    ) -> Self: ...

    @abstractmethod
    def get_graph(
        self,
        config: RunnableConfig | None = None,
        *,
        xray: int | bool = False,
    ) -> DrawableGraph: ...

    @abstractmethod
    async def aget_graph(
        self,
        config: RunnableConfig | None = None,
        *,
        xray: int | bool = False,
    ) -> DrawableGraph: ...

    @abstractmethod
    def get_state(
        self, config: RunnableConfig, *, subgraphs: bool = False
    ) -> StateSnapshot: ...

    @abstractmethod
    async def aget_state(
        self, config: RunnableConfig, *, subgraphs: bool = False
    ) -> StateSnapshot: ...

    @abstractmethod
    def get_state_history(
        self,
        config: RunnableConfig,
        *,
        filter: dict[str, Any] | None = None,
        before: RunnableConfig | None = None,
        limit: int | None = None,
    ) -> Iterator[StateSnapshot]: ...

    @abstractmethod
    def aget_state_history(
        self,
        config: RunnableConfig,
        *,
        filter: dict[str, Any] | None = None,
        before: RunnableConfig | None = None,
        limit: int | None = None,
    ) -> AsyncIterator[StateSnapshot]: ...

    @abstractmethod
    def bulk_update_state(
        self,
        config: RunnableConfig,
        updates: Sequence[Sequence[StateUpdate]],
    ) -> RunnableConfig: ...

    @abstractmethod
    async def abulk_update_state(
        self,
        config: RunnableConfig,
        updates: Sequence[Sequence[StateUpdate]],
    ) -> RunnableConfig: ...

    @abstractmethod
    def update_state(
        self,
        config: RunnableConfig,
        values: dict[str, Any] | Any | None,
        as_node: str | None = None,
    ) -> RunnableConfig: ...

    @abstractmethod
    async def aupdate_state(
        self,
        config: RunnableConfig,
        values: dict[str, Any] | Any | None,
        as_node: str | None = None,
    ) -> RunnableConfig: ...

    @overload
    @abstractmethod
    def stream(
        self,
        input: InputT | Command | None,
        config: RunnableConfig | None = None,
        *,
        context: ContextT | None = None,
        stream_mode: StreamMode | list[StreamMode] | None = None,
        interrupt_before: All | Sequence[str] | None = None,
        interrupt_after: All | Sequence[str] | None = None,
        subgraphs: bool = False,
        version: Literal["v2"],
    ) -> Iterator[StreamPart[OutputT, StateT]]: ...

    @overload
    @abstractmethod
    def stream(
        self,
        input: InputT | Command | None,
        config: RunnableConfig | None = None,
        *,
        context: ContextT | None = None,
        stream_mode: StreamMode | list[StreamMode] | None = None,
        interrupt_before: All | Sequence[str] | None = None,
        interrupt_after: All | Sequence[str] | None = None,
        subgraphs: bool = False,
        version: Literal["v1"] = ...,
    ) -> Iterator[dict[str, Any] | Any]: ...

    @abstractmethod
    def stream(
        self,
        input: InputT | Command | None,
        config: RunnableConfig | None = None,
        *,
        context: ContextT | None = None,
        stream_mode: StreamMode | list[StreamMode] | None = None,
        interrupt_before: All | Sequence[str] | None = None,
        interrupt_after: All | Sequence[str] | None = None,
        subgraphs: bool = False,
        version: Literal["v1", "v2"] = "v1",
    ) -> Iterator[dict[str, Any] | Any]: ...

    @overload
    @abstractmethod
    def astream(
        self,
        input: InputT | Command | None,
        config: RunnableConfig | None = None,
        *,
        context: ContextT | None = None,
        stream_mode: StreamMode | list[StreamMode] | None = None,
        interrupt_before: All | Sequence[str] | None = None,
        interrupt_after: All | Sequence[str] | None = None,
        subgraphs: bool = False,
        version: Literal["v2"],
    ) -> AsyncIterator[StreamPart[OutputT, StateT]]: ...

    @overload
    @abstractmethod
    def astream(
        self,
        input: InputT | Command | None,
        config: RunnableConfig | None = None,
        *,
        context: ContextT | None = None,
        stream_mode: StreamMode | list[StreamMode] | None = None,
        interrupt_before: All | Sequence[str] | None = None,
        interrupt_after: All | Sequence[str] | None = None,
        subgraphs: bool = False,
        version: Literal["v1"] = ...,
    ) -> AsyncIterator[dict[str, Any] | Any]: ...

    @abstractmethod
    def astream(
        self,
        input: InputT | Command | None,
        config: RunnableConfig | None = None,
        *,
        context: ContextT | None = None,
        stream_mode: StreamMode | list[StreamMode] | None = None,
        interrupt_before: All | Sequence[str] | None = None,
        interrupt_after: All | Sequence[str] | None = None,
        subgraphs: bool = False,
        version: Literal["v1", "v2"] = "v1",
    ) -> AsyncIterator[dict[str, Any] | Any]: ...

    @overload
    @abstractmethod
    def invoke(
        self,
        input: InputT | Command | None,
        config: RunnableConfig | None = None,
        *,
        context: ContextT | None = None,
        interrupt_before: All | Sequence[str] | None = None,
        interrupt_after: All | Sequence[str] | None = None,
        version: Literal["v2"],
    ) -> GraphOutput[OutputT]: ...

    @overload
    @abstractmethod
    def invoke(
        self,
        input: InputT | Command | None,
        config: RunnableConfig | None = None,
        *,
        context: ContextT | None = None,
        interrupt_before: All | Sequence[str] | None = None,
        interrupt_after: All | Sequence[str] | None = None,
        version: Literal["v1"] = ...,
    ) -> dict[str, Any] | Any: ...

    @abstractmethod
    def invoke(
        self,
        input: InputT | Command | None,
        config: RunnableConfig | None = None,
        *,
        context: ContextT | None = None,
        interrupt_before: All | Sequence[str] | None = None,
        interrupt_after: All | Sequence[str] | None = None,
        version: Literal["v1", "v2"] = "v1",
    ) -> dict[str, Any] | Any: ...

    @overload
    @abstractmethod
    async def ainvoke(
        self,
        input: InputT | Command | None,
        config: RunnableConfig | None = None,
        *,
        context: ContextT | None = None,
        interrupt_before: All | Sequence[str] | None = None,
        interrupt_after: All | Sequence[str] | None = None,
        version: Literal["v2"],
    ) -> GraphOutput[OutputT]: ...

    @overload
    @abstractmethod
    async def ainvoke(
        self,
        input: InputT | Command | None,
        config: RunnableConfig | None = None,
        *,
        context: ContextT | None = None,
        interrupt_before: All | Sequence[str] | None = None,
        interrupt_after: All | Sequence[str] | None = None,
        version: Literal["v1"] = ...,
    ) -> dict[str, Any] | Any: ...

    @abstractmethod
    async def ainvoke(
        self,
        input: InputT | Command | None,
        config: RunnableConfig | None = None,
        *,
        context: ContextT | None = None,
        interrupt_before: All | Sequence[str] | None = None,
        interrupt_after: All | Sequence[str] | None = None,
        version: Literal["v1", "v2"] = "v1",
    ) -> dict[str, Any] | Any: ...


StreamChunk = tuple[tuple[str, ...], str, Any]


class StreamProtocol:
    __slots__ = ("modes", "__call__")

    modes: set[StreamMode]

    __call__: Callable[[Self, StreamChunk], None]

    def __init__(
        self,
        __call__: Callable[[StreamChunk], None],
        modes: set[StreamMode],
    ) -> None:
        self.__call__ = cast(Callable[[Self, StreamChunk], None], __call__)
        self.modes = modes
