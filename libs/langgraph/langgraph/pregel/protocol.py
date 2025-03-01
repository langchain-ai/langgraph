from abc import ABC, abstractmethod
from typing import (
    Any,
    Iterator,
    Optional,
    Sequence,
    Union,
)

from typing_extensions import Self

from langgraph.pregel.types import All, StateSnapshot, StreamMode
from langgraph.utils.config import AnyConfig
from langgraph.utils.runnable import Runnable


class PregelProtocol(Runnable, ABC):
    @abstractmethod
    def with_config(
        self, config: Optional[AnyConfig] = None, **kwargs: Any
    ) -> Self: ...

    @abstractmethod
    def get_state(
        self, config: AnyConfig, *, subgraphs: bool = False
    ) -> StateSnapshot: ...

    @abstractmethod
    def get_state_history(
        self,
        config: AnyConfig,
        *,
        filter: Optional[dict[str, Any]] = None,
        before: Optional[AnyConfig] = None,
        limit: Optional[int] = None,
    ) -> Iterator[StateSnapshot]: ...

    @abstractmethod
    def update_state(
        self,
        config: AnyConfig,
        values: Optional[Union[dict[str, Any], Any]],
        as_node: Optional[str] = None,
    ) -> AnyConfig: ...

    @abstractmethod
    def stream(
        self,
        input: Union[dict[str, Any], Any],
        config: Optional[AnyConfig] = None,
        *,
        stream_mode: Optional[Union[StreamMode, list[StreamMode]]] = None,
        interrupt_before: Optional[Union[All, Sequence[str]]] = None,
        interrupt_after: Optional[Union[All, Sequence[str]]] = None,
        subgraphs: bool = False,
    ) -> Iterator[Union[dict[str, Any], Any]]: ...

    @abstractmethod
    def invoke(
        self,
        input: Union[dict[str, Any], Any],
        config: Optional[AnyConfig] = None,
        *,
        interrupt_before: Optional[Union[All, Sequence[str]]] = None,
        interrupt_after: Optional[Union[All, Sequence[str]]] = None,
    ) -> Union[dict[str, Any], Any]: ...
