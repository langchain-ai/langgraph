from __future__ import annotations

from collections.abc import Sequence
from typing import Any, NamedTuple, Protocol, Union

from langchain_core.runnables import Runnable, RunnableConfig
from typing_extensions import TypeAlias

from langgraph.constants import EMPTY_SEQ
from langgraph.runtime import Runtime
from langgraph.store.base import BaseStore
from langgraph.types import CachePolicy, RetryPolicy, StreamWriter
from langgraph.typing import ContextT, StateT_contra


class _Node(Protocol[StateT_contra]):
    def __call__(self, state: StateT_contra) -> Any: ...


class _NodeWithConfig(Protocol[StateT_contra]):
    def __call__(self, state: StateT_contra, config: RunnableConfig) -> Any: ...


class _NodeWithWriter(Protocol[StateT_contra]):
    def __call__(self, state: StateT_contra, *, writer: StreamWriter) -> Any: ...


class _NodeWithStore(Protocol[StateT_contra]):
    def __call__(self, state: StateT_contra, *, store: BaseStore) -> Any: ...


class _NodeWithWriterStore(Protocol[StateT_contra]):
    def __call__(
        self, state: StateT_contra, *, writer: StreamWriter, store: BaseStore
    ) -> Any: ...


class _NodeWithConfigWriter(Protocol[StateT_contra]):
    def __call__(
        self, state: StateT_contra, *, config: RunnableConfig, writer: StreamWriter
    ) -> Any: ...


class _NodeWithConfigStore(Protocol[StateT_contra]):
    def __call__(
        self, state: StateT_contra, *, config: RunnableConfig, store: BaseStore
    ) -> Any: ...


class _NodeWithConfigWriterStore(Protocol[StateT_contra]):
    def __call__(
        self,
        state: StateT_contra,
        *,
        config: RunnableConfig,
        writer: StreamWriter,
        store: BaseStore,
    ) -> Any: ...


class _NodeWithRuntime(Protocol[StateT_contra, ContextT]):
    def __call__(self, state: StateT_contra, *, runtime: Runtime[ContextT]) -> Any: ...


# TODO: we probably don't want to explicitly support the config / store signatures once
# we move to adding a context arg. Maybe what we do is we add support for kwargs with param spec
# this is purely for typing purposes though, so can easily change in the coming weeks.
StateNode: TypeAlias = Union[
    _Node[StateT_contra],
    _NodeWithConfig[StateT_contra],
    _NodeWithWriter[StateT_contra],
    _NodeWithStore[StateT_contra],
    _NodeWithWriterStore[StateT_contra],
    _NodeWithConfigWriter[StateT_contra],
    _NodeWithConfigStore[StateT_contra],
    _NodeWithConfigWriterStore[StateT_contra],
    _NodeWithRuntime[StateT_contra, ContextT],
    Runnable[StateT_contra, Any],
]


# TODO: use a dataclass generic on NodeInputType
class StateNodeSpec(NamedTuple):
    runnable: StateNode
    metadata: dict[str, Any] | None
    input_schema: type[Any]
    retry_policy: RetryPolicy | Sequence[RetryPolicy] | None
    cache_policy: CachePolicy | None
    ends: tuple[str, ...] | dict[str, str] | None = EMPTY_SEQ
    defer: bool = False
