from __future__ import annotations

import sys
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Generic, Protocol, Union

from langchain_core.runnables import Runnable, RunnableConfig
from typing_extensions import TypeAlias

from langgraph._internal._typing import EMPTY_SEQ
from langgraph.runtime import Runtime
from langgraph.store.base import BaseStore
from langgraph.types import CachePolicy, RetryPolicy, StreamWriter
from langgraph.typing import ContextT, NodeInputT, NodeInputT_contra

_DC_SLOTS = {"slots": True} if sys.version_info >= (3, 10) else {}


class _Node(Protocol[NodeInputT_contra]):
    def __call__(self, state: NodeInputT_contra) -> Any: ...


class _NodeWithConfig(Protocol[NodeInputT_contra]):
    def __call__(self, state: NodeInputT_contra, config: RunnableConfig) -> Any: ...


class _NodeWithWriter(Protocol[NodeInputT_contra]):
    def __call__(self, state: NodeInputT_contra, *, writer: StreamWriter) -> Any: ...


class _NodeWithStore(Protocol[NodeInputT_contra]):
    def __call__(self, state: NodeInputT_contra, *, store: BaseStore) -> Any: ...


class _NodeWithWriterStore(Protocol[NodeInputT_contra]):
    def __call__(
        self, state: NodeInputT_contra, *, writer: StreamWriter, store: BaseStore
    ) -> Any: ...


class _NodeWithConfigWriter(Protocol[NodeInputT_contra]):
    def __call__(
        self, state: NodeInputT_contra, *, config: RunnableConfig, writer: StreamWriter
    ) -> Any: ...


class _NodeWithConfigStore(Protocol[NodeInputT_contra]):
    def __call__(
        self, state: NodeInputT_contra, *, config: RunnableConfig, store: BaseStore
    ) -> Any: ...


class _NodeWithConfigWriterStore(Protocol[NodeInputT_contra]):
    def __call__(
        self,
        state: NodeInputT_contra,
        *,
        config: RunnableConfig,
        writer: StreamWriter,
        store: BaseStore,
    ) -> Any: ...


class _NodeWithRuntime(Protocol[NodeInputT_contra, ContextT]):
    def __call__(
        self, state: NodeInputT_contra, *, runtime: Runtime[ContextT]
    ) -> Any: ...


# TODO: we probably don't want to explicitly support the config / store signatures once
# we move to adding a context arg. Maybe what we do is we add support for kwargs with param spec
# this is purely for typing purposes though, so can easily change in the coming weeks.
StateNode: TypeAlias = Union[
    _Node[NodeInputT],
    _NodeWithConfig[NodeInputT],
    _NodeWithWriter[NodeInputT],
    _NodeWithStore[NodeInputT],
    _NodeWithWriterStore[NodeInputT],
    _NodeWithConfigWriter[NodeInputT],
    _NodeWithConfigStore[NodeInputT],
    _NodeWithConfigWriterStore[NodeInputT],
    _NodeWithRuntime[NodeInputT, ContextT],
    Runnable[NodeInputT, Any],
]


@dataclass(**_DC_SLOTS)
class StateNodeSpec(Generic[NodeInputT, ContextT]):
    runnable: StateNode[NodeInputT, ContextT]
    metadata: dict[str, Any] | None
    input_schema: type[NodeInputT]
    retry_policy: RetryPolicy | Sequence[RetryPolicy] | None
    cache_policy: CachePolicy | None
    ends: tuple[str, ...] | dict[str, str] | None = EMPTY_SEQ
    defer: bool = False
