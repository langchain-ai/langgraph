from _typeshed import Incomplete
from langchain_core.runnables import Runnable, RunnableConfig as RunnableConfig
from langgraph.channels.base import BaseChannel
from langgraph.graph.graph import Branch, CompiledGraph, Graph
from langgraph.managed.base import ManagedValueSpec as ManagedValueSpec
from langgraph.store.base import BaseStore as BaseStore
from langgraph.types import All as All, Checkpointer as Checkpointer, RetryPolicy as RetryPolicy
from langgraph.utils.runnable import RunnableLike as RunnableLike
from pydantic import BaseModel
from typing import Any, NamedTuple, Sequence, overload
from typing_extensions import Self

logger: Incomplete

class StateNodeSpec(NamedTuple):
    runnable: Runnable
    metadata: dict[str, Any] | None
    input: type[Any]
    retry_policy: RetryPolicy | None
    ends: tuple[str, ...] | dict[str, str] | None = ...

class StateGraph(Graph):
    nodes: dict[str, StateNodeSpec]
    channels: dict[str, BaseChannel]
    managed: dict[str, ManagedValueSpec]
    schemas: dict[type[Any], dict[str, BaseChannel | ManagedValueSpec]]
    schema: Incomplete
    input: Incomplete
    output: Incomplete
    config_schema: Incomplete
    waiting_edges: Incomplete
    def __init__(self, state_schema: type[Any] | None = None, config_schema: type[Any] | None = None, *, input: type[Any] | None = None, output: type[Any] | None = None) -> None: ...
    @overload
    def add_node(self, node: RunnableLike, *, metadata: dict[str, Any] | None = None, input: type[Any] | None = None, retry: RetryPolicy | None = None, destinations: dict[str, str] | tuple[str] | None = None) -> Self: ...
    @overload
    def add_node(self, node: str, action: RunnableLike, *, metadata: dict[str, Any] | None = None, input: type[Any] | None = None, retry: RetryPolicy | None = None, destinations: dict[str, str] | tuple[str] | None = None) -> Self: ...
    def add_edge(self, start_key: str | list[str], end_key: str) -> Self: ...
    def add_sequence(self, nodes: Sequence[RunnableLike | tuple[str, RunnableLike]]) -> Self: ...
    def compile(self, checkpointer: Checkpointer = None, *, store: BaseStore | None = None, interrupt_before: All | list[str] | None = None, interrupt_after: All | list[str] | None = None, debug: bool = False, name: str | None = None) -> CompiledStateGraph: ...

class CompiledStateGraph(CompiledGraph):
    builder: StateGraph
    def get_input_schema(self, config: RunnableConfig | None = None) -> type[BaseModel]: ...
    def get_output_schema(self, config: RunnableConfig | None = None) -> type[BaseModel]: ...
    def attach_node(self, key: str, node: StateNodeSpec | None) -> None: ...
    def attach_edge(self, starts: str | Sequence[str], end: str) -> None: ...
    def attach_branch(self, start: str, name: str, branch: Branch, *, with_reader: bool = True) -> None: ...

CONTROL_BRANCH_PATH: Incomplete
CONTROL_BRANCH: Incomplete
