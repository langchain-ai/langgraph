from __future__ import annotations

import sys
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Generic, Literal, TypeVar

if sys.version_info >= (3, 13):
    ContextT = TypeVar("ContextT", default=None)
else:
    ContextT = TypeVar("ContextT")

if sys.version_info >= (3, 12):
    from typing import TypeAliasType
else:
    from typing_extensions import TypeAliasType

from langgraph_sdk.auth.types import BaseUser

if TYPE_CHECKING:
    from langgraph.store.base import BaseStore

__all__ = [
    "AccessContext",
    "ServerRuntime",
]


AccessContext = Literal[
    "threads.create_run",
    "threads.update",
    "threads.read",
    "assistants.read",
]


@dataclass(kw_only=True, slots=True, frozen=True)
class _ServerRuntimeBase(Generic[ContextT]):
    """Base for server runtime variants.

    !!! warning "Beta"
        This API is in beta and may change in future releases.
    """

    access_context: AccessContext
    """Why the graph factory is being called.

    The server accesses graphs in several contexts beyond just executing runs.
    For example, it calls the graph factory to retrieve schemas, render the
    graph structure, or read state history. This field tells you which
    operation triggered the current call.

    In all contexts, the returned graph must have the same topology (nodes,
    edges, state schema) as the graph used for execution. Use
    `.execution_runtime` to conditionally set up expensive *resources*
    (MCP servers, DB connections) without changing the graph structure.

    Write contexts (graph is used to write state):

    - `threads.create_run` (`graph.astream`) — full graph execution
      (nodes + edges). `context` is available (use `.execution_runtime`
      to narrow).
    - `threads.update` (`graph.aupdate_state`) — does NOT execute node
      functions or evaluate edges. Only runs the node's channel writers
      to apply the provided values to state channels as if the specified
      node had returned them. Reducers are applied and channel triggers
      are set, so the next `invoke`/`stream` call will evaluate edges
      from that node to determine the next step. Does not need access to
      external resources, but a different graph topology will apply
      writes to the wrong channels.

    Read state contexts (graph used to format the returned
    `StateSnapshot`). A different topology may cause `get_state` to
    report incorrect pending tasks. Note that `useStream` uses the state
    history endpoint to render interrupts and support branching:

    - `threads.read` (`graph.aget_state`, `graph.aget_state_history`) —
      the graph structure informs which tasks to include in the prepared
      view of the latest checkpoint and how to process subgraphs.

    Introspection contexts (graph structure only, no execution).
    A different topology may cause schemas and visualizations to not
    match actual execution:

    - `assistants.read` (`graph.aget_graph`, `graph.aget_subgraphs`,
      `graph.aget_schemas`) — return the graph definition, subgraph
      definitions, and input/output/config schemas. Used for
      visualization in the studio UI and to populate schemas for MCP,
      A2A, and other protocol integrations.
    """

    user: BaseUser | None = field(default=None)
    """The authenticated user, or `None` if no custom auth is configured."""

    store: BaseStore
    """Store for the graph run, enabling persistence and memory."""

    @property
    def execution_runtime(self) -> _ExecutionRuntime[ContextT] | None:
        """Narrow to the execution runtime, or `None` if not in an execution context.

        When the server calls the graph factory for `threads.create_run`, the returned
        object provides access to `context` (typed by the graph's
        `context_schema`). For all other access contexts (introspection, state
        reads, state updates), this returns `None`.

        Use this to conditionally set up expensive resources (MCP tool servers,
        database connections, etc.) that are only needed during execution:

        ```python
        import contextlib
        from langgraph_sdk.runtime import ServerRuntime

        @contextlib.asynccontextmanager
        async def my_factory(runtime: ServerRuntime[MyCtx]):
            if ert := runtime.execution_runtime:
                # Only connect to MCP servers when actually executing a run.
                # Introspection calls (get_schema, get_graph, ...) skip this.
                mcp_tools = await connect_mcp(ert.context.mcp_endpoint)
                yield create_agent(model, tools=mcp_tools)
                await disconnect_mcp()
            else:
                yield create_agent(model, tools=[])
        ```
        """
        if isinstance(self, _ExecutionRuntime):
            return self
        return None

    def ensure_user(self) -> BaseUser:
        """Return the authenticated user, or raise if not available.

        When custom auth is configured, `user` is set for all access contexts
        (the factory is only called from HTTP handlers where the auth
        middleware has already run). This method raises only when no custom
        auth is configured.

        Raises:
            PermissionError: If no user is authenticated.
        """
        if self.user is None:
            raise PermissionError(
                f"No authenticated user available in access_context='{self.access_context}'. "
                "Ensure custom auth is configured for the server."
            )
        return self.user


@dataclass(kw_only=True, slots=True, frozen=True)
class _ExecutionRuntime(_ServerRuntimeBase[ContextT], Generic[ContextT]):
    """Runtime for `threads.create_run` — the graph will be fully executed.

    Access this via `.execution_runtime` on `ServerRuntime`. Do not
    construct directly.

    !!! warning "Beta"
        This API is in beta and may change in future releases.
    """

    context: ContextT = field(default=None)  # type: ignore[assignment]
    """The graph run context, typed by the graph's `context_schema`.

    Only available during `threads.create_run`.
    """


@dataclass(kw_only=True, slots=True, frozen=True)
class _ReadRuntime(_ServerRuntimeBase[ContextT], Generic[ContextT]):
    """Runtime for non-execution access contexts.

    Used for introspection (`assistants.read`), state operations
    (`threads.read`), and state updates (`threads.update`).
    No `context` is available.

    !!! warning "Beta"
        This API is in beta and may change in future releases.
    """


ServerRuntime = TypeAliasType(
    "ServerRuntime",
    _ExecutionRuntime[ContextT] | _ReadRuntime[ContextT],
    type_params=(ContextT,),
)
"""Runtime context passed to graph builder factories within the Agent Server.

Requires version 0.7.30 or later of the agent server.

The server calls your graph factory in multiple contexts: executing runs,
reading state, fetching schemas, and more. `ServerRuntime` provides
the authenticated user, store, and access context for every call. Use
`.execution_runtime` to narrow to the execution variant and access
`context`.

Example — conditionally initialize MCP tools only during execution:

```python
import contextlib
from dataclasses import dataclass

from langchain.agents import create_agent
from langgraph_sdk.runtime import ServerRuntime
from my_agent import connect_mcp, disconnect_mcp

@dataclass
class MyCtx:
    mcp_endpoint: str

_readonly_agent = create_agent("anthropic:claude-3-5-haiku", tools=[])

@contextlib.asynccontextmanager
async def my_factory(runtime: ServerRuntime[MyCtx]):
    if ert := runtime.execution_runtime:
        # Only connect to MCP servers for actual runs.
        # Schema / graph introspection calls skip this.
        user_id = runtime.ensure_user().identity
        mcp_tools = await connect_mcp(ert.context.mcp_endpoint, user_id)
        yield create_agent("anthropic:claude-3-5-haiku", tools=mcp_tools)
        await disconnect_mcp()
    else:
        yield _readonly_agent
```

Example — simple factory that ignores context:

```python
from langgraph_sdk.runtime import ServerRuntime

def build_graph(user: BaseUser) -> CompiledGraph:
    ...

async def my_factory(runtime: ServerRuntime) -> CompiledGraph:
    # No generic needed if you don't use context.
    return build_graph(runtime.ensure_user())
```

!!! warning "Beta"
    This API is in beta and may change in future releases.
"""
