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
    "runs.create",
    "threads.update_state",
    "assistants.get_graph",
    "assistants.get_subgraphs",
    "assistants.get_schemas",
    "threads.get_state",
    "threads.get_state_history",
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

    Write contexts (graph is used to write state):

    - `runs.create` (graph.astream) — full graph execution (nodes + edges).
      `context` is available (use `.execution_runtime` to narrow).
    - `threads.update_state` (graph.aupdate_state) — does NOT execute node functions or evaluate
      edges. Only runs the node's channel writers to apply the provided values
      to state channels as if the specified node had returned them. Reducers
      are applied and channel triggers are set, so the next `invoke`/`stream`
      call will evaluate edges from that node to determine the next step.
      Does not need access to external resources, but it does expect the same
      graph topology as `runs.create`.
    
    Read state contexts (graph used to format the returned `StateSnapshot`):
        This impacts read endpoints only. Note that `useStream` uses the state history 
        endpoint to render interrupts and support branching.
    
    - `threads.get_state` (graph.aget_state) — the graph structure informs which tasks to
      include in the prepared view of the latest checkpoint and how to
      process subgraphs.
    - `threads.get_state_history` (graph.aget_state_history) — same as above, for the list of previous checkpoints.
    
    Introspection contexts (graph structure only, no execution):
        This impacts read endpoints only.

    - `assistants.get_graph` (graph.aget_graph) — return the graph definition. Used for visualization in the
        studio UI.
    - `assistants.get_subgraphs` (graph.aget_subgraphs) — return subgraph definitions. Used for visualization in the
        studio UI.
    - `assistants.get_schemas` (graph.aget_schemas) — return input/output/config schemas. Used to populate the assistant
        schema for the studio UI as well as for the MCP, A2A, and other protocol integrations.
    """

    user: BaseUser | None = field(default=None)
    """The authenticated user, or `None` if no custom auth is configured."""

    store: BaseStore | None = field(default=None)
    """Store for the graph run, enabling persistence and memory."""

    permissions: list[str] = field(default_factory=list)
    """Permissions associated with the authenticated user.

    Empty list when no user is authenticated.
    """

    @property
    def execution_runtime(self) -> _ExecutionRuntime[ContextT] | None:
        """Narrow to the execution runtime, or `None` if not in an execution context.

        When the server calls the graph factory for `runs.create`, the returned
        object provides access to `context` (typed by the graph's
        `context_schema`). For all other access contexts (introspection, state
        reads, state updates), this returns `None`.

        Use this to conditionally set up expensive resources (MCP tool servers,
        database connections, etc.) that are only needed during execution:

        ```python
        from langgraph_sdk.runtime import ServerRuntime

        def my_factory(runtime: ServerRuntime[MyCtx]) -> CompiledGraph:
            tools = [search_tool, calculator_tool]

            if ert := runtime.execution_runtime:
                # Only connect to MCP servers when actually executing a run.
                # Introspection calls (get_schema, get_graph, ...) skip this.
                mcp_tools = connect_mcp(ert.context.mcp_endpoint)
                tools.extend(mcp_tools)

            return create_react_agent(model, tools=tools, store=runtime.store)
        ```
        """
        if isinstance(self, _ExecutionRuntime):
            return self
        return None

    def ensure_user(self) -> BaseUser:
        """Return the authenticated user, or raise if not available.

        Raises:
            PermissionError: If no user is authenticated.
        """
        if self.user is None:
            raise PermissionError(
                f"No authenticated user available in access_context='{self.access_context}'. "
                f"User is always available during 'runs.create' and 'threads.update_state'."
            )
        return self.user


@dataclass(kw_only=True, slots=True, frozen=True)
class _ExecutionRuntime(_ServerRuntimeBase[ContextT], Generic[ContextT]):
    """Runtime for `runs.create` — the graph will be fully executed.

    Access this via `.execution_runtime` on `ServerRuntime`. Do not
    construct directly.

    !!! warning "Beta"
        This API is in beta and may change in future releases.
    """

    context: ContextT = field(default=None)  # type: ignore[assignment]
    """The graph run context, typed by the graph's `context_schema`.

    Only available during `runs.create`.
    """


@dataclass(kw_only=True, slots=True, frozen=True)
class _ReadRuntime(_ServerRuntimeBase[ContextT], Generic[ContextT]):
    """Runtime for non-execution access contexts.

    Used for introspection (`assistants.get_graph`, `assistants.get_schemas`, etc.),
    state operations (`threads.get_state`, `threads.get_state_history`), and
    state updates (`threads.update_state`). No `context` is available.

    !!! warning "Beta"
        This API is in beta and may change in future releases.
    """


ServerRuntime = TypeAliasType(
    "ServerRuntime",
    _ExecutionRuntime[ContextT] | _ReadRuntime[ContextT],
    type_params=(ContextT,),
)
"""Runtime context passed to graph builder factories within the Agent Server.

Requires version 0.7.29 or later of the agent server.

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
async def my_factory(runtime: ServerRuntime[MyCtx]) -> CompiledGraph:
    tools = [search_tool, calculator_tool]

    if ert := runtime.execution_runtime:
        # Only connect to MCP servers for actual runs.
        # Schema / graph introspection calls skip this.
        user_id = ert.ensure_user().id
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
    # ... build graph ...
    return graph

def my_factory(runtime: ServerRuntime) -> CompiledGraph:
    # No generic needed if you don't use context.
    return build_graph(runtime.ensure_user())
```

!!! warning "Beta"
    This API is in beta and may change in future releases.
"""
