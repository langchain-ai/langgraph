from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal

from langgraph.runtime import Runtime
from langgraph.types import _DC_KWARGS

if TYPE_CHECKING:
    from langgraph_sdk.auth.types import BaseUser

__all__ = ["AccessContext", "ServerRuntime"]

AccessContext = Literal[
    "runs.create",
    "threads.update_state",
    "assistants.get_graph",
    "assistants.get_subgraphs",
    "assistants.get_schemas",
    "threads.get_state",
    "threads.get_state_history",
]


@dataclass(**_DC_KWARGS)
class ServerRuntime(Runtime):
    """Server-side runtime context passed to graph builder factories.

    Extends the base Runtime with server-specific information about
    the authenticated user and the reason the graph factory is being called.
    """

    user: BaseUser | None = field(default=None)
    """The authenticated user.

    Set when custom auth is configured (always — the factory is only
    called from HTTP handlers where the auth middleware has already run).
    None when no custom auth is configured.
    """

    access_context: AccessContext = field(default="runs.create")
    """Why the graph factory is being called.

    Fine-grained context that tells you both *what* operation triggered
    the graph build and whether the graph will actually be executed.

    Execution contexts:

    - `runs.create` — full graph execution (nodes + edges).
    - `threads.update_state` — does NOT execute node functions or evaluate
      edges. Only runs the node's channel writers (`ChannelWrite` runnables)
      to apply the provided values to state channels as if the specified node
      had returned them. Reducers are applied and channel triggers are set,
      so the next `invoke`/`stream` call will evaluate edges from that node
      to determine the next step.

    Introspection contexts (graph structure only, no execution):

    - `assistants.get_graph` — return the graph definition.
    - `assistants.get_subgraphs` — return subgraph definitions.
    - `assistants.get_schemas` — return input/output/config schemas.

    State contexts (graph used to structure the `StateSnapshot`):

    - `threads.get_state` — the graph structure informs which tasks to
      include in the prepared view of the latest checkpoint and how to
      process subgraphs.
    - `threads.get_state_history` — same as above, for historical states.
    """

    @property
    def is_for_execution(self) -> bool:
        """Whether this graph will be used to write state.

        True for `runs.create` (full graph execution: nodes + edges) and
        `threads.update_state` (runs only channel writers and reducers,
        not node functions or edges).
        """
        return self.access_context in ("runs.create", "threads.update_state")

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
