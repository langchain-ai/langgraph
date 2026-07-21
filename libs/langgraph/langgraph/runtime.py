from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field, replace
from typing import Any, Generic, cast

from langgraph.store.base import BaseStore
from langgraph_sdk.auth.types import BaseUser
from typing_extensions import TypedDict, Unpack

from langgraph._internal._constants import CONF, CONFIG_KEY_RUNTIME
from langgraph.config import get_config
from langgraph.types import _DC_KWARGS, StreamWriter
from langgraph.typing import ContextT

__all__ = (
    "BaseUser",
    "ExecutionInfo",
    "RunControl",
    "Runtime",
    "ServerInfo",
    "get_runtime",
)


@dataclass(frozen=True, slots=True)
class ExecutionInfo:
    """Read-only execution info/metadata for the execution of current thread/run/node."""

    checkpoint_id: str
    """The checkpoint ID for the current execution."""

    checkpoint_ns: str
    """The checkpoint namespace for the current execution."""

    task_id: str
    """The task ID for the current execution."""

    thread_id: str | None = None
    """The thread ID for the current execution.

    None when running without a checkpointer (i.e., no persistence)."""

    run_id: str | None = None
    """The run ID for the current execution.

    None when `run_id` is not provided in the RunnableConfig."""

    node_attempt: int = 1
    """Current node execution attempt number (1-indexed)."""

    node_first_attempt_time: float | None = None
    """Unix timestamp (seconds) for when the first attempt started."""

    def patch(self, **overrides: Any) -> ExecutionInfo:
        """Return a new execution info object with selected fields replaced."""
        return replace(self, **overrides)


@dataclass(frozen=True, slots=True)
class ServerInfo:
    """Metadata injected by LangGraph Server. None when running open-source LangGraph without LangSmith deployments."""

    assistant_id: str
    """The assistant ID for the current execution."""

    graph_id: str
    """The graph ID for the current execution."""

    user: BaseUser | None = None
    """The authenticated user, if any.

    This implements the `BaseUser` protocol from `langgraph_sdk.auth.types`,
    which supports both attribute access (e.g. `user.identity`) and dict-like
    access (e.g. `user["identity"]`).
    """


class RunControl:
    """Run-scoped control surface for cooperative draining.

    Intended for a single graph run. Create a fresh `RunControl` per run;
    reusing a control after `request_drain()` leaves it drained.

    Safe to call from any thread: the drain request is represented by a
    single attribute write, so no lock is needed for this signal.
    If more mutable state is added here, add synchronization.
    """

    __slots__ = ("_drain_reason",)

    def __init__(self) -> None:
        self._drain_reason: str | None = None

    def request_drain(self, reason: str = "shutdown") -> None:
        self._drain_reason = reason

    @property
    def drain_requested(self) -> bool:
        return self._drain_reason is not None

    @property
    def drain_reason(self) -> str | None:
        return self._drain_reason


def _no_op_stream_writer(_: Any) -> None: ...


def _no_op_heartbeat() -> None: ...


class _RuntimeOverrides(TypedDict, Generic[ContextT], total=False):
    context: ContextT
    store: BaseStore | None
    stream_writer: StreamWriter
    heartbeat: Callable[[], None]
    previous: Any
    execution_info: ExecutionInfo
    server_info: ServerInfo | None
    control: RunControl | None


@dataclass(**_DC_KWARGS)
class Runtime(Generic[ContextT]):
    """Convenience class that bundles run-scoped context and other runtime utilities.

    This class is injected into graph nodes and middleware. It provides access to
    `context`, `store`, `stream_writer`, `previous`, and `execution_info`.

    !!! note "Accessing `config`"

        `Runtime` does not include `config`. To access `RunnableConfig`, you can inject
        it directly by adding a `config: RunnableConfig` parameter to your node function
        (recommended), or use `get_config()` from `langgraph.config`.

    !!! note
        `ToolRuntime` (from `langgraph.prebuilt`) is a subclass that provides similar
        functionality but is designed specifically for tools. It shares `context`, `store`,
        and `stream_writer` with `Runtime`, and adds tool-specific attributes like `config`,
        `state`, and `tool_call_id`.

    !!! version-added "Added in version v0.6.0"

    Example:

    ```python
    from typing import TypedDict
    from langgraph.graph import StateGraph
    from dataclasses import dataclass
    from langgraph.runtime import Runtime
    from langgraph.store.memory import InMemoryStore


    @dataclass
    class Context:  # (1)!
        user_id: str


    class State(TypedDict, total=False):
        response: str


    store = InMemoryStore()  # (2)!
    store.put(("users",), "user_123", {"name": "Alice"})


    def personalized_greeting(state: State, runtime: Runtime[Context]) -> State:
        '''Generate personalized greeting using runtime context and store.'''
        user_id = runtime.context.user_id  # (3)!
        name = "unknown_user"
        if runtime.store:
            if memory := runtime.store.get(("users",), user_id):
                name = memory.value["name"]

        response = f"Hello {name}! Nice to see you again."
        return {"response": response}


    graph = (
        StateGraph(state_schema=State, context_schema=Context)
        .add_node("personalized_greeting", personalized_greeting)
        .set_entry_point("personalized_greeting")
        .set_finish_point("personalized_greeting")
        .compile(store=store)
    )

    result = graph.invoke({}, context=Context(user_id="user_123"))
    print(result)
    # > {'response': 'Hello Alice! Nice to see you again.'}
    ```

    1. Define a schema for the runtime context.
    2. Create a store to persist memories and other information.
    3. Use the runtime context to access the `user_id`.
    """

    context: ContextT = field(default=None)  # type: ignore[assignment]
    """Static context for the graph run, like `user_id`, `db_conn`, etc.

    Can also be thought of as 'run dependencies'."""

    store: BaseStore | None = field(default=None)
    """Store for the graph run, enabling persistence and memory."""

    stream_writer: StreamWriter = field(default=_no_op_stream_writer)
    """Function that writes to the custom stream."""

    heartbeat: Callable[[], None] = field(default=_no_op_heartbeat)
    """Record progress for the current node's `idle_timeout`.

    Call this from inside long-running work that does not naturally emit
    writes, stream chunks, child tasks, or LangChain callback events, to
    prevent the node from being treated as idle. It is also the only
    progress signal honored under `TimeoutPolicy(refresh_on="heartbeat")`.
    Outside an idle-timed attempt this is a no-op.
    """

    previous: Any = field(default=None)
    """The previous return value for the given thread.

    Only available with the functional API when a checkpointer is provided.
    """

    execution_info: ExecutionInfo | None = field(default=None)
    """Read-only execution information/metadata for the current node run.

    None before task preparation populates it."""

    server_info: ServerInfo | None = field(default=None)
    """Metadata injected by LangGraph Server. None when running open-source LangGraph without LangSmith deployments."""

    control: RunControl | None = field(default=None)
    """Run-scoped control plane for cooperative draining.

    Populated automatically during graph runs. None outside an active
    graph runtime.
    """

    def merge(self, other: Runtime[ContextT]) -> Runtime[ContextT]:
        """Merge two runtimes together.

        If a value is not provided in the other runtime, the value from the current runtime is used.
        """
        return Runtime(
            context=other.context or self.context,
            store=other.store or self.store,
            stream_writer=other.stream_writer
            if other.stream_writer is not _no_op_stream_writer
            else self.stream_writer,
            heartbeat=other.heartbeat
            if other.heartbeat is not _no_op_heartbeat
            else self.heartbeat,
            previous=self.previous if other.previous is None else other.previous,
            execution_info=other.execution_info or self.execution_info,
            server_info=other.server_info or self.server_info,
            control=other.control or self.control,
        )

    def override(
        self, **overrides: Unpack[_RuntimeOverrides[ContextT]]
    ) -> Runtime[ContextT]:
        """Replace the runtime with a new runtime with the given overrides."""
        return replace(self, **overrides)

    def patch_execution_info(self, **overrides: Any) -> Runtime[ContextT]:
        """Return a new runtime with selected execution_info fields replaced."""
        if self.execution_info is None:
            msg = "Cannot patch execution_info before it has been set"
            raise RuntimeError(msg)
        return replace(
            self,
            execution_info=self.execution_info.patch(**overrides),
        )

    @property
    def drain_requested(self) -> bool:
        return self.control.drain_requested if self.control is not None else False

    @property
    def drain_reason(self) -> str | None:
        return self.control.drain_reason if self.control is not None else None


DEFAULT_RUNTIME = Runtime(
    context=None,
    store=None,
    stream_writer=_no_op_stream_writer,
    heartbeat=_no_op_heartbeat,
    previous=None,
    execution_info=None,
    control=None,
)


def get_runtime(context_schema: type[ContextT] | None = None) -> Runtime[ContextT]:
    """Get the runtime for the current graph run.

    Args:
        context_schema: Optional schema used for type hinting the return type of the runtime.

    Returns:
        The runtime for the current graph run.
    """

    # TODO: in an ideal world, we would have a context manager for
    # the runtime that's independent of the config. this will follow
    # from the removal of the configurable packing
    runtime = cast(Runtime[ContextT], get_config()[CONF].get(CONFIG_KEY_RUNTIME))
    return runtime
