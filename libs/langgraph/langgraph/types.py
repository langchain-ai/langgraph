from __future__ import annotations

import sys
from collections import deque
from collections.abc import Callable, Hashable, Sequence
from dataclasses import asdict, dataclass
from typing import (
    TYPE_CHECKING,
    Any,
    ClassVar,
    Generic,
    Literal,
    NamedTuple,
    TypeVar,
    final,
)
from warnings import warn

from langchain_core.runnables import Runnable, RunnableConfig
from langgraph.checkpoint.base import BaseCheckpointSaver, CheckpointMetadata
from typing_extensions import Unpack, deprecated
from xxhash import xxh3_128_hexdigest

from langgraph._internal._cache import default_cache_key
from langgraph._internal._fields import get_cached_annotated_keys, get_update_as_tuples
from langgraph._internal._retry import default_retry_on
from langgraph._internal._typing import MISSING, DeprecatedKwargs
from langgraph.warnings import LangGraphDeprecatedSinceV10

if TYPE_CHECKING:
    from langgraph.pregel.protocol import PregelProtocol


try:
    from langchain_core.messages.tool import ToolOutputMixin
except ImportError:

    class ToolOutputMixin:  # type: ignore[no-redef]
        pass


__all__ = (
    "All",
    "Checkpointer",
    "StreamMode",
    "StreamWriter",
    "RetryPolicy",
    "CachePolicy",
    "Interrupt",
    "StateUpdate",
    "PregelTask",
    "PregelExecutableTask",
    "StateSnapshot",
    "Send",
    "Command",
    "Durability",
    "interrupt",
)

Durability = Literal["sync", "async", "exit"]
"""Durability mode for the graph execution.
- `"sync"`: Changes are persisted synchronously before the next step starts.
- `"async"`: Changes are persisted asynchronously while the next step executes.
- `"exit"`: Changes are persisted only when the graph exits."""

All = Literal["*"]
"""Special value to indicate that graph should interrupt on all nodes."""

Checkpointer = None | bool | BaseCheckpointSaver
"""Type of the checkpointer to use for a subgraph.
- True enables persistent checkpointing for this subgraph.
- False disables checkpointing, even if the parent graph has a checkpointer.
- None inherits checkpointer from the parent graph."""

StreamMode = Literal[
    "values", "updates", "checkpoints", "tasks", "debug", "messages", "custom"
]
"""How the stream method should emit outputs.

- `"values"`: Emit all values in the state after each step, including interrupts.
    When used with functional API, values are emitted once at the end of the workflow.
- `"updates"`: Emit only the node or task names and updates returned by the nodes or tasks after each step.
    If multiple updates are made in the same step (e.g. multiple nodes are run) then those updates are emitted separately.
- `"custom"`: Emit custom data using from inside nodes or tasks using `StreamWriter`.
- `"messages"`: Emit LLM messages token-by-token together with metadata for any LLM invocations inside nodes or tasks.
- `"checkpoints"`: Emit an event when a checkpoint is created, in the same format as returned by `get_state()`.
- `"tasks"`: Emit events when tasks start and finish, including their results and errors.
- `"debug"`: Emit `"checkpoints"` and `"tasks"` events for debugging purposes.
"""

StreamWriter = Callable[[Any], None]
"""`Callable` that accepts a single argument and writes it to the output stream.
Always injected into nodes if requested as a keyword argument, but it's a no-op
when not using `stream_mode="custom"`."""

_DC_KWARGS = {"kw_only": True, "slots": True, "frozen": True}


class RetryPolicy(NamedTuple):
    """Configuration for retrying nodes.

    !!! version-added "Added in version 0.2.24"
    """

    initial_interval: float = 0.5
    """Amount of time that must elapse before the first retry occurs. In seconds."""
    backoff_factor: float = 2.0
    """Multiplier by which the interval increases after each retry."""
    max_interval: float = 128.0
    """Maximum amount of time that may elapse between retries. In seconds."""
    max_attempts: int = 3
    """Maximum number of attempts to make before giving up, including the first."""
    jitter: bool = True
    """Whether to add random jitter to the interval between retries."""
    retry_on: (
        type[Exception] | Sequence[type[Exception]] | Callable[[Exception], bool]
    ) = default_retry_on
    """List of exception classes that should trigger a retry, or a callable that returns `True` for exceptions that should trigger a retry."""


KeyFuncT = TypeVar("KeyFuncT", bound=Callable[..., str | bytes])


@dataclass(**_DC_KWARGS)
class CachePolicy(Generic[KeyFuncT]):
    """Configuration for caching nodes."""

    key_func: KeyFuncT = default_cache_key  # type: ignore[assignment]
    """Function to generate a cache key from the node's input.
    Defaults to hashing the input with pickle."""

    ttl: int | None = None
    """Time to live for the cache entry in seconds. If `None`, the entry never expires."""


_DEFAULT_INTERRUPT_ID = "placeholder-id"


@final
@dataclass(init=False, slots=True)
class Interrupt:
    """Information about an interrupt that occurred in a node.

    !!! version-added "Added in version 0.2.24"

    !!! version-changed "Changed in version v0.4.0"
        * `interrupt_id` was introduced as a property

    !!! version-changed "Changed in version v0.6.0"

        The following attributes have been removed:

        * `ns`
        * `when`
        * `resumable`
        * `interrupt_id`, deprecated in favor of `id`
    """

    value: Any
    """The value associated with the interrupt."""

    id: str
    """The ID of the interrupt. Can be used to resume the interrupt directly."""

    def __init__(
        self,
        value: Any,
        id: str = _DEFAULT_INTERRUPT_ID,
        **deprecated_kwargs: Unpack[DeprecatedKwargs],
    ) -> None:
        self.value = value

        if (
            (ns := deprecated_kwargs.get("ns", MISSING)) is not MISSING
            and (id == _DEFAULT_INTERRUPT_ID)
            and (isinstance(ns, Sequence))
        ):
            self.id = xxh3_128_hexdigest("|".join(ns).encode())
        else:
            self.id = id

    @classmethod
    def from_ns(cls, value: Any, ns: str) -> Interrupt:
        return cls(value=value, id=xxh3_128_hexdigest(ns.encode()))

    @property
    @deprecated("`interrupt_id` is deprecated. Use `id` instead.", category=None)
    def interrupt_id(self) -> str:
        warn(
            "`interrupt_id` is deprecated. Use `id` instead.",
            LangGraphDeprecatedSinceV10,
            stacklevel=2,
        )
        return self.id


class StateUpdate(NamedTuple):
    values: dict[str, Any] | None
    as_node: str | None = None
    task_id: str | None = None


class PregelTask(NamedTuple):
    """A Pregel task."""

    id: str
    name: str
    path: tuple[str | int | tuple, ...]
    error: Exception | None = None
    interrupts: tuple[Interrupt, ...] = ()
    state: None | RunnableConfig | StateSnapshot = None
    result: Any | None = None


if sys.version_info > (3, 11):
    _T_DC_KWARGS = {"weakref_slot": True, "slots": True, "frozen": True}
else:
    _T_DC_KWARGS = {"frozen": True}


class CacheKey(NamedTuple):
    """Cache key for a task."""

    ns: tuple[str, ...]
    """Namespace for the cache entry."""
    key: str
    """Key for the cache entry."""
    ttl: int | None
    """Time to live for the cache entry in seconds."""


@dataclass(**_T_DC_KWARGS)
class PregelExecutableTask:
    name: str
    input: Any
    proc: Runnable
    writes: deque[tuple[str, Any]]
    config: RunnableConfig
    triggers: Sequence[str]
    retry_policy: Sequence[RetryPolicy]
    cache_key: CacheKey | None
    id: str
    path: tuple[str | int | tuple, ...]
    writers: Sequence[Runnable] = ()
    subgraphs: Sequence[PregelProtocol] = ()


class StateSnapshot(NamedTuple):
    """Snapshot of the state of the graph at the beginning of a step."""

    values: dict[str, Any] | Any
    """Current values of channels."""
    next: tuple[str, ...]
    """The name of the node to execute in each task for this step."""
    config: RunnableConfig
    """Config used to fetch this snapshot."""
    metadata: CheckpointMetadata | None
    """Metadata associated with this snapshot."""
    created_at: str | None
    """Timestamp of snapshot creation."""
    parent_config: RunnableConfig | None
    """Config used to fetch the parent snapshot, if any."""
    tasks: tuple[PregelTask, ...]
    """Tasks to execute in this step. If already attempted, may contain an error."""
    interrupts: tuple[Interrupt, ...]
    """Interrupts that occurred in this step that are pending resolution."""


class Send:
    """A message or packet to send to a specific node in the graph.

    The `Send` class is used within a `StateGraph`'s conditional edges to
    dynamically invoke a node with a custom state at the next step.

    Importantly, the sent state can differ from the core graph's state,
    allowing for flexible and dynamic workflow management.

    One such example is a "map-reduce" workflow where your graph invokes
    the same node multiple times in parallel with different states,
    before aggregating the results back into the main graph's state.

    Attributes:
        node (str): The name of the target node to send the message to.
        arg (Any): The state or message to send to the target node.

    Examples:
        >>> from typing import Annotated
        >>> import operator
        >>> class OverallState(TypedDict):
        ...     subjects: list[str]
        ...     jokes: Annotated[list[str], operator.add]
        >>> from langgraph.types import Send
        >>> from langgraph.graph import END, START
        >>> def continue_to_jokes(state: OverallState):
        ...     return [Send("generate_joke", {"subject": s}) for s in state["subjects"]]
        >>> from langgraph.graph import StateGraph
        >>> builder = StateGraph(OverallState)
        >>> builder.add_node("generate_joke", lambda state: {"jokes": [f"Joke about {state['subject']}"]})
        >>> builder.add_conditional_edges(START, continue_to_jokes)
        >>> builder.add_edge("generate_joke", END)
        >>> graph = builder.compile()
        >>>
        >>> # Invoking with two subjects results in a generated joke for each
        >>> graph.invoke({"subjects": ["cats", "dogs"]})
        {'subjects': ['cats', 'dogs'], 'jokes': ['Joke about cats', 'Joke about dogs']}
    """

    __slots__ = ("node", "arg")

    node: str
    arg: Any

    def __init__(self, /, node: str, arg: Any) -> None:
        """
        Initialize a new instance of the `Send` class.

        Args:
            node: The name of the target node to send the message to.
            arg: The state or message to send to the target node.
        """
        self.node = node
        self.arg = arg

    def __hash__(self) -> int:
        return hash((self.node, self.arg))

    def __repr__(self) -> str:
        return f"Send(node={self.node!r}, arg={self.arg!r})"

    def __eq__(self, value: object) -> bool:
        return (
            isinstance(value, Send)
            and self.node == value.node
            and self.arg == value.arg
        )


N = TypeVar("N", bound=Hashable)


@dataclass(**_DC_KWARGS)
class Command(Generic[N], ToolOutputMixin):
    """One or more commands to update the graph's state and send messages to nodes.

    !!! version-added "Added in version 0.2.24"

    Args:
        graph: graph to send the command to. Supported values are:

            - `None`: the current graph
            - `Command.PARENT`: closest parent graph
        update: Update to apply to the graph's state.
        resume: Value to resume execution with. To be used together with [`interrupt()`][langgraph.types.interrupt].
            Can be one of the following:

            - Mapping of interrupt ids to resume values
            - A single value with which to resume the next interrupt
        goto: Can be one of the following:

            - Name of the node to navigate to next (any node that belongs to the specified `graph`)
            - Sequence of node names to navigate to next
            - `Send` object (to execute a node with the input provided)
            - Sequence of `Send` objects
    """

    graph: str | None = None
    update: Any | None = None
    resume: dict[str, Any] | Any | None = None
    goto: Send | Sequence[Send | N] | N = ()

    def __repr__(self) -> str:
        # get all non-None values
        contents = ", ".join(
            f"{key}={value!r}" for key, value in asdict(self).items() if value
        )
        return f"Command({contents})"

    def _update_as_tuples(self) -> Sequence[tuple[str, Any]]:
        if isinstance(self.update, dict):
            return list(self.update.items())
        elif isinstance(self.update, (list, tuple)) and all(
            isinstance(t, tuple) and len(t) == 2 and isinstance(t[0], str)
            for t in self.update
        ):
            return self.update
        elif keys := get_cached_annotated_keys(type(self.update)):
            return get_update_as_tuples(self.update, keys)
        elif self.update is not None:
            return [("__root__", self.update)]
        else:
            return []

    PARENT: ClassVar[Literal["__parent__"]] = "__parent__"


def interrupt(value: Any) -> Any:
    """Interrupt the graph with a resumable exception from within a node.

    The `interrupt` function enables human-in-the-loop workflows by pausing graph
    execution and surfacing a value to the client. This value can communicate context
    or request input required to resume execution.

    In a given node, the first invocation of this function raises a `GraphInterrupt`
    exception, halting execution. The provided `value` is included with the exception
    and sent to the client executing the graph.

    A client resuming the graph must use the [`Command`][langgraph.types.Command]
    primitive to specify a value for the interrupt and continue execution.
    The graph resumes from the start of the node, **re-executing** all logic.

    If a node contains multiple `interrupt` calls, LangGraph matches resume values
    to interrupts based on their order in the node. This list of resume values
    is scoped to the specific task executing the node and is not shared across tasks.

    To use an `interrupt`, you must enable a checkpointer, as the feature relies
    on persisting the graph state.

    Example:
        ```python
        import uuid
        from typing import Optional
        from typing_extensions import TypedDict

        from langgraph.checkpoint.memory import InMemorySaver
        from langgraph.constants import START
        from langgraph.graph import StateGraph
        from langgraph.types import interrupt, Command


        class State(TypedDict):
            \"\"\"The graph state.\"\"\"

            foo: str
            human_value: Optional[str]
            \"\"\"Human value will be updated using an interrupt.\"\"\"


        def node(state: State):
            answer = interrupt(
                # This value will be sent to the client
                # as part of the interrupt information.
                \"what is your age?\"
            )
            print(f\"> Received an input from the interrupt: {answer}\")
            return {\"human_value\": answer}


        builder = StateGraph(State)
        builder.add_node(\"node\", node)
        builder.add_edge(START, \"node\")

        # A checkpointer must be enabled for interrupts to work!
        checkpointer = InMemorySaver()
        graph = builder.compile(checkpointer=checkpointer)

        config = {
            \"configurable\": {
                \"thread_id\": uuid.uuid4(),
            }
        }

        for chunk in graph.stream({\"foo\": \"abc\"}, config):
            print(chunk)

        # > {'__interrupt__': (Interrupt(value='what is your age?', id='45fda8478b2ef754419799e10992af06'),)}

        command = Command(resume=\"some input from a human!!!\")

        for chunk in graph.stream(Command(resume=\"some input from a human!!!\"), config):
            print(chunk)

        # > Received an input from the interrupt: some input from a human!!!
        # > {'node': {'human_value': 'some input from a human!!!'}}
        ```

    Args:
        value: The value to surface to the client when the graph is interrupted.

    Returns:
        Any: On subsequent invocations within the same node (same task to be precise), returns the value provided during the first invocation

    Raises:
        GraphInterrupt: On the first invocation within the node, halts execution and surfaces the provided value to the client.
    """
    from langgraph._internal._constants import (
        CONFIG_KEY_CHECKPOINT_NS,
        CONFIG_KEY_SCRATCHPAD,
        CONFIG_KEY_SEND,
        RESUME,
    )
    from langgraph.config import get_config
    from langgraph.errors import GraphInterrupt

    conf = get_config()["configurable"]
    # track interrupt index
    scratchpad = conf[CONFIG_KEY_SCRATCHPAD]
    idx = scratchpad.interrupt_counter()
    # find previous resume values
    if scratchpad.resume:
        if idx < len(scratchpad.resume):
            conf[CONFIG_KEY_SEND]([(RESUME, scratchpad.resume)])
            return scratchpad.resume[idx]
    # find current resume value
    v = scratchpad.get_null_resume(True)
    if v is not None:
        assert len(scratchpad.resume) == idx, (scratchpad.resume, idx)
        scratchpad.resume.append(v)
        conf[CONFIG_KEY_SEND]([(RESUME, scratchpad.resume)])
        return v
    # no resume value found
    raise GraphInterrupt(
        (
            Interrupt.from_ns(
                value=value,
                ns=conf[CONFIG_KEY_CHECKPOINT_NS],
            ),
        )
    )


@dataclass(slots=True)
class Overwrite:
    """Bypass a reducer and write the wrapped value directly to a BinaryOperatorAggregate channel.

    Receiving multiple Overwrite values for the same channel in a single super-step will raise an InvalidUpdateError.

    Example:
        >>> from typing import Annotated
        >>> import operator
        >>> from langgraph.graph import StateGraph
        >>> from langgraph.types import Overwrite
        >>>
        >>> class State(TypedDict):
        ...     messages: Annotated[list, operator.add]
        >>>
        >>> def node_a(state: TypedDict):
        ...     # Normal update: uses the reducer (operator.add)
        ...     return {"messages": ["a"]}
        >>>
        >>> def node_b(state: State):
        ...     # Overwrite: bypasses the reducer and replaces the entire value
        ...     return {"messages": Overwrite(value=["b"])}
        >>>
        >>> builder = StateGraph(State)
        >>> builder.add_node("node_a", node_a)
        >>> builder.add_node("node_b", node_b)
        >>> builder.set_entry_point("node_a")
        >>> builder.add_edge("node_a", "node_b")
        >>> graph = builder.compile()
        >>>
        >>> # Without Overwrite in node_b, messages would be ["START", "a", "b"]
        >>> # With Overwrite, messages is just ["b"]
        >>> result = graph.invoke({"messages": ["START"]})
        >>> assert result == {"messages": ["b"]}
    """

    value: Any
    """The value to write directly to the channel, bypassing any reducer."""
