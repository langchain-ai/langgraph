from collections import deque
from dataclasses import dataclass
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Generic,
    Literal,
    NamedTuple,
    Optional,
    Sequence,
    Type,
    TypeVar,
    Union,
    cast,
)

from langchain_core.runnables import Runnable, RunnableConfig
from typing_extensions import Self

from langgraph.checkpoint.base import BaseCheckpointSaver, CheckpointMetadata

if TYPE_CHECKING:
    from langgraph.store.base import BaseStore

All = Literal["*"]
"""Special value to indicate that graph should interrupt on all nodes."""

Checkpointer = Union[None, Literal[False], BaseCheckpointSaver]
"""Type of the checkpointer to use for a subgraph. False disables checkpointing,
even if the parent graph has a checkpointer. None inherits checkpointer."""

StreamMode = Literal["values", "updates", "debug", "messages", "custom"]
"""How the stream method should emit outputs.

- 'values': Emit all values of the state for each step.
- 'updates': Emit only the node name(s) and updates
    that were returned by the node(s) **after** each step.
- 'debug': Emit debug events for each step.
- 'messages': Emit LLM messages token-by-token.
- 'custom': Emit custom output `write: StreamWriter` kwarg of each node.
"""

StreamWriter = Callable[[Any], None]
"""Callable that accepts a single argument and writes it to the output stream.
Always injected into nodes if requested as a keyword argument, but it's a no-op
when not using stream_mode="custom"."""


def default_retry_on(exc: Exception) -> bool:
    import httpx
    import requests

    if isinstance(exc, ConnectionError):
        return True
    if isinstance(
        exc,
        (
            ValueError,
            TypeError,
            ArithmeticError,
            ImportError,
            LookupError,
            NameError,
            SyntaxError,
            RuntimeError,
            ReferenceError,
            StopIteration,
            StopAsyncIteration,
            OSError,
        ),
    ):
        return False
    if isinstance(exc, httpx.HTTPStatusError):
        return 500 <= exc.response.status_code < 600
    if isinstance(exc, requests.HTTPError):
        return 500 <= exc.response.status_code < 600 if exc.response else True
    return True


class RetryPolicy(NamedTuple):
    """Configuration for retrying nodes."""

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
    retry_on: Union[
        Type[Exception], Sequence[Type[Exception]], Callable[[Exception], bool]
    ] = default_retry_on
    """List of exception classes that should trigger a retry, or a callable that returns True for exceptions that should trigger a retry."""


class CachePolicy(NamedTuple):
    """Configuration for caching nodes."""

    pass


@dataclass
class Interrupt:
    value: Any
    when: Literal["during"] = "during"


class PregelTask(NamedTuple):
    id: str
    name: str
    path: tuple[Union[str, int], ...]
    error: Optional[Exception] = None
    interrupts: tuple[Interrupt, ...] = ()
    state: Union[None, RunnableConfig, "StateSnapshot"] = None
    result: Optional[dict[str, Any]] = None


class PregelExecutableTask(NamedTuple):
    name: str
    input: Any
    proc: Runnable
    writes: deque[tuple[str, Any]]
    config: RunnableConfig
    triggers: list[str]
    retry_policy: Optional[RetryPolicy]
    cache_policy: Optional[CachePolicy]
    id: str
    path: tuple[Union[str, int], ...]
    scheduled: bool = False


class StateSnapshot(NamedTuple):
    """Snapshot of the state of the graph at the beginning of a step."""

    values: Union[dict[str, Any], Any]
    """Current values of channels"""
    next: tuple[str, ...]
    """The name of the node to execute in each task for this step."""
    config: RunnableConfig
    """Config used to fetch this snapshot"""
    metadata: Optional[CheckpointMetadata]
    """Metadata associated with this snapshot"""
    created_at: Optional[str]
    """Timestamp of snapshot creation"""
    parent_config: Optional[RunnableConfig]
    """Config used to fetch the parent snapshot, if any"""
    tasks: tuple[PregelTask, ...]
    """Tasks to execute in this step. If already attempted, may contain an error."""


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
        ...
        >>> from langgraph.types import Send
        >>> from langgraph.graph import END, START
        >>> def continue_to_jokes(state: OverallState):
        ...     return [Send("generate_joke", {"subject": s}) for s in state['subjects']]
        ...
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
        Initialize a new instance of the Send class.

        Args:
            node (str): The name of the target node to send the message to.
            arg (Any): The state or message to send to the target node.
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


N = TypeVar("N")


class Control(Generic[N]):
    def __init__(
        self,
        *,
        update_state: Optional[dict[str, Any]] = None,
        trigger: Union[str, Sequence[str]] = (),
        send: Union[Send, Sequence[Send]] = (),
    ) -> None:
        self.update_state = update_state
        self.trigger = trigger
        self.send = send


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


class LoopProtocol:
    config: RunnableConfig
    store: Optional["BaseStore"]
    stream: Optional[StreamProtocol]
    step: int
    stop: int

    def __init__(
        self,
        *,
        step: int,
        stop: int,
        config: RunnableConfig,
        store: Optional["BaseStore"] = None,
        stream: Optional[StreamProtocol] = None,
    ) -> None:
        self.stream = stream
        self.config = config
        self.store = store
        self.step = step
        self.stop = stop
