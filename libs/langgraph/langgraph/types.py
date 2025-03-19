import dataclasses
import sys
from collections import deque
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    ClassVar,
    Generic,
    Hashable,
    Literal,
    NamedTuple,
    Optional,
    Sequence,
    Type,
    TypeVar,
    Union,
    cast,
    get_type_hints,
)

from langchain_core.runnables import Runnable, RunnableConfig
from typing_extensions import Self

from langgraph.checkpoint.base import BaseCheckpointSaver, CheckpointMetadata

if TYPE_CHECKING:
    from langgraph.pregel.protocol import PregelProtocol
    from langgraph.store.base import BaseStore


try:
    from langchain_core.messages.tool import ToolOutputMixin
except ImportError:

    class ToolOutputMixin:  # type: ignore[no-redef]
        pass


All = Literal["*"]
"""Special value to indicate that graph should interrupt on all nodes."""

Checkpointer = Union[None, bool, BaseCheckpointSaver]
"""Type of the checkpointer to use for a subgraph.
- True enables persistent checkpointing for this subgraph.
- False disables checkpointing, even if the parent graph has a checkpointer.
- None inherits checkpointer from the parent graph."""

StreamMode = Literal["values", "updates", "debug", "messages", "custom"]
"""How the stream method should emit outputs.

- `"values"`: Emit all values in the state after each step.
    When used with functional API, values are emitted once at the end of the workflow.
- `"updates"`: Emit only the node or task names and updates returned by the nodes or tasks after each step.
    If multiple updates are made in the same step (e.g. multiple nodes are run) then those updates are emitted separately.
- `"custom"`: Emit custom data using from inside nodes or tasks using `StreamWriter`.
- `"messages"`: Emit LLM messages token-by-token together with metadata for any LLM invocations inside nodes or tasks.
- `"debug"`: Emit debug events with as much information as possible for each step.
"""

StreamWriter = Callable[[Any], None]
"""Callable that accepts a single argument and writes it to the output stream.
Always injected into nodes if requested as a keyword argument, but it's a no-op
when not using stream_mode="custom"."""

if sys.version_info >= (3, 10):
    _DC_KWARGS = {"kw_only": True, "slots": True, "frozen": True}
else:
    _DC_KWARGS = {"frozen": True}


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


@dataclasses.dataclass(**_DC_KWARGS)
class Interrupt:
    value: Any
    resumable: bool = False
    ns: Optional[Sequence[str]] = None
    when: Literal["during"] = dataclasses.field(default="during", repr=False)


class StateUpdate(NamedTuple):
    values: Optional[dict[str, Any]]
    as_node: Optional[str] = None


class PregelTask(NamedTuple):
    id: str
    name: str
    path: tuple[Union[str, int, tuple], ...]
    error: Optional[Exception] = None
    interrupts: tuple[Interrupt, ...] = ()
    state: Union[None, RunnableConfig, "StateSnapshot"] = None
    result: Optional[Any] = None


if sys.version_info > (3, 11):
    _T_DC_KWARGS = {"weakref_slot": True, "slots": True, "frozen": True}
else:
    _T_DC_KWARGS = {"frozen": True}


@dataclasses.dataclass(**_T_DC_KWARGS)
class PregelExecutableTask:
    name: str
    input: Any
    proc: Runnable
    writes: deque[tuple[str, Any]]
    config: RunnableConfig
    triggers: Sequence[str]
    retry_policy: Optional[RetryPolicy]
    cache_policy: Optional[CachePolicy]
    id: str
    path: tuple[Union[str, int, tuple], ...]
    scheduled: bool = False
    writers: Sequence[Runnable] = ()
    subgraphs: Sequence["PregelProtocol"] = ()


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


N = TypeVar("N", bound=Hashable)


@dataclasses.dataclass(**_DC_KWARGS)
class Command(Generic[N], ToolOutputMixin):
    """One or more commands to update the graph's state and send messages to nodes.

    Args:
        graph: graph to send the command to. Supported values are:

            - None: the current graph (default)
            - Command.PARENT: closest parent graph
        update: update to apply to the graph's state.
        resume: value to resume execution with. To be used together with [`interrupt()`][langgraph.types.interrupt].
        goto: can be one of the following:

            - name of the node to navigate to next (any node that belongs to the specified `graph`)
            - sequence of node names to navigate to next
            - `Send` object (to execute a node with the input provided)
            - sequence of `Send` objects
    """

    graph: Optional[str] = None
    update: Optional[Any] = None
    resume: Optional[Union[Any, dict[str, Any]]] = None
    goto: Union[Send, Sequence[Union[Send, str]], str] = ()

    def __repr__(self) -> str:
        # get all non-None values
        contents = ", ".join(
            f"{key}={value!r}"
            for key, value in dataclasses.asdict(self).items()
            if value
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
        elif hints := get_type_hints(type(self.update)):
            return [(k, getattr(self.update, k)) for k in hints]
        elif self.update is not None:
            return [("__root__", self.update)]
        else:
            return []

    PARENT: ClassVar[Literal["__parent__"]] = "__parent__"


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


@dataclasses.dataclass(**{**_DC_KWARGS, "frozen": False})
class PregelScratchpad:
    # call
    call_counter: Callable[[], int]
    # interrupt
    interrupt_counter: Callable[[], int]
    get_null_resume: Callable[[bool], Any]
    resume: list[Any]
    # subgraph
    subgraph_counter: Callable[[], int]


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

        from langgraph.checkpoint.memory import MemorySaver
        from langgraph.constants import START
        from langgraph.graph import StateGraph
        from langgraph.types import interrupt


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
        checkpointer = MemorySaver()
        graph = builder.compile(checkpointer=checkpointer)

        config = {
            \"configurable\": {
                \"thread_id\": uuid.uuid4(),
            }
        }

        for chunk in graph.stream({\"foo\": \"abc\"}, config):
            print(chunk)
        ```

        ```pycon
        {'__interrupt__': (Interrupt(value='what is your age?', resumable=True, ns=['node:62e598fa-8653-9d6d-2046-a70203020e37'], when='during'),)}
        ```

        ```python
        command = Command(resume=\"some input from a human!!!\")

        for chunk in graph.stream(Command(resume=\"some input from a human!!!\"), config):
            print(chunk)
        ```

        ```pycon
        Received an input from the interrupt: some input from a human!!!
        {'node': {'human_value': 'some input from a human!!!'}}
        ```

    Args:
        value: The value to surface to the client when the graph is interrupted.

    Returns:
        Any: On subsequent invocations within the same node (same task to be precise), returns the value provided during the first invocation

    Raises:
        GraphInterrupt: On the first invocation within the node, halts execution and surfaces the provided value to the client.
    """
    from langgraph.constants import (
        CONFIG_KEY_CHECKPOINT_NS,
        CONFIG_KEY_SCRATCHPAD,
        CONFIG_KEY_SEND,
        NS_SEP,
        RESUME,
    )
    from langgraph.errors import GraphInterrupt
    from langgraph.utils.config import get_config

    conf = get_config()["configurable"]
    # track interrupt index
    scratchpad: PregelScratchpad = conf[CONFIG_KEY_SCRATCHPAD]
    idx = scratchpad.interrupt_counter()
    # find previous resume values
    if scratchpad.resume:
        if idx < len(scratchpad.resume):
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
            Interrupt(
                value=value,
                resumable=True,
                ns=cast(str, conf[CONFIG_KEY_CHECKPOINT_NS]).split(NS_SEP),
            ),
        )
    )
