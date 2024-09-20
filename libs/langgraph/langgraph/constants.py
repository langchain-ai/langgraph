from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Literal, Mapping

INPUT = "__input__"
CONFIG_KEY_SEND = "__pregel_send"
CONFIG_KEY_READ = "__pregel_read"
CONFIG_KEY_CHECKPOINTER = "__pregel_checkpointer"
CONFIG_KEY_STREAM = "__pregel_stream"
CONFIG_KEY_STREAM_WRITER = "__pregel_stream_writer"
CONFIG_KEY_STORE = "__pregel_store"
CONFIG_KEY_RESUMING = "__pregel_resuming"
CONFIG_KEY_TASK_ID = "__pregel_task_id"
CONFIG_KEY_DEDUPE_TASKS = "__pregel_dedupe_tasks"
CONFIG_KEY_ENSURE_LATEST = "__pregel_ensure_latest"
CONFIG_KEY_DELEGATE = "__pregel_delegate"
# this one part of public API so more readable
CONFIG_KEY_CHECKPOINT_MAP = "checkpoint_map"
INTERRUPT = "__interrupt__"
ERROR = "__error__"
NO_WRITES = "__no_writes__"
SCHEDULED = "__scheduled__"
TASKS = "__pregel_tasks"  # for backwards compat, this is the original name of PUSH
PUSH = "__pregel_push"
PULL = "__pregel_pull"
RUNTIME_PLACEHOLDER = "__pregel_runtime_placeholder__"
RESERVED = {
    SCHEDULED,
    INTERRUPT,
    ERROR,
    NO_WRITES,
    TASKS,
    PUSH,
    PULL,
    CONFIG_KEY_SEND,
    CONFIG_KEY_READ,
    CONFIG_KEY_CHECKPOINTER,
    CONFIG_KEY_CHECKPOINT_MAP,
    CONFIG_KEY_STREAM,
    CONFIG_KEY_STREAM_WRITER,
    CONFIG_KEY_STORE,
    CONFIG_KEY_RESUMING,
    CONFIG_KEY_TASK_ID,
    CONFIG_KEY_DEDUPE_TASKS,
    CONFIG_KEY_ENSURE_LATEST,
    CONFIG_KEY_DELEGATE,
    INPUT,
    RUNTIME_PLACEHOLDER,
}
TAG_HIDDEN = "langsmith:hidden"

START = "__start__"
END = "__end__"

NS_SEP = "|"
NS_END = ":"

EMPTY_MAP: Mapping[str, Any] = MappingProxyType({})


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
        >>> from langgraph.constants import Send
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


@dataclass
class Interrupt:
    value: Any
    when: Literal["during"] = "during"
