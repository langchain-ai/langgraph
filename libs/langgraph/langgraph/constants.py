from typing import Any, Optional
from uuid import uuid4

INPUT = "__input__"
CONFIG_KEY_SEND = "__pregel_send"
CONFIG_KEY_READ = "__pregel_read"
CONFIG_KEY_CHECKPOINTER = "__pregel_checkpointer"
CONFIG_KEY_RESUMING = "__pregel_resuming"
INTERRUPT = "__interrupt__"
TASKS = "__pregel_tasks"
RESERVED = {
    INTERRUPT,
    TASKS,
    CONFIG_KEY_SEND,
    CONFIG_KEY_READ,
    CONFIG_KEY_CHECKPOINTER,
    CONFIG_KEY_RESUMING,
    INPUT,
}
TAG_HIDDEN = "langsmith:hidden"

START = "__start__"
END = "__end__"

CHECKPOINT_NAMESPACE_SEPARATOR = "|"
SEND_CHECKPOINT_NAMESPACE_SEPARATOR = ":"


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
        id (str): ID associated with the Send.

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
    id: Optional[str]

    def __init__(self, /, node: str, arg: Any, id: Optional[str] = None) -> None:
        """
        Initialize a new instance of the Send class.

        Args:
            node (str): The name of the target node to send the message to.
            arg (Any): The state or message to send to the target node.
            id (str): ID associated with the Send.
        """
        self.node = node
        self.arg = arg
        self.id = id or str(uuid4())

    def __hash__(self) -> int:
        return hash((self.node, self.arg))

    def __repr__(self) -> str:
        return f"Send(node={self.node!r}, arg={self.arg!r}, id={self.id!r})"

    def __eq__(self, value: object) -> bool:
        return (
            isinstance(value, Send)
            and self.node == value.node
            and self.arg == value.arg
        )
