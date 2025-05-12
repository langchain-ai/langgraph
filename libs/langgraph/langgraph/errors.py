from collections.abc import Sequence
from enum import Enum
from typing import Any

from langgraph.checkpoint.base import EmptyChannelError  # noqa: F401
from langgraph.types import Command, Interrupt

# EmptyChannelError re-exported for backwards compatibility


class ErrorCode(Enum):
    GRAPH_RECURSION_LIMIT = "GRAPH_RECURSION_LIMIT"
    INVALID_CONCURRENT_GRAPH_UPDATE = "INVALID_CONCURRENT_GRAPH_UPDATE"
    INVALID_GRAPH_NODE_RETURN_VALUE = "INVALID_GRAPH_NODE_RETURN_VALUE"
    MULTIPLE_SUBGRAPHS = "MULTIPLE_SUBGRAPHS"
    INVALID_CHAT_HISTORY = "INVALID_CHAT_HISTORY"


def create_error_message(*, message: str, error_code: ErrorCode) -> str:
    return (
        f"{message}\n"
        "For troubleshooting, visit: https://python.langchain.com/docs/"
        f"troubleshooting/errors/{error_code.value}"
    )


class GraphRecursionError(RecursionError):
    """Raised when the graph has exhausted the maximum number of steps.

    This prevents infinite loops. To increase the maximum number of steps,
    run your graph with a config specifying a higher `recursion_limit`.

    Troubleshooting Guides:

    - [GRAPH_RECURSION_LIMIT](https://python.langchain.com/docs/troubleshooting/errors/GRAPH_RECURSION_LIMIT)

    Examples:

        graph = builder.compile()
        graph.invoke(
            {"messages": [("user", "Hello, world!")]},
            # The config is the second positional argument
            {"recursion_limit": 1000},
        )
    """

    pass


class InvalidUpdateError(Exception):
    """Raised when attempting to update a channel with an invalid set of updates.

    Troubleshooting Guides:

    - [INVALID_CONCURRENT_GRAPH_UPDATE](https://python.langchain.com/docs/troubleshooting/errors/INVALID_CONCURRENT_GRAPH_UPDATE)
    - [INVALID_GRAPH_NODE_RETURN_VALUE](https://python.langchain.com/docs/troubleshooting/errors/INVALID_GRAPH_NODE_RETURN_VALUE)
    """

    pass


class GraphBubbleUp(Exception):
    pass


class GraphInterrupt(GraphBubbleUp):
    """Raised when a subgraph is interrupted, suppressed by the root graph.
    Never raised directly, or surfaced to the user."""

    def __init__(self, interrupts: Sequence[Interrupt] = ()) -> None:
        super().__init__(interrupts)


class NodeInterrupt(GraphInterrupt):
    """Raised by a node to interrupt execution."""

    def __init__(self, value: Any) -> None:
        super().__init__([Interrupt(value=value)])


class GraphDelegate(GraphBubbleUp):
    """Raised when a graph is delegated (for distributed mode)."""

    def __init__(self, *args: dict[str, Any]) -> None:
        super().__init__(*args)


class ParentCommand(GraphBubbleUp):
    args: tuple[Command]

    def __init__(self, command: Command) -> None:
        super().__init__(command)


class EmptyInputError(Exception):
    """Raised when graph receives an empty input."""

    pass


class TaskNotFound(Exception):
    """Raised when the executor is unable to find a task (for distributed mode)."""

    pass


class CheckpointNotLatest(Exception):
    """Raised when the checkpoint is not the latest version (for distributed mode)."""

    pass
