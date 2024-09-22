from typing import Any, Sequence

from langgraph.checkpoint.base import EmptyChannelError  # noqa: F401
from langgraph.constants import Interrupt

# EmptyChannelError re-exported for backwards compatibility


class GraphRecursionError(RecursionError):
    """Raised when the graph has exhausted the maximum number of steps.

    This prevents infinite loops. To increase the maximum number of steps,
    run your graph with a config specifying a higher `recursion_limit`.

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
    """Raised when attempting to update a channel with an invalid set of updates."""

    pass


class GraphInterrupt(Exception):
    """Raised when a subgraph is interrupted, suppressed by the root graph.
    Never raised directly, or surfaced to the user."""

    def __init__(self, interrupts: Sequence[Interrupt] = ()) -> None:
        super().__init__(interrupts)


class NodeInterrupt(GraphInterrupt):
    """Raised by a node to interrupt execution."""

    def __init__(self, value: Any) -> None:
        super().__init__([Interrupt(value)])


class GraphDelegate(Exception):
    """Raised when a graph is delegated (for distributed mode)."""

    def __init__(self, *args: dict[str, Any]) -> None:
        super().__init__(*args)


class EmptyInputError(Exception):
    """Raised when graph receives an empty input."""

    pass


class TaskNotFound(Exception):
    """Raised when the executor is unable to find a task (for distributed mode)."""

    pass


class CheckpointNotLatest(Exception):
    """Raised when the checkpoint is not the latest version (for distributed mode)."""

    pass
