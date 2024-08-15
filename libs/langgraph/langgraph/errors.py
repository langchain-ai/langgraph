from typing import Any

from langchain_core.runnables import ensure_config

from langgraph.checkpoint.base import EmptyChannelError
from langgraph.constants import Interrupt


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
    """Raised when attempting to update a channel with an invalid sequence of updates."""

    pass


class GraphInterrupt(Exception):
    """Raised when a subgraph is interrupted."""

    def __init__(self, interrupts: list[Interrupt]) -> None:
        super().__init__(interrupts)


class NodeInterrupt(GraphInterrupt):
    """Raised by a node to interrupt execution."""

    def __init__(self, *values: Any) -> None:
        config = ensure_config()
        node = config["configurable"]["checkpoint_ns"]
        super().__init__([Interrupt("during", node, v) for v in values])


class EmptyInputError(Exception):
    """Raised when graph receives an empty input."""

    pass


__all__ = [
    "GraphRecursionError",
    "InvalidUpdateError",
    "GraphInterrupt",
    "EmptyInputError",
    "EmptyChannelError",
]
