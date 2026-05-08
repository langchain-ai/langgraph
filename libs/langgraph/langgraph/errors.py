from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from enum import Enum
from typing import Any, Literal
from warnings import warn

# EmptyChannelError is re-exported from langgraph.channels.base
from langgraph.checkpoint.base import EmptyChannelError  # noqa: F401
from typing_extensions import deprecated

from langgraph.types import Command, Interrupt
from langgraph.warnings import LangGraphDeprecatedSinceV10

__all__ = (
    "EmptyChannelError",
    "ErrorCode",
    "GraphDrained",
    "GraphRecursionError",
    "InvalidUpdateError",
    "GraphBubbleUp",
    "GraphInterrupt",
    "NodeError",
    "NodeInterrupt",
    "NodeTimeoutError",
    "ParentCommand",
    "EmptyInputError",
    "TaskNotFound",
)


class ErrorCode(Enum):
    GRAPH_RECURSION_LIMIT = "GRAPH_RECURSION_LIMIT"
    INVALID_CONCURRENT_GRAPH_UPDATE = "INVALID_CONCURRENT_GRAPH_UPDATE"
    INVALID_GRAPH_NODE_RETURN_VALUE = "INVALID_GRAPH_NODE_RETURN_VALUE"
    MULTIPLE_SUBGRAPHS = "MULTIPLE_SUBGRAPHS"
    INVALID_CHAT_HISTORY = "INVALID_CHAT_HISTORY"


def create_error_message(*, message: str, error_code: ErrorCode) -> str:
    return (
        f"{message}\n"
        "For troubleshooting, visit: https://docs.langchain.com/oss/python/langgraph/"
        f"errors/{error_code.value}"
    )


class GraphBubbleUp(Exception):
    pass


class GraphDrained(GraphBubbleUp):
    """Raised when a graph run exits early due to a drain request.

    This indicates the graph stopped cooperatively at a superstep boundary
    because `RunControl.request_drain()` was called (e.g., in response to
    SIGTERM). The checkpoint is saved and the run can be resumed later.
    """

    def __init__(self, reason: str = "shutdown") -> None:
        self.reason = reason
        super().__init__(f"Graph drained: {reason}")


class GraphRecursionError(RecursionError):
    """Raised when the graph has exhausted the maximum number of steps.

    This prevents infinite loops. To increase the maximum number of steps,
    run your graph with a config specifying a higher `recursion_limit`.

    Troubleshooting guides:

    - [`GRAPH_RECURSION_LIMIT`](https://docs.langchain.com/oss/python/langgraph/GRAPH_RECURSION_LIMIT)

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

    Troubleshooting guides:

    - [`INVALID_CONCURRENT_GRAPH_UPDATE`](https://docs.langchain.com/oss/python/langgraph/INVALID_CONCURRENT_GRAPH_UPDATE)
    - [`INVALID_GRAPH_NODE_RETURN_VALUE`](https://docs.langchain.com/oss/python/langgraph/INVALID_GRAPH_NODE_RETURN_VALUE)
    """

    pass


class GraphInterrupt(GraphBubbleUp):
    """Raised when a subgraph is interrupted, suppressed by the root graph.
    Never raised directly, or surfaced to the user."""

    def __init__(self, interrupts: Sequence[Interrupt] = ()) -> None:
        super().__init__(interrupts)


@deprecated(
    "NodeInterrupt is deprecated. Please use [`interrupt`][langgraph.types.interrupt] instead.",
    category=None,
)
class NodeInterrupt(GraphInterrupt):
    """Raised by a node to interrupt execution."""

    def __init__(self, value: Any, id: str | None = None) -> None:
        warn(
            "NodeInterrupt is deprecated. Please use `langgraph.types.interrupt` instead.",
            LangGraphDeprecatedSinceV10,
            stacklevel=2,
        )
        if id is None:
            super().__init__([Interrupt(value=value)])
        else:
            super().__init__([Interrupt(value=value, id=id)])


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


@dataclass(frozen=True, slots=True)
class NodeError:
    """Failure context passed to a node-level error handler.

    Inject by adding a parameter typed `NodeError` to a handler registered via
    `StateGraph.add_node(..., error_handler=...)`:

    ```python
    def handler(state: State, error: NodeError) -> Command:
        return Command(update={"status": f"recovered from {error.node}: {error.error}"})
    ```
    """

    node: str
    """Name of the node whose execution failed."""

    error: BaseException
    """Exception raised by the failed node."""


class NodeTimeoutError(Exception):
    """Raised when a node invocation exceeds one of its configured timeouts.

    Does **not** inherit from the built-in `TimeoutError` (a subclass of
    `OSError`) so that the default `RetryPolicy` treats it as retryable.

    Both `idle_timeout` and `run_timeout` reflect the configured policy at the
    time of the failure (each is `None` if not configured). `kind` and
    `timeout` identify which one fired.
    """

    node: str
    timeout: float
    run_timeout: float | None
    idle_timeout: float | None
    elapsed: float
    kind: Literal["idle", "run"]

    def __init__(
        self,
        node: str,
        elapsed: float,
        *,
        kind: Literal["idle", "run"],
        idle_timeout: float | None = None,
        run_timeout: float | None = None,
    ) -> None:
        if kind == "idle":
            if idle_timeout is None:
                raise ValueError("idle_timeout is required when kind='idle'")
            message = (
                f"Node '{node}' exceeded its idle timeout of "
                f"{idle_timeout:.3f}s without making progress "
                f"(elapsed: {elapsed:.3f}s)."
            )
            self.timeout = idle_timeout
        elif kind == "run":
            if run_timeout is None:
                raise ValueError("run_timeout is required when kind='run'")
            message = (
                f"Node '{node}' exceeded its run timeout of "
                f"{run_timeout:.3f}s (elapsed: {elapsed:.3f}s)."
            )
            self.timeout = run_timeout
        else:
            raise ValueError("kind must be 'idle' or 'run'")
        super().__init__(message)
        self.node = node
        self.elapsed = elapsed
        self.kind = kind
        self.idle_timeout = idle_timeout
        self.run_timeout = run_timeout
