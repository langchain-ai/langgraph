from collections import deque
from typing import Any, Callable, Literal, NamedTuple, Optional, Type, Union

from langchain_core.runnables import Runnable, RunnableConfig

from langgraph.checkpoint.base import CheckpointMetadata
from langgraph.constants import Interrupt


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
        Type[Exception], tuple[Type[Exception], ...], Callable[[Exception], bool]
    ] = default_retry_on
    """List of exception classes that should trigger a retry, or a callable that returns True for exceptions that should trigger a retry."""


class CachePolicy(NamedTuple):
    """Configuration for caching nodes."""

    cache_key: Optional[Callable[[Any, Optional[RunnableConfig]], str]]
    """
    A function that takes in the input and config, and returns a string key 
    under which the output should be cached.
    """
    # TODO: implement cache_ttl
    # cache_ttl: Optional[float] = None
    # """
    # Time-to-live for the cached value, in seconds. If not provided, the value will be cached indefinitely.
    # We'd probably want to store this in a bucket way instead of a TTL timeline.
    # """


class PregelTask(NamedTuple):
    id: str
    name: str
    error: Optional[Exception] = None
    interrupts: tuple[Interrupt, ...] = ()
    state: Union[None, RunnableConfig, "StateSnapshot"] = None


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
    path: tuple[str, ...]
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


All = Literal["*"]

StreamMode = Literal["values", "updates", "debug"]
"""How the stream method should emit outputs.

- 'values': Emit all values of the state for each step.
- 'updates': Emit only the node name(s) and updates
    that were returned by the node(s) **after** each step.
- 'debug': Emit debug events for each step.
"""
