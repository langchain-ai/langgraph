from collections import deque
from typing import Any, Callable, Literal, NamedTuple, Optional, Type, Union

from langchain_core.runnables import Runnable, RunnableConfig

from langgraph.checkpoint.base import CheckpointMetadata


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


class PregelTaskDescription(NamedTuple):
    name: str
    input: Any


class PregelExecutableTask(NamedTuple):
    name: str
    input: Any
    proc: Runnable
    writes: deque[tuple[str, Any]]
    config: RunnableConfig
    triggers: list[str]
    retry_policy: Optional[RetryPolicy]
    id: str


class StateSnapshot(NamedTuple):
    values: Union[dict[str, Any], Any]
    """Current values of channels"""
    next: tuple[str]
    """Nodes to execute in the next step, if any"""
    config: RunnableConfig
    """Config used to fetch this snapshot"""
    metadata: Optional[CheckpointMetadata]
    """Metadata associated with this snapshot"""
    created_at: Optional[str]
    """Timestamp of snapshot creation"""
    parent_config: Optional[RunnableConfig] = None
    """Config used to fetch the parent snapshot, if any"""


All = Literal["*"]
