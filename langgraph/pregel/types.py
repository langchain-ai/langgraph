from collections import deque
from typing import Any, Literal, NamedTuple, Optional, Union

from langchain_core.runnables import Runnable, RunnableConfig


class PregelTaskDescription(NamedTuple):
    name: str
    input: Any


class PregelExecutableTask(NamedTuple):
    name: str
    input: Any
    proc: Runnable
    writes: deque[tuple[str, Any]]
    config: Optional[RunnableConfig] = None


class StateSnapshot(NamedTuple):
    values: Union[dict[str, Any], Any]
    """Current values of channels"""
    next: tuple[str]
    """Nodes to execute in the next step, if any"""
    config: RunnableConfig
    """Config used to fetch this snapshot"""
    parent_config: Optional[RunnableConfig] = None
    """Config used to fetch the parent snapshot, if any"""


All = Literal["*"]
