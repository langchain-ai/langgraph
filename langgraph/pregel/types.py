from collections import deque
from typing import Any, Literal, NamedTuple, Optional, Union

from langchain_core.runnables import Runnable, RunnableConfig

from langgraph.checkpoint.base import CheckpointMetadata


class PregelTaskDescription(NamedTuple):
    name: str
    input: Any


class PregelExecutableTask(NamedTuple):
    name: str
    input: Any
    proc: Runnable
    writes: deque[tuple[str, Any]]
    config: Optional[RunnableConfig]
    triggers: list[str]


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
