from abc import ABC
from collections import defaultdict
from datetime import datetime, timezone
from typing import (
    Any,
    AsyncIterator,
    Dict,
    Iterator,
    List,
    Literal,
    NamedTuple,
    Optional,
    TypedDict,
    TypeVar,
    Union,
)

from langchain_core.runnables import ConfigurableFieldSpec, RunnableConfig

from langgraph.channels.base import BaseChannel
from langgraph.checkpoint.id import uuid6
from langgraph.constants import Send
from langgraph.serde.base import SerializerProtocol
from langgraph.serde.jsonplus import JsonPlusSerializer

V = TypeVar("V", int, float, str)


# Marked as total=False to allow for future expansion.
class CheckpointMetadata(TypedDict, total=False):
    source: Literal["input", "loop", "update"]
    """The source of the checkpoint.
    - "input": The checkpoint was created from an input to invoke/stream/batch.
    - "loop": The checkpoint was created from inside the pregel loop.
    - "update": The checkpoint was created from a manual state update.
    """
    step: int
    """The step number of the checkpoint.
    -1 for the first "input" checkpoint.
    0 for the first "loop" checkpoint.
    ... for the nth checkpoint afterwards.
    """
    writes: dict[str, Any]
    """The writes that were made between the previous checkpoint and this one.

    Mapping from node name to writes emitted by that node.
    """
    score: Optional[int]
    """The score of the checkpoint.
    
    The score can be used to mark a checkpoint as "good".
    """


class Checkpoint(TypedDict):
    """State snapshot at a given point in time."""

    v: int
    """The version of the checkpoint format. Currently 1."""
    id: str
    """The ID of the checkpoint. This is both unique and monotonically 
    increasing, so can be used for sorting checkpoints from first to last."""
    ts: str
    """The timestamp of the checkpoint in ISO 8601 format."""
    channel_values: dict[str, Any]
    """The values of the channels at the time of the checkpoint.
    
    Mapping from channel name to channel snapshot value.
    """
    channel_versions: defaultdict[str, Union[str, int, float]]
    """The versions of the channels at the time of the checkpoint.
    
    The keys are channel names and the values are the logical time step
    at which the channel was last updated.
    """
    versions_seen: defaultdict[str, defaultdict[str, Union[str, int, float]]]
    """Map from node ID to map from channel name to version seen.
    
    This keeps track of the versions of the channels that each node has seen.
    
    Used to determine which nodes to execute next.
    """
    pending_sends: List[Send]
    """List of packets sent to nodes but not yet processed.
    Cleared by the next checkpoint."""


def _seen_dict():
    return defaultdict(int)


def empty_checkpoint() -> Checkpoint:
    return Checkpoint(
        v=1,
        id=str(uuid6(clock_seq=-2)),
        ts=datetime.now(timezone.utc).isoformat(),
        channel_values={},
        channel_versions=defaultdict(int),
        versions_seen=defaultdict(_seen_dict),
        pending_sends=[],
    )


def copy_checkpoint(checkpoint: Checkpoint) -> Checkpoint:
    return Checkpoint(
        v=checkpoint["v"],
        ts=checkpoint["ts"],
        id=checkpoint["id"],
        channel_values=checkpoint["channel_values"].copy(),
        channel_versions=defaultdict(int, checkpoint["channel_versions"]),
        versions_seen=defaultdict(
            _seen_dict,
            {k: defaultdict(int, v) for k, v in checkpoint["versions_seen"].items()},
        ),
        pending_sends=checkpoint.get("pending_sends", []).copy(),
    )


class CheckpointTuple(NamedTuple):
    config: RunnableConfig
    checkpoint: Checkpoint
    metadata: CheckpointMetadata
    parent_config: Optional[RunnableConfig] = None


CheckpointThreadId = ConfigurableFieldSpec(
    id="thread_id",
    annotation=str,
    name="Thread ID",
    description=None,
    default="",
    is_shared=True,
)

CheckpointThreadTs = ConfigurableFieldSpec(
    id="thread_ts",
    annotation=Optional[str],
    name="Thread Timestamp",
    description="Pass to fetch a past checkpoint. If None, fetches the latest checkpoint.",
    default=None,
    is_shared=True,
)


class BaseCheckpointSaver(ABC):
    serde: SerializerProtocol = JsonPlusSerializer()

    def __init__(
        self,
        *,
        serde: Optional[SerializerProtocol] = None,
    ) -> None:
        self.serde = serde or self.serde

    @property
    def config_specs(self) -> list[ConfigurableFieldSpec]:
        return [CheckpointThreadId, CheckpointThreadTs]

    def get(self, config: RunnableConfig) -> Optional[Checkpoint]:
        if value := self.get_tuple(config):
            return value.checkpoint

    def get_tuple(self, config: RunnableConfig) -> Optional[CheckpointTuple]:
        raise NotImplementedError

    def list(
        self,
        config: Optional[RunnableConfig],
        *,
        filter: Optional[Dict[str, Any]] = None,
        before: Optional[RunnableConfig] = None,
        limit: Optional[int] = None,
    ) -> Iterator[CheckpointTuple]:
        raise NotImplementedError

    def put(
        self,
        config: RunnableConfig,
        checkpoint: Checkpoint,
        metadata: CheckpointMetadata,
    ) -> RunnableConfig:
        raise NotImplementedError

    async def aget(self, config: RunnableConfig) -> Optional[Checkpoint]:
        if value := await self.aget_tuple(config):
            return value.checkpoint

    async def aget_tuple(self, config: RunnableConfig) -> Optional[CheckpointTuple]:
        raise NotImplementedError

    def alist(
        self,
        config: Optional[RunnableConfig],
        *,
        filter: Optional[Dict[str, Any]] = None,
        before: Optional[RunnableConfig] = None,
        limit: Optional[int] = None,
    ) -> AsyncIterator[CheckpointTuple]:
        raise NotImplementedError
        yield

    async def aput(
        self,
        config: RunnableConfig,
        checkpoint: Checkpoint,
        metadata: CheckpointMetadata,
    ) -> RunnableConfig:
        raise NotImplementedError

    def get_next_version(self, current: Optional[V], channel: BaseChannel) -> V:
        return current + 1 if current is not None else 1
