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
    Tuple,
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
    """Metadata associated with a checkpoint."""

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
    channel_versions: dict[str, Union[str, int, float]]
    """The versions of the channels at the time of the checkpoint.
    
    The keys are channel names and the values are the logical time step
    at which the channel was last updated.
    """
    versions_seen: defaultdict[str, dict[str, Union[str, int, float]]]
    """Map from node ID to map from channel name to version seen.
    
    This keeps track of the versions of the channels that each node has seen.
    
    Used to determine which nodes to execute next.
    """
    pending_sends: List[Send]
    """List of packets sent to nodes but not yet processed.
    Cleared by the next checkpoint."""


def empty_checkpoint() -> Checkpoint:
    return Checkpoint(
        v=1,
        id=str(uuid6(clock_seq=-2)),
        ts=datetime.now(timezone.utc).isoformat(),
        channel_values={},
        channel_versions={},
        versions_seen=defaultdict(dict),
        pending_sends=[],
    )


def copy_checkpoint(checkpoint: Checkpoint) -> Checkpoint:
    return Checkpoint(
        v=checkpoint["v"],
        ts=checkpoint["ts"],
        id=checkpoint["id"],
        channel_values=checkpoint["channel_values"].copy(),
        channel_versions=checkpoint["channel_versions"].copy(),
        versions_seen=defaultdict(
            dict,
            {k: v.copy() for k, v in checkpoint["versions_seen"].items()},
        ),
        pending_sends=checkpoint.get("pending_sends", []).copy(),
    )


class CheckpointTuple(NamedTuple):
    """A tuple containing a checkpoint and its associated data."""

    config: RunnableConfig
    checkpoint: Checkpoint
    metadata: CheckpointMetadata
    parent_config: Optional[RunnableConfig] = None
    pending_writes: Optional[List[Tuple[str, str, Any]]] = None


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
    """Base class for creating a graph checkpointer.

    Checkpointers allow LangGraph agents to persist their state
    within and across multiple interactions.

    Attributes:
        serde (SerializerProtocol): Serializer for encoding/decoding checkpoints.

    Note:
        When creating a custom checkpoint saver, consider implementing async
        versions to avoid blocking the main thread.
    """

    serde: SerializerProtocol = JsonPlusSerializer()

    def __init__(
        self,
        *,
        serde: Optional[SerializerProtocol] = None,
    ) -> None:
        self.serde = serde or self.serde

    @property
    def config_specs(self) -> list[ConfigurableFieldSpec]:
        """Define the configuration options for the checkpoint saver.

        Returns:
            list[ConfigurableFieldSpec]: List of configuration field specs.
        """
        return [CheckpointThreadId, CheckpointThreadTs]

    def get(self, config: RunnableConfig) -> Optional[Checkpoint]:
        """Fetch a checkpoint using the given configuration.

        Args:
            config (RunnableConfig): Configuration specifying which checkpoint to retrieve.

        Returns:
            Optional[Checkpoint]: The requested checkpoint, or None if not found.
        """
        if value := self.get_tuple(config):
            return value.checkpoint

    def get_tuple(self, config: RunnableConfig) -> Optional[CheckpointTuple]:
        """Fetch a checkpoint tuple using the given configuration.

        Args:
            config (RunnableConfig): Configuration specifying which checkpoint to retrieve.

        Returns:
            Optional[CheckpointTuple]: The requested checkpoint tuple, or None if not found.

        Raises:
            NotImplementedError: Implement this method in your custom checkpoint saver.
        """
        raise NotImplementedError

    def list(
        self,
        config: Optional[RunnableConfig],
        *,
        filter: Optional[Dict[str, Any]] = None,
        before: Optional[RunnableConfig] = None,
        limit: Optional[int] = None,
    ) -> Iterator[CheckpointTuple]:
        """List checkpoints that match the given criteria.

        Args:
            config (Optional[RunnableConfig]): Base configuration for filtering checkpoints.
            filter (Optional[Dict[str, Any]]): Additional filtering criteria.
            before (Optional[RunnableConfig]): List checkpoints created before this configuration.
            limit (Optional[int]): Maximum number of checkpoints to return.

        Returns:
            Iterator[CheckpointTuple]: Iterator of matching checkpoint tuples.

        Raises:
            NotImplementedError: Implement this method in your custom checkpoint saver.
        """
        raise NotImplementedError

    def put(
        self,
        config: RunnableConfig,
        checkpoint: Checkpoint,
        metadata: CheckpointMetadata,
    ) -> RunnableConfig:
        """Store a checkpoint with its configuration and metadata.

        Args:
            config (RunnableConfig): Configuration for the checkpoint.
            checkpoint (Checkpoint): The checkpoint to store.
            metadata (CheckpointMetadata): Additional metadata for the checkpoint.

        Returns:
            RunnableConfig: Updated configuration after storing the checkpoint.

        Raises:
            NotImplementedError: Implement this method in your custom checkpoint saver.
        """
        raise NotImplementedError

    def put_writes(
        self,
        config: RunnableConfig,
        writes: List[Tuple[str, Any]],
        task_id: str,
    ) -> None:
        """Store intermediate writes linked to a checkpoint.

        Args:
            config (RunnableConfig): Configuration of the related checkpoint.
            writes (List[Tuple[str, Any]]): List of writes to store.
            task_id (str): Identifier for the task creating the writes.

        Raises:
            NotImplementedError: Implement this method in your custom checkpoint saver.
        """
        raise NotImplementedError(
            "This method was added in langgraph 0.1.7. Please update your checkpoint saver to implement it."
        )

    async def aget(self, config: RunnableConfig) -> Optional[Checkpoint]:
        """Asynchronously fetch a checkpoint using the given configuration.

        Args:
            config (RunnableConfig): Configuration specifying which checkpoint to retrieve.

        Returns:
            Optional[Checkpoint]: The requested checkpoint, or None if not found.
        """
        if value := await self.aget_tuple(config):
            return value.checkpoint

    async def aget_tuple(self, config: RunnableConfig) -> Optional[CheckpointTuple]:
        """Asynchronously fetch a checkpoint tuple using the given configuration.

        Args:
            config (RunnableConfig): Configuration specifying which checkpoint to retrieve.

        Returns:
            Optional[CheckpointTuple]: The requested checkpoint tuple, or None if not found.

        Raises:
            NotImplementedError: Implement this method in your custom checkpoint saver.
        """
        raise NotImplementedError

    async def alist(
        self,
        config: Optional[RunnableConfig],
        *,
        filter: Optional[Dict[str, Any]] = None,
        before: Optional[RunnableConfig] = None,
        limit: Optional[int] = None,
    ) -> AsyncIterator[CheckpointTuple]:
        """Asynchronously list checkpoints that match the given criteria.

        Args:
            config (Optional[RunnableConfig]): Base configuration for filtering checkpoints.
            filter (Optional[Dict[str, Any]]): Additional filtering criteria for metadata.
            before (Optional[RunnableConfig]): List checkpoints created before this configuration.
            limit (Optional[int]): Maximum number of checkpoints to return.

        Returns:
            AsyncIterator[CheckpointTuple]: Async iterator of matching checkpoint tuples.

        Raises:
            NotImplementedError: Implement this method in your custom checkpoint saver.
        """
        raise NotImplementedError
        yield

    async def aput(
        self,
        config: RunnableConfig,
        checkpoint: Checkpoint,
        metadata: CheckpointMetadata,
    ) -> RunnableConfig:
        """Asynchronously store a checkpoint with its configuration and metadata.

        Args:
            config (RunnableConfig): Configuration for the checkpoint.
            checkpoint (Checkpoint): The checkpoint to store.
            metadata (CheckpointMetadata): Additional metadata for the checkpoint.

        Returns:
            RunnableConfig: Updated configuration after storing the checkpoint.

        Raises:
            NotImplementedError: Implement this method in your custom checkpoint saver.
        """
        raise NotImplementedError

    async def aput_writes(
        self,
        config: RunnableConfig,
        writes: List[Tuple[str, Any]],
        task_id: str,
    ) -> None:
        """Asynchronously store intermediate writes linked to a checkpoint.

        Args:
            config (RunnableConfig): Configuration of the related checkpoint.
            writes (List[Tuple[str, Any]]): List of writes to store.
            task_id (str): Identifier for the task creating the writes.

        Raises:
            NotImplementedError: Implement this method in your custom checkpoint saver.
        """
        raise NotImplementedError(
            "This method was added in langgraph 0.1.7. Please update your checkpoint saver to implement it."
        )

    def get_next_version(self, current: Optional[V], channel: BaseChannel) -> V:
        """Generate the next version ID for a channel.

        Default is to use integer versions, incrementing by 1. If you override, you can use str/int/float versions,
        as long as they are monotonically increasing.

        Args:
            current (Optional[V]): The current version identifier (int, float, or str).
            channel (BaseChannel): The channel being versioned.

        Returns:
            V: The next version identifier, which must be increasing.
        """
        return current + 1 if current is not None else 1
