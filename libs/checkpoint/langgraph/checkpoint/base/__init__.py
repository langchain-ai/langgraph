from collections.abc import Iterator, Mapping, Sequence
from datetime import datetime, timezone
from typing import (  # noqa: UP035
    Any,
    Dict,
    Generic,
    List,
    Literal,
    NamedTuple,
    Optional,
    Tuple,
    TypedDict,
    TypeVar,
    Union,
)

from langgraph.checkpoint.base.id import uuid6
from langgraph.checkpoint.serde.base import SerializerProtocol, maybe_add_typed_methods
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
from langgraph.checkpoint.serde.types import (
    ERROR,
    INTERRUPT,
    RESUME,
    SCHEDULED,
    ChannelProtocol,
    SendProtocol,
)

V = TypeVar("V", int, float, str)
PendingWrite = Tuple[str, str, Any]


class CheckpointConfig(TypedDict, total=False):
    """Configuration for a Runnable."""

    metadata: dict[str, Any]
    """
    Metadata for this call and any sub-calls (eg. a Chain calling an LLM).
    Keys should be strings, values should be JSON-serializable.
    """

    configurable: dict[str, Any]
    """
    Runtime values for attributes previously made configurable on this Runnable,
    or sub-Runnables, through .configurable_fields() or .configurable_alternatives().
    Check .output_schema() for a description of the attributes that have been made
    configurable.
    """


# Marked as total=False to allow for future expansion.
class CheckpointMetadata(TypedDict, total=False):
    """Metadata associated with a checkpoint."""

    source: Literal["input", "loop", "update", "fork"]
    """The source of the checkpoint.

    - "input": The checkpoint was created from an input to invoke/stream/batch.
    - "loop": The checkpoint was created from inside the pregel loop.
    - "update": The checkpoint was created from a manual state update.
    - "fork": The checkpoint was created as a copy of another checkpoint.
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
    parents: dict[str, str]
    """The IDs of the parent checkpoints.

    Mapping from checkpoint namespace to checkpoint ID.
    """


class TaskInfo(TypedDict):
    status: Literal["scheduled", "success", "error"]


ChannelVersions = dict[str, Union[str, int, float]]


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
    Mapping from channel name to deserialized channel snapshot value.
    """
    channel_versions: ChannelVersions
    """The versions of the channels at the time of the checkpoint.
    The keys are channel names and the values are monotonically increasing
    version strings for each channel.
    """
    versions_seen: dict[str, ChannelVersions]
    """Map from node ID to map from channel name to version seen.
    This keeps track of the versions of the channels that each node has seen.
    Used to determine which nodes to execute next.
    """
    pending_sends: List[SendProtocol]
    """List of inputs pushed to nodes but not yet processed.
    Cleared by the next checkpoint."""


def empty_checkpoint() -> Checkpoint:
    return Checkpoint(
        v=1,
        id=str(uuid6(clock_seq=-2)),
        ts=datetime.now(timezone.utc).isoformat(),
        channel_values={},
        channel_versions={},
        versions_seen={},
        pending_sends=[],
    )


def copy_checkpoint(checkpoint: Checkpoint) -> Checkpoint:
    return Checkpoint(
        v=checkpoint["v"],
        ts=checkpoint["ts"],
        id=checkpoint["id"],
        channel_values=checkpoint["channel_values"].copy(),
        channel_versions=checkpoint["channel_versions"].copy(),
        versions_seen={k: v.copy() for k, v in checkpoint["versions_seen"].items()},
        pending_sends=checkpoint.get("pending_sends", []).copy(),
    )


def create_checkpoint(
    checkpoint: Checkpoint,
    channels: Optional[Mapping[str, ChannelProtocol]],
    step: int,
    *,
    id: Optional[str] = None,
) -> Checkpoint:
    """Create a checkpoint for the given channels."""
    ts = datetime.now(timezone.utc).isoformat()
    if channels is None:
        values = checkpoint["channel_values"]
    else:
        values = {}
        for k, v in channels.items():
            if k not in checkpoint["channel_versions"]:
                continue
            try:
                values[k] = v.checkpoint()
            except EmptyChannelError:
                pass
    return Checkpoint(
        v=1,
        ts=ts,
        id=id or str(uuid6(clock_seq=step)),
        channel_values=values,
        channel_versions=checkpoint["channel_versions"],
        versions_seen=checkpoint["versions_seen"],
        pending_sends=checkpoint.get("pending_sends", []),
    )


class CheckpointTuple(NamedTuple):
    """A tuple containing a checkpoint and its associated data."""

    config: CheckpointConfig
    checkpoint: Checkpoint
    metadata: CheckpointMetadata
    parent_config: Optional[CheckpointConfig] = None
    pending_writes: Optional[List[PendingWrite]] = None


class BaseCheckpointSaver(Generic[V]):
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
        self.serde = maybe_add_typed_methods(serde or self.serde)

    def get(self, config: CheckpointConfig) -> Optional[Checkpoint]:
        """Fetch a checkpoint using the given configuration.

        Args:
            config (CheckpointConfig): Configuration specifying which checkpoint to retrieve.

        Returns:
            Optional[Checkpoint]: The requested checkpoint, or None if not found.
        """
        if value := self.get_tuple(config):
            return value.checkpoint

    def get_tuple(self, config: CheckpointConfig) -> Optional[CheckpointTuple]:
        """Fetch a checkpoint tuple using the given configuration.

        Args:
            config (CheckpointConfig): Configuration specifying which checkpoint to retrieve.

        Returns:
            Optional[CheckpointTuple]: The requested checkpoint tuple, or None if not found.

        Raises:
            NotImplementedError: Implement this method in your custom checkpoint saver.
        """
        raise NotImplementedError

    def list(
        self,
        config: Optional[CheckpointConfig],
        *,
        filter: Optional[Dict[str, Any]] = None,
        before: Optional[CheckpointConfig] = None,
        limit: Optional[int] = None,
    ) -> Iterator[CheckpointTuple]:
        """List checkpoints that match the given criteria.

        Args:
            config (Optional[CheckpointConfig]): Base configuration for filtering checkpoints.
            filter (Optional[Dict[str, Any]]): Additional filtering criteria.
            before (Optional[CheckpointConfig]): List checkpoints created before this configuration.
            limit (Optional[int]): Maximum number of checkpoints to return.

        Returns:
            Iterator[CheckpointTuple]: Iterator of matching checkpoint tuples.

        Raises:
            NotImplementedError: Implement this method in your custom checkpoint saver.
        """
        raise NotImplementedError

    def put(
        self,
        config: CheckpointConfig,
        checkpoint: Checkpoint,
        metadata: CheckpointMetadata,
        new_versions: ChannelVersions,
    ) -> CheckpointConfig:
        """Store a checkpoint with its configuration and metadata.

        Args:
            config (CheckpointConfig): Configuration for the checkpoint.
            checkpoint (Checkpoint): The checkpoint to store.
            metadata (CheckpointMetadata): Additional metadata for the checkpoint.
            new_versions (ChannelVersions): New channel versions as of this write.

        Returns:
            CheckpointConfig: Updated configuration after storing the checkpoint.

        Raises:
            NotImplementedError: Implement this method in your custom checkpoint saver.
        """
        raise NotImplementedError

    def put_writes(
        self,
        config: CheckpointConfig,
        writes: Sequence[Tuple[str, Any]],
        task_id: str,
        task_path: str = "",
    ) -> None:
        """Store intermediate writes linked to a checkpoint.

        Args:
            config (CheckpointConfig): Configuration of the related checkpoint.
            writes (List[Tuple[str, Any]]): List of writes to store.
            task_id (str): Identifier for the task creating the writes.
            task_path (str): Path of the task creating the writes.

        Raises:
            NotImplementedError: Implement this method in your custom checkpoint saver.
        """
        raise NotImplementedError

    def get_next_version(self, current: Optional[V], channel: ChannelProtocol) -> V:
        """Generate the next version ID for a channel.

        Default is to use integer versions, incrementing by 1. If you override, you can use str/int/float versions,
        as long as they are monotonically increasing.

        Args:
            current (Optional[V]): The current version identifier (int, float, or str).
            channel (BaseChannel): The channel being versioned.

        Returns:
            V: The next version identifier, which must be increasing.
        """
        if isinstance(current, str):
            raise NotImplementedError
        elif current is None:
            return 1
        else:
            return current + 1


class EmptyChannelError(Exception):
    """Raised when attempting to get the value of a channel that hasn't been updated
    for the first time yet."""

    pass


def get_checkpoint_id(config: CheckpointConfig) -> Optional[str]:
    """Get checkpoint ID in a backwards-compatible manner (fallback on thread_ts)."""
    return config["configurable"].get(
        "checkpoint_id", config["configurable"].get("thread_ts")
    )


def get_checkpoint_metadata(
    config: CheckpointConfig, metadata: CheckpointMetadata
) -> CheckpointMetadata:
    """Get checkpoint metadata in a backwards-compatible manner."""
    metadata = metadata.copy()
    for obj in (config.get("metadata"), config.get("configurable")):
        if not obj:
            continue
        for key in obj:
            if key in metadata or key in EXCLUDED_METADATA_KEYS or key.startswith("__"):
                continue
            v = obj[key]
            if isinstance(v, (str, int, bool, float)):
                metadata[key] = v  # type: ignore[literal-required]
    return metadata


"""
Mapping from error type to error index.
Regular writes just map to their index in the list of writes being saved.
Special writes (e.g. errors) map to negative indices, to avoid those writes from
conflicting with regular writes.
Each Checkpointer implementation should use this mapping in put_writes.
"""
WRITES_IDX_MAP = {ERROR: -1, SCHEDULED: -2, INTERRUPT: -3, RESUME: -4}

EXCLUDED_METADATA_KEYS = {
    "checkpoint_id",
    "checkpoint_ns",
    "checkpoint_map",
}
