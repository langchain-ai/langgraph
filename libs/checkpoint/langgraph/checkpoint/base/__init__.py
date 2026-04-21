from __future__ import annotations

import asyncio
import copy
import dataclasses
import logging
from collections.abc import AsyncIterator, Collection, Iterator, Mapping, Sequence
from typing import (  # noqa: UP035
    Any,
    Generic,
    Literal,
    NamedTuple,
    TypedDict,
    TypeVar,
)

from langchain_core.runnables import RunnableConfig

from langgraph.checkpoint.base.id import uuid6
from langgraph.checkpoint.serde.base import SerializerProtocol, maybe_add_typed_methods
from langgraph.checkpoint.serde.encrypted import EncryptedSerializer
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
from langgraph.checkpoint.serde.types import (
    ERROR,
    INTERRUPT,
    RESUME,
    SCHEDULED,
    ChannelProtocol,
)

V = TypeVar("V", int, float, str)
PendingWrite = tuple[str, str, Any]


@dataclasses.dataclass
class DeltaValue:
    """Returned by DeltaChannel.checkpoint(). Represents one step's writes."""

    delta: list[Any]
    prev_checkpoint_id: (
        str | None
    )  # ID of checkpoint containing previous blob; None = chain root


@dataclasses.dataclass
class DeltaChainValue:
    """Passed to DeltaChannel.from_checkpoint(). Assembled during checkpoint hydration."""

    base: list[Any] | None  # starting accumulated value; None = start from empty
    deltas: list[list[Any]]  # per-step write-sets, ordered oldest → newest


CheckpointHydrationKind = Literal["delta"]


@dataclasses.dataclass(frozen=True)
class IncrementalChannelSpec:
    """Describes a checkpoint field that needs saver-side materialization."""

    name: str
    kind: CheckpointHydrationKind


@dataclasses.dataclass(frozen=True)
class CheckpointHydrationPlan:
    """Lists the checkpoint fields eligible for saver-side materialization."""

    channels: tuple[IncrementalChannelSpec, ...]


logger = logging.getLogger(__name__)
_MISSING_SENTINEL = object()


# Marked as total=False to allow for future expansion.
class CheckpointMetadata(TypedDict, total=False):
    """Metadata associated with a checkpoint."""

    source: Literal["input", "loop", "update", "fork"]
    """The source of the checkpoint.

    - `"input"`: The checkpoint was created from an input to invoke/stream/batch.
    - `"loop"`: The checkpoint was created from inside the pregel loop.
    - `"update"`: The checkpoint was created from a manual state update.
    - `"fork"`: The checkpoint was created as a copy of another checkpoint.
    """
    step: int
    """The step number of the checkpoint.

    `-1` for the first `"input"` checkpoint.
    `0` for the first `"loop"` checkpoint.
    `...` for the `nth` checkpoint afterwards.
    """
    parents: dict[str, str]
    """The IDs of the parent checkpoints.

    Mapping from checkpoint namespace to checkpoint ID.
    """
    run_id: str
    """The ID of the run that created this checkpoint."""


ChannelVersions = dict[str, str | int | float]


class Checkpoint(TypedDict):
    """State snapshot at a given point in time."""

    v: int
    """The version of the checkpoint format. Currently `1`."""
    id: str
    """The ID of the checkpoint.
    
    This is both unique and monotonically increasing, so can be used for sorting
    checkpoints from first to last."""
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
    updated_channels: list[str] | None
    """The channels that were updated in this checkpoint.
    """


def copy_checkpoint(checkpoint: Checkpoint) -> Checkpoint:
    return Checkpoint(
        v=checkpoint["v"],
        ts=checkpoint["ts"],
        id=checkpoint["id"],
        channel_values=checkpoint["channel_values"].copy(),
        channel_versions=checkpoint["channel_versions"].copy(),
        versions_seen={k: v.copy() for k, v in checkpoint["versions_seen"].items()},
        pending_sends=checkpoint.get("pending_sends", []).copy(),
        updated_channels=checkpoint.get("updated_channels", None),
    )


class CheckpointTuple(NamedTuple):
    """A tuple containing a checkpoint and its associated data."""

    config: RunnableConfig
    checkpoint: Checkpoint
    metadata: CheckpointMetadata
    parent_config: RunnableConfig | None = None
    pending_writes: list[PendingWrite] | None = None


class BaseCheckpointSaver(Generic[V]):
    """Base class for creating a graph checkpointer.

    Checkpointers allow LangGraph agents to persist their state
    within and across multiple interactions.

    When a checkpointer is configured, you should pass a `thread_id` in the config when
    invoking the graph:

    ```python
    config = {"configurable": {"thread_id": "my-thread"}}
    graph.invoke(inputs, config)
    ```

    The `thread_id` is the primary key used to store and retrieve checkpoints. Without
    it, the checkpointer cannot save state, resume from interrupts, or enable
    time-travel debugging.

    How you choose ``thread_id`` depends on your use case:

    - **Single-shot workflows**: Use a unique ID (e.g., uuid4) for each run when
        executions are independent.
    - **Conversational memory**: Reuse the same `thread_id` across invocations
        to accumulate state (e.g., chat history) within a conversation.

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
        serde: SerializerProtocol | None = None,
    ) -> None:
        self.serde = maybe_add_typed_methods(serde or self.serde)

    @property
    def config_specs(self) -> list:
        """Define the configuration options for the checkpoint saver.

        Returns:
            list: List of configuration field specs.
        """
        return []

    def get(self, config: RunnableConfig) -> Checkpoint | None:
        """Fetch a checkpoint using the given configuration.

        Args:
            config: Configuration specifying which checkpoint to retrieve.

        Returns:
            The requested checkpoint, or `None` if not found.
        """
        if value := self.get_tuple(config):
            return value.checkpoint

    def get_tuple(self, config: RunnableConfig) -> CheckpointTuple | None:
        """Fetch a checkpoint tuple using the given configuration.

        Args:
            config: Configuration specifying which checkpoint to retrieve.

        Returns:
            The requested checkpoint tuple, or `None` if not found.

        Raises:
            NotImplementedError: Implement this method in your custom checkpoint saver.
        """
        raise NotImplementedError

    def list(
        self,
        config: RunnableConfig | None,
        *,
        filter: dict[str, Any] | None = None,
        before: RunnableConfig | None = None,
        limit: int | None = None,
    ) -> Iterator[CheckpointTuple]:
        """List checkpoints that match the given criteria.

        Args:
            config: Base configuration for filtering checkpoints.
            filter: Additional filtering criteria.
            before: List checkpoints created before this configuration.
            limit: Maximum number of checkpoints to return.

        Returns:
            Iterator of matching checkpoint tuples.

        Raises:
            NotImplementedError: Implement this method in your custom checkpoint saver.
        """
        raise NotImplementedError

    def put(
        self,
        config: RunnableConfig,
        checkpoint: Checkpoint,
        metadata: CheckpointMetadata,
        new_versions: ChannelVersions,
    ) -> RunnableConfig:
        """Store a checkpoint with its configuration and metadata.

        Args:
            config: Configuration for the checkpoint.
            checkpoint: The checkpoint to store.
            metadata: Additional metadata for the checkpoint.
            new_versions: New channel versions as of this write.

        Returns:
            RunnableConfig: Updated configuration after storing the checkpoint.

        Raises:
            NotImplementedError: Implement this method in your custom checkpoint saver.
        """
        raise NotImplementedError

    def put_writes(
        self,
        config: RunnableConfig,
        writes: Sequence[tuple[str, Any]],
        task_id: str,
        task_path: str = "",
    ) -> None:
        """Store intermediate writes linked to a checkpoint.

        Args:
            config: Configuration of the related checkpoint.
            writes: List of writes to store.
            task_id: Identifier for the task creating the writes.
            task_path: Path of the task creating the writes.

        Raises:
            NotImplementedError: Implement this method in your custom checkpoint saver.
        """
        raise NotImplementedError

    def delete_thread(
        self,
        thread_id: str,
    ) -> None:
        """Delete all checkpoints and writes associated with a specific thread ID.

        Args:
            thread_id: The thread ID whose checkpoints should be deleted.
        """
        raise NotImplementedError

    def delete_for_runs(
        self,
        run_ids: Sequence[str],
    ) -> None:
        """Delete all checkpoints and writes associated with the given run IDs.

        Args:
            run_ids: The run IDs whose checkpoints should be deleted.
        """
        raise NotImplementedError

    def copy_thread(
        self,
        source_thread_id: str,
        target_thread_id: str,
    ) -> None:
        """Copy all checkpoints and writes from one thread to another.

        Args:
            source_thread_id: The thread ID to copy from.
            target_thread_id: The thread ID to copy to.
        """
        raise NotImplementedError

    def prune(
        self,
        thread_ids: Sequence[str],
        *,
        strategy: str = "keep_latest",
    ) -> None:
        """Prune checkpoints for the given threads.

        Args:
            thread_ids: The thread IDs to prune.
            strategy: The pruning strategy. `"keep_latest"` retains only the most
                recent checkpoint per namespace. `"delete"` removes all checkpoints.
        """
        raise NotImplementedError

    async def aget(self, config: RunnableConfig) -> Checkpoint | None:
        """Asynchronously fetch a checkpoint using the given configuration.

        Args:
            config: Configuration specifying which checkpoint to retrieve.

        Returns:
            The requested checkpoint, or `None` if not found.
        """
        if value := await self.aget_tuple(config):
            return value.checkpoint

    async def aget_tuple(self, config: RunnableConfig) -> CheckpointTuple | None:
        """Asynchronously fetch a checkpoint tuple using the given configuration.

        Args:
            config: Configuration specifying which checkpoint to retrieve.

        Returns:
            The requested checkpoint tuple, or `None` if not found.

        Raises:
            NotImplementedError: Implement this method in your custom checkpoint saver.
        """
        raise NotImplementedError

    async def alist(
        self,
        config: RunnableConfig | None,
        *,
        filter: dict[str, Any] | None = None,
        before: RunnableConfig | None = None,
        limit: int | None = None,
    ) -> AsyncIterator[CheckpointTuple]:
        """Asynchronously list checkpoints that match the given criteria.

        Args:
            config: Base configuration for filtering checkpoints.
            filter: Additional filtering criteria for metadata.
            before: List checkpoints created before this configuration.
            limit: Maximum number of checkpoints to return.

        Returns:
            Async iterator of matching checkpoint tuples.

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
        new_versions: ChannelVersions,
    ) -> RunnableConfig:
        """Asynchronously store a checkpoint with its configuration and metadata.

        Args:
            config: Configuration for the checkpoint.
            checkpoint: The checkpoint to store.
            metadata: Additional metadata for the checkpoint.
            new_versions: New channel versions as of this write.

        Returns:
            RunnableConfig: Updated configuration after storing the checkpoint.

        Raises:
            NotImplementedError: Implement this method in your custom checkpoint saver.
        """
        raise NotImplementedError

    async def aput_writes(
        self,
        config: RunnableConfig,
        writes: Sequence[tuple[str, Any]],
        task_id: str,
        task_path: str = "",
    ) -> None:
        """Asynchronously store intermediate writes linked to a checkpoint.

        Args:
            config: Configuration of the related checkpoint.
            writes: List of writes to store.
            task_id: Identifier for the task creating the writes.
            task_path: Path of the task creating the writes.

        Raises:
            NotImplementedError: Implement this method in your custom checkpoint saver.
        """
        raise NotImplementedError

    async def adelete_thread(
        self,
        thread_id: str,
    ) -> None:
        """Delete all checkpoints and writes associated with a specific thread ID.

        Args:
            thread_id: The thread ID whose checkpoints should be deleted.
        """
        raise NotImplementedError

    async def adelete_for_runs(
        self,
        run_ids: Sequence[str],
    ) -> None:
        """Asynchronously delete all checkpoints and writes for the given run IDs.

        Args:
            run_ids: The run IDs whose checkpoints should be deleted.
        """
        raise NotImplementedError

    async def acopy_thread(
        self,
        source_thread_id: str,
        target_thread_id: str,
    ) -> None:
        """Asynchronously copy all checkpoints and writes from one thread to another.

        Args:
            source_thread_id: The thread ID to copy from.
            target_thread_id: The thread ID to copy to.
        """
        raise NotImplementedError

    async def aprune(
        self,
        thread_ids: Sequence[str],
        *,
        strategy: str = "keep_latest",
    ) -> None:
        """Asynchronously prune checkpoints for the given threads.

        Args:
            thread_ids: The thread IDs to prune.
            strategy: The pruning strategy. `"keep_latest"` retains only the most
                recent checkpoint per namespace. `"delete"` removes all checkpoints.
        """
        raise NotImplementedError

    def materialize_checkpoint(
        self,
        config: RunnableConfig,
        checkpoint: Checkpoint,
        plan: CheckpointHydrationPlan | None = None,
    ) -> Checkpoint:
        """Materialize any saver-managed incremental values in a checkpoint."""
        if plan is None or not plan.channels:
            return checkpoint

        thread_id = str(config["configurable"]["thread_id"])
        checkpoint_ns = config["configurable"].get("checkpoint_ns", "")
        current_checkpoint_id = checkpoint.get("id")
        assembled: dict[str, Any] = {}

        for spec in plan.channels:
            value = checkpoint["channel_values"].get(spec.name)
            if spec.kind != "delta" or not isinstance(value, DeltaValue):
                continue

            assembled_value = self._materialize_delta_channel(
                thread_id=thread_id,
                checkpoint_ns=checkpoint_ns,
                current_checkpoint_id=current_checkpoint_id,
                channel=spec.name,
                value=value,
            )
            if assembled_value is not None:
                assembled[spec.name] = assembled_value

        if not assembled:
            return checkpoint

        return {
            **checkpoint,
            "channel_values": {**checkpoint["channel_values"], **assembled},
        }

    def materialize_checkpoint_tuple(
        self,
        value: CheckpointTuple,
        plan: CheckpointHydrationPlan | None = None,
    ) -> CheckpointTuple:
        """Materialize incremental values for a single checkpoint tuple."""
        checkpoint = self.materialize_checkpoint(value.config, value.checkpoint, plan)
        if checkpoint is value.checkpoint:
            return value
        return value._replace(checkpoint=checkpoint)

    def materialize_checkpoint_tuples(
        self,
        values: Sequence[CheckpointTuple],
        plan: CheckpointHydrationPlan | None = None,
    ) -> Sequence[CheckpointTuple]:
        """Materialize incremental values for a batch of checkpoint tuples."""
        return [self.materialize_checkpoint_tuple(value, plan) for value in values]

    async def amaterialize_checkpoint(
        self,
        config: RunnableConfig,
        checkpoint: Checkpoint,
        plan: CheckpointHydrationPlan | None = None,
    ) -> Checkpoint:
        """Async materialization hook for saver-managed incremental values."""
        if plan is None or not plan.channels:
            return checkpoint

        thread_id = str(config["configurable"]["thread_id"])
        checkpoint_ns = config["configurable"].get("checkpoint_ns", "")
        current_checkpoint_id = checkpoint.get("id")

        targets = [
            (spec, checkpoint["channel_values"][spec.name])
            for spec in plan.channels
            if spec.kind == "delta"
            and isinstance(checkpoint["channel_values"].get(spec.name), DeltaValue)
        ]
        if not targets:
            return checkpoint

        # Walks for independent channels can run concurrently — each has its own chain.
        results = await asyncio.gather(
            *(
                self._amaterialize_delta_channel(
                    thread_id=thread_id,
                    checkpoint_ns=checkpoint_ns,
                    current_checkpoint_id=current_checkpoint_id,
                    channel=spec.name,
                    value=value,
                )
                for spec, value in targets
            )
        )
        assembled = {
            spec.name: result
            for (spec, _), result in zip(targets, results, strict=True)
            if result is not None
        }

        if not assembled:
            return checkpoint

        return {
            **checkpoint,
            "channel_values": {**checkpoint["channel_values"], **assembled},
        }

    async def amaterialize_checkpoint_tuple(
        self,
        value: CheckpointTuple,
        plan: CheckpointHydrationPlan | None = None,
    ) -> CheckpointTuple:
        """Async materialization hook for a single checkpoint tuple."""
        checkpoint = await self.amaterialize_checkpoint(
            value.config, value.checkpoint, plan
        )
        if checkpoint is value.checkpoint:
            return value
        return value._replace(checkpoint=checkpoint)

    async def amaterialize_checkpoint_tuples(
        self,
        values: Sequence[CheckpointTuple],
        plan: CheckpointHydrationPlan | None = None,
    ) -> Sequence[CheckpointTuple]:
        """Async materialization hook for a batch of checkpoint tuples."""
        return [
            await self.amaterialize_checkpoint_tuple(value, plan) for value in values
        ]

    def get_channel_blob(
        self,
        thread_id: str,
        checkpoint_ns: str,
        checkpoint_id: str,
        channel: str,
    ) -> Any:
        """Look up a single channel blob by checkpoint ID + channel name.

        Returns NotImplemented if this saver does not support efficient
        per-channel-version blob lookup. The pregel layer will fall back to
        get_tuple() traversal in that case.

        Savers with a dedicated blob store (InMemorySaver, PostgresSaver)
        should override this for O(1) performance.
        """
        return NotImplemented

    async def aget_channel_blob(
        self,
        thread_id: str,
        checkpoint_ns: str,
        checkpoint_id: str,
        channel: str,
    ) -> Any:
        """Look up a single channel blob by checkpoint ID + channel name (async).

        Returns NotImplemented if this saver does not support efficient
        per-channel-version blob lookup. The pregel layer will fall back to
        aget_tuple() traversal in that case.

        Savers with a dedicated blob store (InMemorySaver, PostgresSaver)
        should override this for O(1) performance.
        """
        return NotImplemented

    def get_next_version(self, current: V | None, channel: None) -> V:
        """Generate the next version ID for a channel.

        Default is to use integer versions, incrementing by `1`.

        If you override, you can use `str`/`int`/`float` versions, as long as they are monotonically increasing.

        Args:
            current: The current version identifier (`int`, `float`, or `str`).
            channel: Deprecated argument, kept for backwards compatibility.

        Returns:
            V: The next version identifier, which must be increasing.
        """
        if isinstance(current, str):
            raise NotImplementedError
        elif current is None:
            return 1
        else:
            return current + 1

    def with_allowlist(
        self, extra_allowlist: Collection[tuple[str, ...]]
    ) -> BaseCheckpointSaver[V]:
        """Return a shallow clone with a derived msgpack allowlist."""
        serde = _with_msgpack_allowlist(self.serde, extra_allowlist)
        if serde is self.serde:
            return self
        clone = copy.copy(self)
        clone.serde = maybe_add_typed_methods(serde)
        return clone

    def _materialize_delta_channel(
        self,
        *,
        thread_id: str,
        checkpoint_ns: str,
        current_checkpoint_id: str | None,
        channel: str,
        value: DeltaValue,
    ) -> DeltaChainValue | None:
        chain_deltas: list[list[Any]] = []
        base: list[Any] | None = None
        cursor: DeltaValue = value
        visited: set[str] = {current_checkpoint_id} if current_checkpoint_id else set()

        while True:
            chain_deltas.append(cursor.delta)
            prev_id = cursor.prev_checkpoint_id
            if prev_id is None:
                break
            if prev_id in visited:
                logger.warning(
                    "DeltaChannel chain cycle at checkpoint %r for channel %r; breaking",
                    prev_id,
                    channel,
                )
                break
            visited.add(prev_id)

            blob = self.get_channel_blob(thread_id, checkpoint_ns, prev_id, channel)
            if blob is not NotImplemented:
                if isinstance(blob, DeltaValue):
                    cursor = blob
                    continue
                base = blob
                break

            parent_config: RunnableConfig = {
                "configurable": {
                    "thread_id": thread_id,
                    "checkpoint_ns": checkpoint_ns,
                    "checkpoint_id": prev_id,
                }
            }
            parent_tuple = self.get_tuple(parent_config)
            if parent_tuple is None:
                logger.warning(
                    "DeltaChannel chain broken: checkpoint %r not found for channel %r",
                    prev_id,
                    channel,
                )
                break
            prev_val = parent_tuple.checkpoint["channel_values"].get(
                channel, _MISSING_SENTINEL
            )
            if prev_val is _MISSING_SENTINEL:
                break
            if isinstance(prev_val, DeltaValue):
                cursor = prev_val
            else:
                base = prev_val
                break

        chain_deltas.reverse()
        return DeltaChainValue(base=base, deltas=chain_deltas)

    async def _amaterialize_delta_channel(
        self,
        *,
        thread_id: str,
        checkpoint_ns: str,
        current_checkpoint_id: str | None,
        channel: str,
        value: DeltaValue,
    ) -> DeltaChainValue | None:
        chain_deltas: list[list[Any]] = []
        base: list[Any] | None = None
        cursor: DeltaValue = value
        visited: set[str] = {current_checkpoint_id} if current_checkpoint_id else set()

        while True:
            chain_deltas.append(cursor.delta)
            prev_id = cursor.prev_checkpoint_id
            if prev_id is None:
                break
            if prev_id in visited:
                logger.warning(
                    "DeltaChannel chain cycle at checkpoint %r for channel %r; breaking",
                    prev_id,
                    channel,
                )
                break
            visited.add(prev_id)

            blob = await self.aget_channel_blob(
                thread_id, checkpoint_ns, prev_id, channel
            )
            if blob is not NotImplemented:
                if isinstance(blob, DeltaValue):
                    cursor = blob
                    continue
                base = blob
                break

            parent_config: RunnableConfig = {
                "configurable": {
                    "thread_id": thread_id,
                    "checkpoint_ns": checkpoint_ns,
                    "checkpoint_id": prev_id,
                }
            }
            parent_tuple = await self.aget_tuple(parent_config)
            if parent_tuple is None:
                logger.warning(
                    "DeltaChannel chain broken: checkpoint %r not found for channel %r",
                    prev_id,
                    channel,
                )
                break
            prev_val = parent_tuple.checkpoint["channel_values"].get(
                channel, _MISSING_SENTINEL
            )
            if prev_val is _MISSING_SENTINEL:
                break
            if isinstance(prev_val, DeltaValue):
                cursor = prev_val
            else:
                base = prev_val
                break

        chain_deltas.reverse()
        return DeltaChainValue(base=base, deltas=chain_deltas)


def _with_msgpack_allowlist(
    serde: SerializerProtocol, extra_allowlist: Collection[tuple[str, ...]]
) -> SerializerProtocol:
    if isinstance(serde, JsonPlusSerializer):
        return serde.with_msgpack_allowlist(extra_allowlist)
    if isinstance(serde, EncryptedSerializer):
        inner = serde.serde
        if isinstance(inner, JsonPlusSerializer):
            updated_inner = inner.with_msgpack_allowlist(extra_allowlist)
            if updated_inner is inner:
                return serde
            return EncryptedSerializer(serde.cipher, updated_inner)
    logger.warning(
        "Serializer %s does not support msgpack allowlist. "
        "Strict msgpack deserialization will not be enforced.",
        type(serde).__name__,
    )
    return serde


class EmptyChannelError(Exception):
    """Raised when attempting to get the value of a channel that hasn't been updated
    for the first time yet."""

    pass


def get_checkpoint_id(config: RunnableConfig) -> str | None:
    """Get checkpoint ID."""
    return config["configurable"].get("checkpoint_id")


def get_checkpoint_metadata(
    config: RunnableConfig, metadata: CheckpointMetadata
) -> CheckpointMetadata:
    """Get checkpoint metadata in a backwards-compatible manner."""
    metadata = {
        k: v.replace("\u0000", "") if isinstance(v, str) else v
        for k, v in metadata.items()
    }
    for obj in (config.get("metadata"), config.get("configurable")):
        if not obj:
            continue
        for key, v in obj.items():
            if key in metadata or key in EXCLUDED_METADATA_KEYS or key.startswith("__"):
                continue
            elif isinstance(v, str):
                metadata[key] = v.replace("\u0000", "")
            elif isinstance(v, (int, bool, float)):
                metadata[key] = v
    return metadata


def get_serializable_checkpoint_metadata(
    config: RunnableConfig, metadata: CheckpointMetadata
) -> CheckpointMetadata:
    """Get checkpoint metadata in a backwards-compatible manner."""
    checkpoint_metadata = get_checkpoint_metadata(config, metadata)
    if "writes" in checkpoint_metadata:
        checkpoint_metadata.pop("writes")
    return checkpoint_metadata


"""
Mapping from error type to error index.
Regular writes just map to their index in the list of writes being saved.
Special writes (e.g. errors) map to negative indices, to avoid those writes from
conflicting with regular writes.
Each Checkpointer implementation should use this mapping in put_writes.
"""
WRITES_IDX_MAP = {ERROR: -1, SCHEDULED: -2, INTERRUPT: -3, RESUME: -4}

EXCLUDED_METADATA_KEYS = {
    "thread_id",
    "checkpoint_id",
    "checkpoint_ns",
    "checkpoint_map",
    "langgraph_step",
    "langgraph_node",
    "langgraph_triggers",
    "langgraph_path",
    "langgraph_checkpoint_ns",
}

# --- below are deprecated utilities used by past versions of LangGraph ---

LATEST_VERSION = 2


def empty_checkpoint() -> Checkpoint:
    from datetime import datetime, timezone

    return Checkpoint(
        v=LATEST_VERSION,
        id=str(uuid6(clock_seq=-2)),
        ts=datetime.now(timezone.utc).isoformat(),
        channel_values={},
        channel_versions={},
        versions_seen={},
        pending_sends=[],
        updated_channels=None,
    )


def create_checkpoint(
    checkpoint: Checkpoint,
    channels: Mapping[str, ChannelProtocol] | None,
    step: int,
    *,
    id: str | None = None,
) -> Checkpoint:
    """Create a checkpoint for the given channels."""
    from datetime import datetime, timezone

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
        v=LATEST_VERSION,
        ts=ts,
        id=id or str(uuid6(clock_seq=step)),
        channel_values=values,
        channel_versions=checkpoint["channel_versions"],
        versions_seen=checkpoint["versions_seen"],
        pending_sends=checkpoint.get("pending_sends", []),
        updated_channels=None,
    )
