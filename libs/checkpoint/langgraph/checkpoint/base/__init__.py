from __future__ import annotations

import copy
import logging
from collections.abc import AsyncIterator, Collection, Iterator, Mapping, Sequence
from typing import (
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
    DELTA_SENTINEL as DELTA_SENTINEL,
)
from langgraph.checkpoint.serde.types import (
    ERROR,
    INTERRUPT,
    RESUME,
    SCHEDULED,
    ChannelProtocol,
    _DeltaSnapshot,
)

V = TypeVar("V", int, float, str)
PendingWrite = tuple[str, str, Any]


logger = logging.getLogger(__name__)


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


class _ChannelWritesHistory(NamedTuple):
    """Result of `BaseCheckpointSaver._get_channel_writes_history`.

    Storage-level view of what one channel wrote across the ancestor chain
    of a target checkpoint:

      * `seed` — the nearest ancestor's stored blob value for this channel,
        or `DELTA_SENTINEL` if the walk reached the root without finding a
        stored value. A non-sentinel seed typically indicates a pre-delta
        snapshot preserved across a channel-type migration (e.g.
        `BinaryOperatorAggregate` storage extended under `DeltaChannel`).
      * `writes` — on-path deltas oldest→newest, one `PendingWrite` per
        step that wrote to this channel. Writes stored at the target
        checkpoint itself are pending for the next super-step and are
        excluded.

    Experimental: method surface may change; the NamedTuple shape is the
    contract.
    """

    seed: Any
    writes: list[PendingWrite]


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

    def _get_tuple_raw(self, config: RunnableConfig) -> CheckpointTuple | None:
        """Pure storage read used by `_get_channel_writes_history`.

        Must return the same value as `get_tuple` but must NOT trigger channel
        reconstruction (i.e., must not call `channels_from_checkpoint`). The
        default implementation delegates to `get_tuple`, which is correct for
        savers whose `get_tuple` is a pure storage query (the common case).

        Override this if your saver performs channel hydration inside `get_tuple`.
        Doing so structurally breaks the otherwise-possible cycle:
            _get_channel_writes_history -> _get_tuple_raw -> get_tuple
                                        -> channels_from_checkpoint
                                        -> _get_channel_writes_history (cycle!)
        """
        return self.get_tuple(config)

    async def _aget_tuple_raw(self, config: RunnableConfig) -> CheckpointTuple | None:
        """Async version of `_get_tuple_raw`. See docstring there."""
        return await self.aget_tuple(config)

    def _get_channel_writes_history(
        self, config: RunnableConfig, channel: str
    ) -> _ChannelWritesHistory:
        """**Experimental.** Query one channel's writes along the parent chain.

        Storage-level query, not channel semantics: returns `(seed, writes)`
        reflecting what storage knows about a single channel across the
        ancestor chain of the target checkpoint identified by `config`.

        * `writes` — on-path deltas oldest→newest as `PendingWrite` tuples.
          Writes stored at the target `checkpoint_id` itself are pending
          for the next super-step and are excluded.
        * `seed` — the nearest ancestor's stored blob value for this
          channel; `DELTA_SENTINEL` if the walk reached the root without
          finding a stored value. A non-sentinel seed typically indicates
          a pre-delta snapshot preserved across a channel-type migration.

        Walks the **parent chain** (not `list(before=...)`): for forked
        threads, only on-path ancestors contribute.

        Reference implementation walks `get_tuple` + `parent_config`,
        inspecting each ancestor's `channel_values[channel]` for the seed
        terminator. Savers with direct storage access (`InMemorySaver`,
        `PostgresSaver`) override for performance; the return contract is
        fixed here.

        Underscore-prefixed because the method surface is experimental.
        """
        collected: list[PendingWrite] = []  # newest first; reversed at the end
        target_tuple = self._get_tuple_raw(config)
        cursor_config: RunnableConfig | None = (
            target_tuple.parent_config if target_tuple else None
        )
        while cursor_config is not None:
            tup = self._get_tuple_raw(cursor_config)
            if tup is None:
                break
            # Collect this ancestor's writes FIRST — they encode the
            # transition from this ancestor's state to its child's, so
            # they must be included whether or not this ancestor is the
            # seed terminator.
            if tup.pending_writes:
                # Within a superstep, pending_writes are oldest→newest;
                # reverse to scan newest-first.
                for write in reversed(tup.pending_writes):
                    if write[1] != channel:
                        continue
                    collected.append(write)
            # Seed terminator: any non-sentinel blob on an ancestor
            # establishes the reconstruction base. Stop here.
            ancestor_value = tup.checkpoint["channel_values"].get(channel)
            if ancestor_value is not None and ancestor_value is not DELTA_SENTINEL:
                collected.reverse()
                return _ChannelWritesHistory(seed=ancestor_value, writes=collected)
            cursor_config = tup.parent_config
        collected.reverse()
        return _ChannelWritesHistory(seed=DELTA_SENTINEL, writes=collected)

    async def _aget_channel_writes_history(
        self, config: RunnableConfig, channel: str
    ) -> _ChannelWritesHistory:
        """Async version of `_get_channel_writes_history`. See docstring there."""
        collected: list[PendingWrite] = []
        target_tuple = await self._aget_tuple_raw(config)
        cursor_config: RunnableConfig | None = (
            target_tuple.parent_config if target_tuple else None
        )
        while cursor_config is not None:
            tup = await self._aget_tuple_raw(cursor_config)
            if tup is None:
                break
            if tup.pending_writes:
                for write in reversed(tup.pending_writes):
                    if write[1] != channel:
                        continue
                    collected.append(write)
            ancestor_value = tup.checkpoint["channel_values"].get(channel)
            if ancestor_value is not None and ancestor_value is not DELTA_SENTINEL:
                collected.reverse()
                return _ChannelWritesHistory(seed=ancestor_value, writes=collected)
            cursor_config = tup.parent_config
        collected.reverse()
        return _ChannelWritesHistory(seed=DELTA_SENTINEL, writes=collected)

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
