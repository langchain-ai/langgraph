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
from typing_extensions import NotRequired

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


logger = logging.getLogger(__name__)


def _parse_supersteps_since_last_snapshot_by_channel(
    metadata: Mapping[str, Any],
    channels: Sequence[str],
) -> dict[str, int]:
    """Per-channel supersteps-since-last-snapshot, parsed from metadata.

    Reads `metadata.counters_since_delta_snapshot[ch]`, a `(updates,
    supersteps)` pair, and returns `{ch: supersteps}` for channels whose
    supersteps count is positive. Channels with no counter entry (just
    snapshotted, or never written) are omitted — their seed, if any, is the
    target checkpoint itself and needs no ancestor walk.

    Used to drive the ancestor-walk depth in `get_delta_channel_history`:
    a channel's seed snapshot sits exactly `supersteps` hops back along the
    parent chain.
    """
    counters = metadata.get("counters_since_delta_snapshot") or {}
    result: dict[str, int] = {}
    for ch in channels:
        entry = counters.get(ch)
        if not isinstance(entry, (list, tuple)) or len(entry) < 2:
            continue
        try:
            supersteps = int(entry[1])
        except (TypeError, ValueError):
            continue
        if supersteps > 0:
            result[ch] = supersteps
    return result


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
    counters_since_delta_snapshot: dict[str, tuple[int, int]]
    """Per-channel counters since the last `_DeltaSnapshot` was written.

    !!! warning "Beta"

        This metadata field backs `DeltaChannel` (beta). The key name and
        contents may change while the delta-channel design stabilizes.

    Maps channel name -> `(updates, supersteps)`:

    - index 0 (`updates`): number of supersteps that wrote to this channel
      since its last snapshot blob.
    - index 1 (`supersteps`): total supersteps elapsed since this channel's
      last snapshot, regardless of whether the channel was written.

    A snapshot fires when EITHER `updates >= ch.snapshot_frequency` OR
    `supersteps >= DELTA_MAX_SUPERSTEPS_SINCE_SNAPSHOT` (system-wide bound,
    default 5000, env `LANGGRAPH_DELTA_MAX_SUPERSTEPS_SINCE_SNAPSHOT`).
    The supersteps bound prevents unbounded ancestor walks on threads where
    a delta channel exists but is no longer being updated.

    Absent on threads that don't use delta channels. Persisted as a
    2-element list in JSON (no native tuple).
    """


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


class DeltaChannelHistory(TypedDict):
    """Per-channel result entry from `BaseCheckpointSaver.get_delta_channel_history`.

    !!! warning "Beta"

        Part of the `DeltaChannel` support surface; in beta. Field names and
        semantics may change.

    Storage-level view of what one channel contributed across the ancestor
    chain of a target checkpoint:

    * `writes` — on-path deltas oldest→newest as `PendingWrite` tuples.
      Always present; possibly empty. Already filtered to one channel.
      Writes stored at the target checkpoint itself are pending for the
      next super-step and are excluded.
    * `seed` — the stored value at the nearest ancestor whose
      `channel_values[ch]` is populated. Omitted if the walk reached the
      root without finding any stored value (consumer treats absence as
      "start empty"). Typically a `_DeltaSnapshot` for delta channels with
      finite snapshot frequency, or a plain value for threads migrated
      from a pre-delta channel type.
    """

    writes: list[PendingWrite]
    seed: NotRequired[Any]


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

        !!! warning "DeltaChannel"

            Deleting a run that produced ancestor `checkpoint_writes` — or
            the only `_DeltaSnapshot` blob — for a still-live thread will
            break reconstruction of any `DeltaChannel` whose history
            depended on those rows. See the `DeltaChannel` note on `prune`
            for safe-recovery strategies.
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

        !!! warning "DeltaChannel"

            Implementations must copy the **complete** parent chain (all
            ancestor checkpoints and their `checkpoint_writes`) — copying
            only the head checkpoint will leave the target thread with
            `DeltaChannel` state that cannot be reconstructed (no path back
            to a `_DeltaSnapshot` ancestor). Equivalently, the copy must
            include enough ancestors that every `DeltaChannel`-backed key
            has either a `_DeltaSnapshot` in `channel_values` somewhere in
            the chain, or a complete write history back to the chain root.
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

        !!! warning "DeltaChannel"

            Custom implementations must be `DeltaChannel`-aware. `DeltaChannel`
            stores only a sentinel in `channel_values` for non-snapshot steps;
            reconstruction walks the parent chain via
            `get_delta_channel_history`, accumulating rows from
            `checkpoint_writes` until it reaches an ancestor whose
            `channel_values` contains a `_DeltaSnapshot` blob (written every
            `snapshot_frequency` updates).

            A naive `"keep_latest"` that drops intermediate checkpoints and
            their writes can sever that chain: the surviving "latest"
            checkpoint is rarely a snapshot point itself, so its delta
            channels would silently reconstruct as empty (no error raised —
            `get_delta_channel_history` simply returns no `seed`). Safe
            options when the graph uses `DeltaChannel`:

            * Walk back from each kept checkpoint and preserve every
              ancestor (plus its `checkpoint_writes`) up to the nearest one
              whose `channel_values` already contains a `_DeltaSnapshot` for
              every `DeltaChannel`-backed key.
            * Force a fresh snapshot on the kept checkpoint before deleting
              ancestors — rewrite `channel_values[k] = _DeltaSnapshot(value)`
              for each delta channel `k` (resolving `value` via the existing
              ancestor walk first), then prune.
            * Skip pruning threads whose graph uses `DeltaChannel` until one
              of the above is implemented.
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

        !!! warning "DeltaChannel"

            See `delete_for_runs` — deleting rows a still-live thread's
            `DeltaChannel` reconstruction depends on (writes between the
            head and its nearest `_DeltaSnapshot` ancestor) will silently
            corrupt that channel's state.
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

        !!! warning "DeltaChannel"

            See `copy_thread` — the copy must carry the complete parent
            chain (or at least back to a `_DeltaSnapshot` ancestor for every
            `DeltaChannel`) so the target thread can reconstruct delta
            state.
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

        !!! warning "DeltaChannel"

            See `prune` for the full `DeltaChannel` caveat. In short:
            `"keep_latest"` must not drop ancestor checkpoints / writes that
            sit between the kept checkpoint and the nearest `_DeltaSnapshot`
            ancestor, or delta channels will silently reconstruct as empty.
        """
        raise NotImplementedError

    def get_delta_channel_history(
        self, *, config: RunnableConfig, channels: Sequence[str]
    ) -> Mapping[str, DeltaChannelHistory]:
        """Walk the parent chain returning per-channel writes + seed.

        !!! warning "Beta"

            This method is part of the `DeltaChannel` support surface and is
            in beta. The signature, return shape (`DeltaChannelHistory`), and
            interaction with `_DeltaSnapshot` blobs may change. Override at
            your own risk; the default implementation will continue to work
            against the public `BaseCheckpointSaver` contract.

        For each requested channel, walks ancestors of the checkpoint
        identified by `config` (following `parent_config`) and accumulates
        `pending_writes` for that channel. The walk terminates per-channel
        at the nearest ancestor whose `channel_values[ch]` is populated;
        that value is returned as `seed`. If the walk reaches the root
        without finding a stored value, `seed` is omitted from that
        channel's entry — the consumer treats the absence as "start
        empty."

        Walks the **parent chain** (not `list(before=...)`): for forked
        threads, only on-path ancestors contribute.

        The default implementation walks `get_tuple` + `parent_config`
        once for all channels — each ancestor visited once, not once per
        channel. Savers with direct storage access (`InMemorySaver`,
        `PostgresSaver`) override for performance; the return contract is
        fixed here.

        Args:
            config: Configuration identifying the target checkpoint.
            channels: Channel names to walk for. Empty → empty mapping.

        Returns:
            Per-channel `DeltaChannelHistory` for every name in `channels`.
        """
        if not channels:
            return {}
        channels = list(channels)
        target_tuple = self.get_tuple(config)
        if target_tuple is None:
            return {ch: {"writes": []} for ch in channels}
        supersteps_by_ch = _parse_supersteps_since_last_snapshot_by_channel(
            target_tuple.metadata or {}, channels
        )
        max_supersteps = max(supersteps_by_ch.values(), default=0)

        chain: list[CheckpointTuple] = []
        has_reached_root = False
        cursor_config: RunnableConfig | None = target_tuple.parent_config
        while cursor_config is not None and len(chain) < max_supersteps:
            tup = self.get_tuple(cursor_config)
            if tup is None:
                break
            chain.append(tup)
            if tup.parent_config is None:
                has_reached_root = True
                break
            cursor_config = tup.parent_config
        return self._assemble_default_delta_history(
            channels, supersteps_by_ch, chain, has_reached_root, config
        )

    async def aget_delta_channel_history(
        self, *, config: RunnableConfig, channels: Sequence[str]
    ) -> Mapping[str, DeltaChannelHistory]:
        """Async version of `get_delta_channel_history`.

        !!! warning "Beta"

            This method is part of the `DeltaChannel` support surface and is
            in beta. See `get_delta_channel_history` for caveats.
        """
        if not channels:
            return {}
        channels = list(channels)
        target_tuple = await self.aget_tuple(config)
        if target_tuple is None:
            return {ch: {"writes": []} for ch in channels}
        supersteps_by_ch = _parse_supersteps_since_last_snapshot_by_channel(
            target_tuple.metadata or {}, channels
        )
        max_supersteps = max(supersteps_by_ch.values(), default=0)

        chain: list[CheckpointTuple] = []
        has_reached_root = False
        cursor_config: RunnableConfig | None = target_tuple.parent_config
        while cursor_config is not None and len(chain) < max_supersteps:
            tup = await self.aget_tuple(cursor_config)
            if tup is None:
                break
            chain.append(tup)
            if tup.parent_config is None:
                has_reached_root = True
                break
            cursor_config = tup.parent_config
        return self._assemble_default_delta_history(
            channels, supersteps_by_ch, chain, has_reached_root, config
        )

    def _assemble_default_delta_history(
        self,
        channels: Sequence[str],
        supersteps_by_ch: Mapping[str, int],
        chain: Sequence[CheckpointTuple],
        has_reached_root: bool,
        config: RunnableConfig,
    ) -> dict[str, DeltaChannelHistory]:
        """Slice the walked parent chain into per-channel histories.

        For each channel the seed snapshot sits `supersteps` hops back along
        `chain` (newest first): the seed checkpoint is `chain[supersteps - 1]`
        and its `channel_values[ch]` is the seed value. The seed checkpoint's
        own writes are replayed on top of a `_DeltaSnapshot` seed but skipped
        for a migrated plain-value seed (the legacy non-delta blob already
        incorporates them).

        When the chain is shorter than `supersteps` but the root was reached,
        the persisted chain is "compressed" relative to the logical superstep
        count (`durability="exit"`, or a thread that never snapshotted). The
        seed candidate is then the oldest checkpoint (`chain[-1]`); if its
        `channel_values[ch]` is absent the channel has no seed and replays
        the full chain on an empty baseline.
        """
        # Imported lazily to avoid a hard checkpoint→serde-types coupling at
        # module import; only the delta surface needs the runtime check.
        from langgraph.checkpoint.serde.types import _DeltaSnapshot

        thread_id = config["configurable"].get("thread_id")
        result: dict[str, DeltaChannelHistory] = {}
        for ch in channels:
            entry: DeltaChannelHistory = {"writes": []}
            bound = supersteps_by_ch.get(ch, 0)
            if bound <= 0:
                result[ch] = entry
                continue
            if len(chain) >= bound:
                seed_depth = bound
            elif has_reached_root:
                seed_depth = len(chain)
            else:
                logger.warning(
                    "cannot find seed snapshot for delta channel "
                    "(thread_id=%s, channel=%s)",
                    thread_id,
                    ch,
                )
                result[ch] = entry
                continue
            if seed_depth <= 0:
                result[ch] = entry
                continue
            chain_slice: Sequence[CheckpointTuple] = chain[:seed_depth]
            seed_tuple: CheckpointTuple | None = chain[seed_depth - 1]

            skip_seed_checkpoint_writes = False
            channel_values = seed_tuple.checkpoint["channel_values"]
            if ch in channel_values:
                seed_value = channel_values[ch]
                entry["seed"] = seed_value
                skip_seed_checkpoint_writes = not isinstance(seed_value, _DeltaSnapshot)
            else:
                # No stored value at the oldest checkpoint → empty baseline,
                # so the chain's own writes are all replayed from empty.
                seed_tuple = None

            collected: list[PendingWrite] = []
            # Chain is newest→oldest; replay oldest→newest.
            for tup in reversed(list(chain_slice)):
                if skip_seed_checkpoint_writes and tup is seed_tuple:
                    continue
                for write in tup.pending_writes or []:
                    if write[1] == ch:
                        collected.append(write)
            entry["writes"] = collected
            result[ch] = entry
        return result

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
