from __future__ import annotations

from collections.abc import (
    AsyncIterator,
    Awaitable,
    Callable,
    Collection,
    Iterator,
    Mapping,
    Sequence,
)
from dataclasses import dataclass
from typing import Any, Generic, Literal, TypeVar

from langchain_core.runnables import RunnableConfig

from langgraph.checkpoint.base import (
    BaseCheckpointSaver,
    ChannelVersions,
    Checkpoint,
    CheckpointMetadata,
    CheckpointTuple,
    DeltaChannelHistory,
)

V = TypeVar("V", int, float, str)
CheckpointWriteKind = Literal["checkpoint", "writes"]


@dataclass(frozen=True)
class CheckpointWrite:
    """A pending checkpoint persistence operation passed to a guard."""

    kind: CheckpointWriteKind
    """The write path being guarded."""

    config: RunnableConfig
    """Runnable config associated with the write."""

    checkpoint: Checkpoint | None = None
    """Checkpoint payload for `put`/`aput` writes."""

    metadata: CheckpointMetadata | None = None
    """Checkpoint metadata for `put`/`aput` writes."""

    new_versions: ChannelVersions | None = None
    """Channel versions for `put`/`aput` writes."""

    writes: Sequence[tuple[str, Any]] | None = None
    """Pending writes for `put_writes`/`aput_writes` calls."""

    task_id: str | None = None
    """Task id for pending writes."""

    task_path: str = ""
    """Task path for pending writes."""


CheckpointWriteGuard = Callable[[CheckpointWrite], None]
AsyncCheckpointWriteGuard = Callable[[CheckpointWrite], Awaitable[None]]


class GuardedCheckpointSaver(BaseCheckpointSaver[V], Generic[V]):
    """Wrap a checkpoint saver and run a policy before persistence writes.

    `GuardedCheckpointSaver` is intentionally policy-agnostic: the guard can
    call an OWASP ASI06 scanner, enforce schema invariants, append an audit
    event, or raise to reject/quarantine a write before it reaches durable
    storage.
    """

    def __init__(
        self,
        inner: BaseCheckpointSaver[V],
        guard: CheckpointWriteGuard,
        *,
        aguard: AsyncCheckpointWriteGuard | None = None,
    ) -> None:
        super().__init__(serde=inner.serde)
        self.inner = inner
        self.guard = guard
        self.aguard = aguard

    @property
    def config_specs(self) -> list:
        return self.inner.config_specs

    def get_tuple(self, config: RunnableConfig) -> CheckpointTuple | None:
        return self.inner.get_tuple(config)

    def list(
        self,
        config: RunnableConfig | None,
        *,
        filter: dict[str, Any] | None = None,
        before: RunnableConfig | None = None,
        limit: int | None = None,
    ) -> Iterator[CheckpointTuple]:
        return self.inner.list(config, filter=filter, before=before, limit=limit)

    def put(
        self,
        config: RunnableConfig,
        checkpoint: Checkpoint,
        metadata: CheckpointMetadata,
        new_versions: ChannelVersions,
    ) -> RunnableConfig:
        self.guard(
            CheckpointWrite(
                kind="checkpoint",
                config=config,
                checkpoint=checkpoint,
                metadata=metadata,
                new_versions=new_versions,
            )
        )
        return self.inner.put(config, checkpoint, metadata, new_versions)

    def put_writes(
        self,
        config: RunnableConfig,
        writes: Sequence[tuple[str, Any]],
        task_id: str,
        task_path: str = "",
    ) -> None:
        self.guard(
            CheckpointWrite(
                kind="writes",
                config=config,
                writes=writes,
                task_id=task_id,
                task_path=task_path,
            )
        )
        return self.inner.put_writes(config, writes, task_id, task_path)

    def delete_thread(self, thread_id: str) -> None:
        return self.inner.delete_thread(thread_id)

    def delete_for_runs(self, run_ids: Sequence[str]) -> None:
        return self.inner.delete_for_runs(run_ids)

    def copy_thread(self, source_thread_id: str, target_thread_id: str) -> None:
        return self.inner.copy_thread(source_thread_id, target_thread_id)

    def prune(
        self,
        thread_ids: Sequence[str],
        *,
        strategy: str = "keep_latest",
    ) -> None:
        return self.inner.prune(thread_ids, strategy=strategy)

    async def aget_tuple(self, config: RunnableConfig) -> CheckpointTuple | None:
        return await self.inner.aget_tuple(config)

    async def alist(
        self,
        config: RunnableConfig | None,
        *,
        filter: dict[str, Any] | None = None,
        before: RunnableConfig | None = None,
        limit: int | None = None,
    ) -> AsyncIterator[CheckpointTuple]:
        async for item in self.inner.alist(
            config, filter=filter, before=before, limit=limit
        ):
            yield item

    async def aput(
        self,
        config: RunnableConfig,
        checkpoint: Checkpoint,
        metadata: CheckpointMetadata,
        new_versions: ChannelVersions,
    ) -> RunnableConfig:
        write = CheckpointWrite(
            kind="checkpoint",
            config=config,
            checkpoint=checkpoint,
            metadata=metadata,
            new_versions=new_versions,
        )
        if self.aguard is not None:
            await self.aguard(write)
        else:
            self.guard(write)
        return await self.inner.aput(config, checkpoint, metadata, new_versions)

    async def aput_writes(
        self,
        config: RunnableConfig,
        writes: Sequence[tuple[str, Any]],
        task_id: str,
        task_path: str = "",
    ) -> None:
        write = CheckpointWrite(
            kind="writes",
            config=config,
            writes=writes,
            task_id=task_id,
            task_path=task_path,
        )
        if self.aguard is not None:
            await self.aguard(write)
        else:
            self.guard(write)
        return await self.inner.aput_writes(config, writes, task_id, task_path)

    async def adelete_thread(self, thread_id: str) -> None:
        return await self.inner.adelete_thread(thread_id)

    async def adelete_for_runs(self, run_ids: Sequence[str]) -> None:
        return await self.inner.adelete_for_runs(run_ids)

    async def acopy_thread(self, source_thread_id: str, target_thread_id: str) -> None:
        return await self.inner.acopy_thread(source_thread_id, target_thread_id)

    async def aprune(
        self,
        thread_ids: Sequence[str],
        *,
        strategy: str = "keep_latest",
    ) -> None:
        return await self.inner.aprune(thread_ids, strategy=strategy)

    def get_delta_channel_history(
        self, *, config: RunnableConfig, channels: Sequence[str]
    ) -> Mapping[str, DeltaChannelHistory]:
        return self.inner.get_delta_channel_history(config=config, channels=channels)

    async def aget_delta_channel_history(
        self, *, config: RunnableConfig, channels: Sequence[str]
    ) -> Mapping[str, DeltaChannelHistory]:
        return await self.inner.aget_delta_channel_history(
            config=config, channels=channels
        )

    def get_next_version(self, current: V | None, channel: None) -> V:
        return self.inner.get_next_version(current, channel)

    def with_allowlist(
        self, extra_allowlist: Collection[tuple[str, ...]]
    ) -> GuardedCheckpointSaver[V]:
        return GuardedCheckpointSaver(
            self.inner.with_allowlist(extra_allowlist),
            self.guard,
            aguard=self.aguard,
        )
