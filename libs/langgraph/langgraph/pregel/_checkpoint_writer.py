from __future__ import annotations

import asyncio
import os
import queue
import threading
from collections.abc import Callable
from contextlib import AbstractAsyncContextManager, AbstractContextManager
from dataclasses import dataclass
from types import TracebackType
from typing import Any

from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.base import (
    ChannelVersions,
    Checkpoint,
    CheckpointMetadata,
)

QUEUE_PUT_TIMEOUT = 0.05
CHECKPOINT_BACKLOG_ENV_VAR = "LANGGRAPH_CHECKPOINT_BACKLOG"
DEFAULT_CHECKPOINT_BACKLOG = 10


@dataclass(frozen=True)
class CheckpointRequest:
    config: RunnableConfig
    checkpoint: Checkpoint
    metadata: CheckpointMetadata
    new_versions: ChannelVersions


def _raise(error: BaseException) -> None:
    raise error


def resolve_checkpoint_backlog() -> int:
    if raw := os.getenv(CHECKPOINT_BACKLOG_ENV_VAR):
        try:
            backlog = int(raw)
        except ValueError:
            return DEFAULT_CHECKPOINT_BACKLOG
        if backlog > 0:
            return backlog
    return DEFAULT_CHECKPOINT_BACKLOG


class SyncCheckpointWriter(AbstractContextManager):
    def __init__(
        self,
        put: Callable[
            [RunnableConfig, Checkpoint, CheckpointMetadata, ChannelVersions], Any
        ],
        *,
        max_pending: int | None = None,
    ) -> None:
        self.put = put
        max_pending = (
            resolve_checkpoint_backlog() if max_pending is None else max_pending
        )
        self.queue: queue.Queue[CheckpointRequest | None] = queue.Queue(max_pending)
        self.error: BaseException | None = None
        self.closed = False
        self.thread = threading.Thread(
            target=self._run,
            name="langgraph-checkpoint-writer",
            daemon=True,
        )

    def __enter__(self) -> SyncCheckpointWriter:
        self.thread.start()
        return self

    def submit(self, request: CheckpointRequest) -> None:
        self._ensure_open()
        while True:
            self._raise_if_broken()
            try:
                self.queue.put(request, timeout=QUEUE_PUT_TIMEOUT)
            except queue.Full:
                continue
            else:
                self._raise_if_broken()
                return

    def _run(self) -> None:
        while True:
            item = self.queue.get()
            if item is None:
                return
            try:
                self.put(
                    item.config,
                    item.checkpoint,
                    item.metadata,
                    item.new_versions,
                )
            except BaseException as exc:
                self.error = exc
                return

    def _ensure_open(self) -> None:
        if self.closed:
            raise RuntimeError("Checkpoint writer is closed")

    def _raise_if_broken(self) -> None:
        if self.error is not None:
            _raise(self.error)

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> bool | None:
        self.closed = True
        while self.thread.is_alive():
            if self.error is not None:
                break
            try:
                self.queue.put(None, timeout=QUEUE_PUT_TIMEOUT)
            except queue.Full:
                continue
            else:
                break
        self.thread.join()
        if exc_type is None and self.error is not None:
            _raise(self.error)
        return None


class AsyncCheckpointWriter(AbstractAsyncContextManager):
    def __init__(
        self,
        put: Callable[
            [RunnableConfig, Checkpoint, CheckpointMetadata, ChannelVersions], Any
        ],
        *,
        max_pending: int | None = None,
    ) -> None:
        self.put = put
        max_pending = (
            resolve_checkpoint_backlog() if max_pending is None else max_pending
        )
        self.queue: asyncio.Queue[CheckpointRequest | None] = asyncio.Queue(max_pending)
        self.error: BaseException | None = None
        self.closed = False
        self.task: asyncio.Task[None] | None = None

    async def __aenter__(self) -> AsyncCheckpointWriter:
        self.task = asyncio.create_task(self._run(), name="langgraph-checkpoint-writer")
        return self

    async def submit(self, request: CheckpointRequest) -> None:
        self._ensure_open()
        while True:
            self._raise_if_broken()
            try:
                await asyncio.wait_for(
                    self.queue.put(request),
                    timeout=QUEUE_PUT_TIMEOUT,
                )
            except asyncio.TimeoutError:
                continue
            else:
                self._raise_if_broken()
                return

    async def _run(self) -> None:
        while True:
            item = await self.queue.get()
            if item is None:
                return
            try:
                await self.put(
                    item.config,
                    item.checkpoint,
                    item.metadata,
                    item.new_versions,
                )
            except BaseException as exc:
                self.error = exc
                return

    def _ensure_open(self) -> None:
        if self.closed:
            raise RuntimeError("Checkpoint writer is closed")

    def _raise_if_broken(self) -> None:
        if self.error is not None:
            _raise(self.error)

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.closed = True
        while self.task is not None and not self.task.done():
            if self.error is not None:
                break
            try:
                await asyncio.wait_for(
                    self.queue.put(None),
                    timeout=QUEUE_PUT_TIMEOUT,
                )
            except asyncio.TimeoutError:
                continue
            else:
                break
        if self.task is not None:
            await self.task
        if exc_type is None and self.error is not None:
            _raise(self.error)
