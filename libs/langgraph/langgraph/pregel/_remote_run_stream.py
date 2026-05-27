# libs/langgraph/langgraph/pregel/_remote_run_stream.py
from __future__ import annotations

import asyncio
import logging
import sys
from collections.abc import AsyncIterator, Iterator
from typing import Any

from langgraph_sdk._async.stream import AsyncThreadStream
from langgraph_sdk._sync.stream import SyncThreadStream
from langgraph_sdk.client import LangGraphClient, SyncLangGraphClient

logger = logging.getLogger(__name__)

_SENTINEL: Any = object()


class _RemoteGraphRunStream:
    """Sync adapter: SyncThreadStream -> GraphRunStream surface."""

    def __init__(
        self,
        *,
        sync_client: SyncLangGraphClient,
        sdk_thread: SyncThreadStream,
        input: Any,
        config: dict[str, Any] | None,
        metadata: dict[str, Any] | None,
    ) -> None:
        self._client = sync_client
        self._sdk = sdk_thread
        self._start_kwargs: dict[str, Any] = {
            "input": input,
            "config": config,
            "metadata": metadata,
        }
        self._run_id: str | None = None
        self._closed = False
        self._events_iter: Iterator[Any] | None = None


class _AsyncRemoteGraphRunStream:
    """Async adapter: AsyncThreadStream -> AsyncGraphRunStream surface."""

    def __init__(
        self,
        *,
        client: LangGraphClient,
        sdk_thread: AsyncThreadStream,
        input: Any,
        config: dict[str, Any] | None,
        metadata: dict[str, Any] | None,
    ) -> None:
        self._client = client
        self._sdk = sdk_thread
        self._start_kwargs: dict[str, Any] = {
            "input": input,
            "config": config,
            "metadata": metadata,
        }
        self._run_id: str | None = None
        self._closed = False
        self._events_aiter: AsyncIterator[Any] | None = None
