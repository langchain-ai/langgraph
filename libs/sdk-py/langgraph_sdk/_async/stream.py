"""Async thread-centric streaming surface for the v3 protocol.

`AsyncThreadStream` is an async context manager that owns a
`ProtocolSseTransport` for one thread, dispatches `run.start` commands,
and exposes a raw `events` async iterable.

Direct port of `libs/sdk/src/client/stream/index.ts`.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any

import httpx
from langchain_protocol import Event

from langgraph_sdk.stream.transport import EventStreamHandle, ProtocolSseTransport

# All public protocol channels used by the raw `events` surface.
_ALL_CHANNELS: list[str] = [
    "values",
    "updates",
    "messages",
    "tools",
    "lifecycle",
    "input",
    "checkpoints",
    "tasks",
    "custom",
]


class RunModule:
    """Command dispatcher for `run.start`.

    Bound to one `AsyncThreadStream`; accesses its transport and id allocator.
    """

    def __init__(self, owner: AsyncThreadStream) -> None:
        self._owner = owner

    async def start(
        self,
        *,
        input: Any = None,
        config: dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Send `run.start` to the server. Returns the result (`{"run_id": ...}`)."""
        params: dict[str, Any] = {"assistant_id": self._owner.assistant_id}
        if input is not None:
            params["input"] = input
        if config is not None:
            params["config"] = config
        if metadata is not None:
            params["metadata"] = metadata
        return await self._owner._send_command("run.start", params)


class RunModule:
    """Command dispatcher for `run.start`.

    Bound to one `AsyncThreadStream`; accesses its transport and id allocator.
    """

    def __init__(self, owner: AsyncThreadStream) -> None:
        self._owner = owner

    async def start(
        self,
        *,
        input: Any = None,
        config: dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Send `run.start` to the server. Returns the result (`{"run_id": ...}`)."""
        params: dict[str, Any] = {"assistant_id": self._owner.assistant_id}
        if input is not None:
            params["input"] = input
        if config is not None:
            params["config"] = config
        if metadata is not None:
            params["metadata"] = metadata
        return await self._owner._send_command("run.start", params)


class AsyncThreadStream:
    """Async context manager for one thread's v3 streaming session.

    Construct via `client.threads.stream(thread_id=None, *, assistant_id, ...)`
    rather than instantiating directly.
    """

    def __init__(
        self,
        *,
        client: httpx.AsyncClient,
        thread_id: str,
        assistant_id: str,
    ) -> None:
        self._http_client = client
        self.thread_id = thread_id
        self.assistant_id = assistant_id
        self._closed = False
        self._transport: ProtocolSseTransport | None = None
        self._open_handles: list[EventStreamHandle] = []
        self._next_command_id = 1
        self.run = RunModule(self)

    async def __aenter__(self) -> AsyncThreadStream:
        if self._closed:
            raise RuntimeError("AsyncThreadStream is closed and cannot be re-entered.")
        self._transport = ProtocolSseTransport(
            client=self._http_client,
            thread_id=self.thread_id,
        )
        return self

    async def __aexit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        try:
            await self.close()
        except BaseException as close_err:
            if exc is None:
                raise
            # Original exception takes precedence; chain close error as context.
            close_err.__context__ = exc

    @property
    def events(self) -> AsyncIterator[Event]:
        """Return a fresh subscription to ALL channels.

        Each property access opens a new subscription; callers iterating twice
        will see two independent streams (both filtered by the same channel union).
        Terminates when the stream closes (server hangup, `__aexit__`, or
        transport-level close).
        """
        if self._transport is None:
            raise RuntimeError("AsyncThreadStream not entered — use `async with`.")
        handle = self._transport.open_event_stream({"channels": _ALL_CHANNELS})
        self._open_handles.append(handle)
        return handle.events

    async def close(self) -> None:
        """Tear down the thread stream. Idempotent."""
        if self._closed:
            return
        self._closed = True
        for handle in self._open_handles:
            await handle.close()
        if self._transport is not None:
            await self._transport.close()

    async def _send_command(
        self, method: str, params: dict[str, Any]
    ) -> dict[str, Any]:
        """Send a protocol command and return the `result` payload.

        Returns `{}` for 202/204 responses (no body). Raises `RuntimeError`
        with the protocol code/message when the server returns an error
        envelope (`{"type": "error", ...}`).
        """
        if self._transport is None:
            raise RuntimeError("AsyncThreadStream not entered — use `async with`.")
        command_id = self._next_command_id
        self._next_command_id += 1
        response = await self._transport.send_command(
            {"id": command_id, "method": method, "params": params}
        )
        if response is None:
            # 202/204 — no body. Caller gets an empty result.
            return {}
        if response.get("type") == "error":
            code = response.get("error", "unknown")
            message = response.get("message", "")
            raise RuntimeError(f"Protocol error [{code}]: {message}")
        return response.get("result", {})
