"""Async thread-centric streaming primitives."""

from __future__ import annotations

import time
from collections.abc import AsyncIterator, Mapping, Sequence
from typing import Any, cast

from langgraph_sdk._async.http import HttpClient
from langgraph_sdk.protocol import (
    Channel,
    Command,
    CommandMethod,
    CommandResponse,
    ErrorResponse,
    Event,
    SubscribeParams,
)
from langgraph_sdk.schema import QueryParamTypes, StreamPart


def _stream_part_to_event(part: StreamPart) -> Event:
    """Normalize an SSE ``StreamPart`` into a protocol event envelope."""
    if isinstance(part.data, dict) and part.data.get("type") == "event":
        event = cast(Event, part.data)
    else:
        event = {
            "type": "event",
            "method": part.event,
            "params": {
                "namespace": [],
                "timestamp": int(time.time() * 1000),
                "data": part.data,
            },
        }
    if part.id is not None and "event_id" not in event:
        event["event_id"] = part.id
    return event


class EventSubscription:
    """Async iterable handle for a filtered event subscription."""

    def __init__(
        self,
        subscription_id: str,
        params: SubscribeParams,
        events: AsyncIterator[Event],
        on_unsubscribe: Any,
    ) -> None:
        self.subscription_id = subscription_id
        self.params = params
        self._events = events
        self._on_unsubscribe = on_unsubscribe

    def __aiter__(self) -> AsyncIterator[Event]:
        return self._events

    async def unsubscribe(self) -> None:
        await self._on_unsubscribe(self.subscription_id)


class ProtocolSseTransport:
    """SSE transport for the thread-centric protocol."""

    def __init__(
        self,
        http: HttpClient,
        thread_id: str,
        *,
        headers: Mapping[str, str] | None = None,
        params: QueryParamTypes | None = None,
    ) -> None:
        self.http = http
        self.thread_id = thread_id
        self.headers = headers
        self.params = params
        self.commands_path = f"/v2/threads/{thread_id}/commands"
        self.events_path = f"/v2/threads/{thread_id}/events"

    async def send(self, command: Command) -> CommandResponse | ErrorResponse | None:
        return await self.http.post(
            self.commands_path,
            json=cast(dict[str, Any], command),
            headers=self.headers,
            params=self.params,
        )

    def open_event_stream(self, params: SubscribeParams) -> AsyncIterator[Event]:
        async def iterate() -> AsyncIterator[Event]:
            async for part in self.http.stream(
                self.events_path,
                "POST",
                json=cast(dict[str, Any], params),
                headers=self.headers,
                params=self.params,
            ):
                yield _stream_part_to_event(part)

        return iterate()


class RunModule:
    """Run commands exposed by ``ThreadStream.run``."""

    def __init__(self, stream: ThreadStream) -> None:
        self._stream = stream

    async def input(self, params: Mapping[str, Any]) -> Any:
        return await self._stream.command("run.input", dict(params))


class InputModule:
    """Human-input commands exposed by ``ThreadStream.input``."""

    def __init__(self, stream: ThreadStream) -> None:
        self._stream = stream

    async def respond(self, params: Mapping[str, Any]) -> Any:
        return await self._stream.command("input.respond", dict(params))

    async def inject(self, params: Mapping[str, Any]) -> Any:
        return await self._stream.command("input.inject", dict(params))


class StateModule:
    """State commands exposed by ``ThreadStream.state``."""

    def __init__(self, stream: ThreadStream) -> None:
        self._stream = stream

    async def get(self, params: Mapping[str, Any] | None = None) -> Any:
        return await self._stream.command("state.get", dict(params or {}))

    async def list_checkpoints(self, params: Mapping[str, Any] | None = None) -> Any:
        return await self._stream.command("state.listCheckpoints", dict(params or {}))

    async def fork(self, params: Mapping[str, Any]) -> Any:
        return await self._stream.command("state.fork", dict(params))


class AgentModule:
    """Agent commands exposed by ``ThreadStream.agent``."""

    def __init__(self, stream: ThreadStream) -> None:
        self._stream = stream

    async def get_tree(self, params: Mapping[str, Any] | None = None) -> Any:
        return await self._stream.command("agent.getTree", dict(params or {}))


class ThreadStream:
    """High-level async wrapper around a thread protocol transport."""

    def __init__(
        self,
        transport: ProtocolSseTransport,
        *,
        assistant_id: str,
        starting_command_id: int = 0,
    ) -> None:
        if not assistant_id:
            raise ValueError("assistant_id is required")
        self.transport = transport
        self.assistant_id = assistant_id
        self._next_command_id = starting_command_id
        self._next_subscription_id = 0
        self.run = RunModule(self)
        self.input = InputModule(self)
        self.state = StateModule(self)
        self.agent = AgentModule(self)

    @property
    def thread_id(self) -> str:
        return self.transport.thread_id

    def _command_id(self) -> str:
        self._next_command_id += 1
        return str(self._next_command_id)

    def _subscription_id(self) -> str:
        self._next_subscription_id += 1
        return f"sub-{self._next_subscription_id}"

    async def command(self, method: CommandMethod, params: dict[str, Any]) -> Any:
        command: Command = {
            "id": self._command_id(),
            "method": method,
            "params": params,
        }
        if method == "run.input":
            command["params"] = {"assistant_id": self.assistant_id, **params}
        response = await self.transport.send(command)
        if response is None:
            return None
        if "error" in response:
            raise RuntimeError(response["error"])
        return response.get("result")

    async def subscribe(
        self,
        channels: Sequence[Channel | str] | SubscribeParams,
        *,
        namespaces: Sequence[Sequence[str]] | None = None,
        depth: int | None = None,
    ) -> EventSubscription:
        if isinstance(channels, dict):
            params = SubscribeParams(**channels)
        else:
            params = SubscribeParams(channels=list(channels))
            if namespaces is not None:
                params["namespaces"] = [list(ns) for ns in namespaces]
            if depth is not None:
                params["depth"] = depth

        subscription_id = self._subscription_id()
        events = self.transport.open_event_stream(params)
        return EventSubscription(subscription_id, params, events, self._unsubscribe)

    async def _unsubscribe(self, subscription_id: str) -> None:
        await self.command(
            "subscription.unsubscribe",
            {"subscription_id": subscription_id},
        )

    async def close(self) -> None:
        return None

