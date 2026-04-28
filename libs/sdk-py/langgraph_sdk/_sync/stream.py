"""Synchronous thread-centric streaming primitives."""

from __future__ import annotations

import time
from collections.abc import Iterator, Mapping, Sequence
from typing import Any, cast

from langgraph_sdk._sync.http import SyncHttpClient
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


class SyncEventSubscription:
    """Iterator handle for a filtered event subscription."""

    def __init__(
        self,
        subscription_id: str,
        params: SubscribeParams,
        events: Iterator[Event],
        on_unsubscribe: Any,
    ) -> None:
        self.subscription_id = subscription_id
        self.params = params
        self._events = events
        self._on_unsubscribe = on_unsubscribe

    def __iter__(self) -> Iterator[Event]:
        return self._events

    def unsubscribe(self) -> None:
        self._on_unsubscribe(self.subscription_id)


class SyncProtocolSseTransport:
    """SSE transport for the thread-centric protocol."""

    def __init__(
        self,
        http: SyncHttpClient,
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

    def send(self, command: Command) -> CommandResponse | ErrorResponse | None:
        return self.http.post(
            self.commands_path,
            json=cast(dict[str, Any], command),
            headers=self.headers,
            params=self.params,
        )

    def open_event_stream(self, params: SubscribeParams) -> Iterator[Event]:
        for part in self.http.stream(
            self.events_path,
            "POST",
            json=cast(dict[str, Any], params),
            headers=self.headers,
            params=self.params,
        ):
            yield _stream_part_to_event(part)


class SyncRunModule:
    """Run commands exposed by ``SyncThreadStream.run``."""

    def __init__(self, stream: SyncThreadStream) -> None:
        self._stream = stream

    def input(self, params: Mapping[str, Any]) -> Any:
        return self._stream.command("run.input", dict(params))


class SyncInputModule:
    """Human-input commands exposed by ``SyncThreadStream.input``."""

    def __init__(self, stream: SyncThreadStream) -> None:
        self._stream = stream

    def respond(self, params: Mapping[str, Any]) -> Any:
        return self._stream.command("input.respond", dict(params))

    def inject(self, params: Mapping[str, Any]) -> Any:
        return self._stream.command("input.inject", dict(params))


class SyncStateModule:
    """State commands exposed by ``SyncThreadStream.state``."""

    def __init__(self, stream: SyncThreadStream) -> None:
        self._stream = stream

    def get(self, params: Mapping[str, Any] | None = None) -> Any:
        return self._stream.command("state.get", dict(params or {}))

    def list_checkpoints(self, params: Mapping[str, Any] | None = None) -> Any:
        return self._stream.command("state.listCheckpoints", dict(params or {}))

    def fork(self, params: Mapping[str, Any]) -> Any:
        return self._stream.command("state.fork", dict(params))


class SyncAgentModule:
    """Agent commands exposed by ``SyncThreadStream.agent``."""

    def __init__(self, stream: SyncThreadStream) -> None:
        self._stream = stream

    def get_tree(self, params: Mapping[str, Any] | None = None) -> Any:
        return self._stream.command("agent.getTree", dict(params or {}))


class SyncThreadStream:
    """High-level sync wrapper around a thread protocol transport."""

    def __init__(
        self,
        transport: SyncProtocolSseTransport,
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
        self.run = SyncRunModule(self)
        self.input = SyncInputModule(self)
        self.state = SyncStateModule(self)
        self.agent = SyncAgentModule(self)

    @property
    def thread_id(self) -> str:
        return self.transport.thread_id

    def _command_id(self) -> str:
        self._next_command_id += 1
        return str(self._next_command_id)

    def _subscription_id(self) -> str:
        self._next_subscription_id += 1
        return f"sub-{self._next_subscription_id}"

    def command(self, method: CommandMethod, params: dict[str, Any]) -> Any:
        command: Command = {
            "id": self._command_id(),
            "method": method,
            "params": params,
        }
        if method == "run.input":
            command["params"] = {"assistant_id": self.assistant_id, **params}
        response = self.transport.send(command)
        if response is None:
            return None
        if "error" in response:
            raise RuntimeError(response["error"])
        return response.get("result")

    def subscribe(
        self,
        channels: Sequence[Channel | str] | SubscribeParams,
        *,
        namespaces: Sequence[Sequence[str]] | None = None,
        depth: int | None = None,
    ) -> SyncEventSubscription:
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
        return SyncEventSubscription(subscription_id, params, events, self._unsubscribe)

    def _unsubscribe(self, subscription_id: str) -> None:
        self.command(
            "subscription.unsubscribe",
            {"subscription_id": subscription_id},
        )

    def close(self) -> None:
        return None

