"""Transformer that projects `tools` channel events into `ToolCallStream`s."""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import Any

from langgraph.stream._types import ProtocolEvent, StreamTransformer
from langgraph.stream.stream_channel import StreamChannel

from langgraph.prebuilt._tool_call_stream import ToolCallStream


class ToolCallTransformer(StreamTransformer):
    """Project `tools` channel events into `ToolCallStream` handles.

    Each `tool-started` event spawns a `ToolCallStream`, pushed onto
    `run.tool_calls`. Subsequent `tool-output-delta` events append to
    that stream's deltas log; `tool-finished` and `tool-error` close it.

    Native transformer — the `tool_calls` projection is exposed as a
    direct attribute on the run stream.

    A nameless `StreamChannel[ToolCallStream]` is used (no protocol
    auto-forwarding) because the live handles are not serializable and
    should not be injected into the main event log. Wire consumers
    subscribe to the `tools` channel instead, where the raw protocol
    events flow through untouched by this transformer (`process`
    returns `True`).

    Registered explicitly by users at compile time via
    `builder.compile(transformers=[ToolCallTransformer])` — not a
    default built-in, so the `tools` channel is user-opt-in.
    """

    _native = True
    required_stream_modes = ("tools",)

    def __init__(self, scope: tuple[str, ...] = ()) -> None:
        super().__init__(scope)
        self._log: StreamChannel[ToolCallStream] = StreamChannel()
        self._active: dict[str, ToolCallStream] = {}
        self._is_async = False
        self._pump_fn: Callable[[], bool] | None = None
        self._apump_fn: Callable[[], Awaitable[bool]] | None = None

    def init(self) -> dict[str, Any]:
        return {"tool_calls": self._log}

    def _bind_pump(self, fn: Callable[[], bool]) -> None:
        """Wire the sync pull callback onto this transformer.

        Called by `StreamMux.bind_pump`. Stored so each new
        `ToolCallStream` created by `process` can wire its deltas log
        for pump-driven iteration.
        """
        self._pump_fn = fn
        self._is_async = False

    def _bind_apump(self, fn: Callable[[], Awaitable[bool]]) -> None:
        """Async counterpart to `_bind_pump`."""
        self._apump_fn = fn
        self._is_async = True

    def _new_stream(
        self,
        tool_call_id: str,
        tool_name: str,
        tool_input: dict[str, Any] | None,
    ) -> ToolCallStream:
        stream = ToolCallStream(tool_call_id, tool_name, tool_input)
        stream._bind(is_async=self._is_async)
        if self._apump_fn is not None:
            stream._output_deltas._arequest_more = self._apump_fn
        if self._pump_fn is not None:
            stream._output_deltas._request_more = self._pump_fn
        return stream

    def process(self, event: ProtocolEvent) -> bool:
        if event["method"] != "tools":
            return True

        # Only project events emitted at this transformer's scope. Subgraph
        # events still flow through the parent's mux (the parent's main
        # event log keeps them) but they belong to the child mini-mux's
        # `tool_calls` projection, not the parent's.
        if tuple(event["params"]["namespace"]) != self.scope:
            return True

        data = event["params"]["data"]
        tool_call_id = data.get("tool_call_id")
        if tool_call_id is None:
            return True
        event_type = data.get("event")

        stream: ToolCallStream | None
        if event_type == "tool-started":
            stream = self._new_stream(
                tool_call_id,
                data.get("tool_name", ""),
                data.get("input"),
            )
            self._active[tool_call_id] = stream
            self._log.push(stream)
        elif event_type == "tool-output-delta":
            stream = self._active.get(tool_call_id)
            if stream is not None:
                stream._push_delta(data.get("delta"))
        elif event_type == "tool-finished":
            stream = self._active.pop(tool_call_id, None)
            if stream is not None:
                stream._finish(data.get("output"))
        elif event_type == "tool-error":
            stream = self._active.pop(tool_call_id, None)
            if stream is not None:
                stream._fail(data.get("message", ""))

        # Pass-through — wire consumers subscribe to the `tools` channel
        # directly and reconstruct handles client-side.
        return True

    def finalize(self) -> None:
        """Close any still-active tool streams left open at run end."""
        for stream in self._active.values():
            if not stream.completed:
                stream._finish(None)
        self._active.clear()

    def fail(self, err: BaseException) -> None:
        """Fail any still-active tool streams when the run errors."""
        message = str(err)
        for stream in self._active.values():
            if not stream.completed:
                stream._fail(message)
        self._active.clear()
