"""Transformer that projects `tools` channel events into `ToolCallStream`s."""

from __future__ import annotations

import logging
import re
import time
import uuid
from collections.abc import Awaitable, Callable
from typing import Any

from langgraph.stream._types import ProtocolEvent, StreamTransformer
from langgraph.stream.stream_channel import StreamChannel

from langgraph.prebuilt._tool_call_stream import ToolCallStream

logger = logging.getLogger(__name__)

_SAFE_STRING_RE = re.compile(r"[^\w\s\-.,;:!?()\[\]{}/\\@#$%^&*+=|<>~`'\"]", re.UNICODE)
_MAX_STRING_LEN = 65536
_MAX_TOOL_NAME_LEN = 256


def _sanitize_string(value: Any, max_len: int = _MAX_STRING_LEN) -> str | None:
    """Sanitize a string value from MCP server output."""
    if value is None:
        return None
    if not isinstance(value, str):
        value = str(value)
    value = value[:max_len]
    return value


def _sanitize_tool_name(value: Any) -> str:
    """Sanitize a tool name from MCP server output."""
    if value is None:
        return ""
    if not isinstance(value, str):
        value = str(value)
    value = value[:_MAX_TOOL_NAME_LEN]
    value = re.sub(r"[^\w\-.]", "", value)
    return value


def _sanitize_tool_input(value: Any) -> dict[str, Any] | None:
    """Sanitize tool input dict from MCP server output."""
    if value is None:
        return None
    if not isinstance(value, dict):
        logger.warning(
            "tool_input_sanitization_failed: expected dict, got %s; coercing to None",
            type(value).__name__,
        )
        return None
    return value


def _sanitize_delta(value: Any) -> Any:
    """Sanitize a delta value from MCP server output."""
    if isinstance(value, str):
        return value[:_MAX_STRING_LEN]
    return value


def _sanitize_output(value: Any) -> Any:
    """Sanitize output value from MCP server output."""
    if isinstance(value, str):
        return value[:_MAX_STRING_LEN]
    return value


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

    An optional `allowed_tools` set may be provided at construction time
    to restrict which tool names are permitted. If `allowed_tools` is
    not None, any tool-started event for a tool not in the set will be
    denied and logged.
    """

    _native = True
    required_stream_modes = ("tools",)

    def __init__(
        self,
        scope: tuple[str, ...] = (),
        allowed_tools: set[str] | None = None,
    ) -> None:
        super().__init__(scope)
        self._log: StreamChannel[ToolCallStream] = StreamChannel()
        self._active: dict[str, ToolCallStream] = {}
        self._is_async = False
        self._pump_fn: Callable[[], bool] | None = None
        self._apump_fn: Callable[[], Awaitable[bool]] | None = None
        self._allowed_tools: set[str] | None = allowed_tools
        # Audit trail: list of audit records for forensic readiness
        self._audit_trail: list[dict[str, Any]] = []

    def _write_audit(self, record: dict[str, Any]) -> None:
        """Write an audit record to the audit trail and log it."""
        record.setdefault("timestamp", time.time())
        record.setdefault("audit_id", str(uuid.uuid4()))
        self._audit_trail.append(record)
        logger.info("audit_event: %s", record)

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

        # Log all interactions with the MCP server (Instruction 1)
        logger.info(
            "mcp_interaction: event_type=%s tool_call_id=%s scope=%s",
            event_type,
            tool_call_id,
            self.scope,
        )

        stream: ToolCallStream | None
        if event_type == "tool-started":
            raw_tool_name = data.get("tool_name", "")
            raw_tool_input = data.get("input")

            # Sanitize MCP server output (Instruction 2)
            tool_name = _sanitize_tool_name(raw_tool_name)
            tool_input = _sanitize_tool_input(raw_tool_input)

            # Enforce tool allow list (Instruction 4)
            if self._allowed_tools is not None and tool_name not in self._allowed_tools:
                self._write_audit({
                    "event": "tool_denied",
                    "tool_call_id": tool_call_id,
                    "tool_name": tool_name,
                    "reason": "tool_not_in_allow_list",
                    "policy": "allowed_tools",
                    "scope": self.scope,
                })
                logger.warning(
                    "tool_denied: tool_name=%s tool_call_id=%s not in allowed_tools",
                    tool_name,
                    tool_call_id,
                )
                return True

            # Audit record for tool-started (Instruction 3)
            self._write_audit({
                "event": "tool_started",
                "tool_call_id": tool_call_id,
                "tool_name": tool_name,
                "tool_input": tool_input,
                "scope": self.scope,
            })

            stream = self._new_stream(
                tool_call_id,
                tool_name,
                tool_input,
            )
            self._active[tool_call_id] = stream
            self._log.push(stream)
        elif event_type == "tool-output-delta":
            # Sanitize delta from MCP server (Instruction 2)
            raw_delta = data.get("delta")
            delta = _sanitize_delta(raw_delta)

            logger.info(
                "mcp_tool_output_delta: tool_call_id=%s",
                tool_call_id,
            )

            stream = self._active.get(tool_call_id)
            if stream is not None:
                stream._push_delta(delta)
        elif event_type == "tool-finished":
            # Sanitize output from MCP server (Instruction 2)
            raw_output = data.get("output")
            output = _sanitize_output(raw_output)

            stream = self._active.pop(tool_call_id, None)
            if stream is not None:
                # Audit record for tool-finished (Instruction 3)
                self._write_audit({
                    "event": "tool_finished",
                    "tool_call_id": tool_call_id,
                    "tool_name": stream.tool_name if hasattr(stream, "tool_name") else None,
                    "scope": self.scope,
                })
                stream._finish(output)
        elif event_type == "tool-error":
            # Sanitize error message from MCP server (Instruction 2)
            raw_message = data.get("message", "")
            message = _sanitize_string(raw_message) or ""

            stream = self._active.pop(tool_call_id, None)
            if stream is not None:
                # Audit record for tool-error (Instruction 3)
                self._write_audit({
                    "event": "tool_error",
                    "tool_call_id": tool_call_id,
                    "tool_name": stream.tool_name if hasattr(stream, "tool_name") else None,
                    "message": message,
                    "scope": self.scope,
                })
                stream._fail(message)

        # Pass-through — wire consumers subscribe to the `tools` channel
        # directly and reconstruct handles client-side.
        return True

    def finalize(self) -> None:
        """Close any still-active tool streams left open at run end."""
        for tool_call_id, stream in list(self._active.items()):
            if not stream.completed:
                # Audit record for forced finalization (Instruction 3)
                self._write_audit({
                    "event": "tool_finalized_at_run_end",
                    "tool_call_id": tool_call_id,
                    "tool_name": stream.tool_name if hasattr(stream, "tool_name") else None,
                    "scope": self.scope,
                    "reason": "run_ended_with_active_stream",
                })
                logger.info(
                    "mcp_tool_finalized: tool_call_id=%s reason=run_ended_with_active_stream",
                    tool_call_id,
                )
                stream._finish(None)
        self._active.clear()

    def fail(self, err: BaseException) -> None:
        """Fail any still-active tool streams when the run errors."""
        message = str(err)
        for tool_call_id, stream in list(self._active.items()):
            if not stream.completed:
                # Audit record for forced failure (Instruction 3)
                self._write_audit({
                    "event": "tool_failed_at_run_error",
                    "tool_call_id": tool_call_id,
                    "tool_name": stream.tool_name if hasattr(stream, "tool_name") else None,
                    "scope": self.scope,
                    "error": message,
                    "reason": "run_error",
                })
                logger.info(
                    "mcp_tool_failed: tool_call_id=%s reason=run_error error=%s",
                    tool_call_id,
                    message,
                )
                stream._fail(message)
        self._active.clear()