from __future__ import annotations

from collections.abc import AsyncIterator, Callable, Iterator
from contextvars import Token
from typing import Any, TypeVar, cast
from uuid import UUID

from langchain_core.callbacks import BaseCallbackHandler

from langgraph._internal._constants import NS_SEP
from langgraph.config import _tool_call_writer
from langgraph.pregel.protocol import StreamChunk

try:
    from langchain_core.tracers._streaming import _StreamingCallbackHandler
except ImportError:
    _StreamingCallbackHandler = object  # type: ignore[assignment,misc]


T = TypeVar("T")

ToolCallWriter = Callable[[Any], None]
"""A closure bound to a single tool call that emits `tool-output-delta` events."""


class StreamToolCallHandler(BaseCallbackHandler, _StreamingCallbackHandler):
    """Callback handler that emits tool-call lifecycle events on the stream.

    Fires on LangChain's `on_tool_*` callbacks and pushes to the `tools`
    stream mode. Emits `tool-started` / `tool-output-delta` /
    `tool-finished` / `tool-error` payloads keyed by `tool_call_id`.

    While a tool is executing, this handler sets `_tool_call_writer` to a
    closure bound to that call's namespace and `tool_call_id`. The
    `emit_tool_output_delta` helper in `langgraph.config` reads that
    ContextVar so tool bodies can stream partial output without threading
    the writer through their own signature.

    Attached by `Pregel.stream` / `astream` when `"tools"` is in
    `stream_modes`. `run_inline = True` keeps event ordering
    deterministic.
    """

    run_inline = True

    def __init__(self, stream: Callable[[StreamChunk], None]) -> None:
        """Initialize the handler.

        Args:
            stream: Callable that accepts a `StreamChunk` tuple
                `(namespace, mode, payload)` and enqueues it.
        """
        self.stream = stream
        # run_id → (namespace, tool_call_id, ContextVar token)
        # `on_tool_end` does not receive `tool_call_id` in kwargs, so
        # we correlate by `run_id` which is present on every callback.
        self._run_to_call: dict[
            UUID, tuple[tuple[str, ...], str, Token[ToolCallWriter | None]]
        ] = {}

    @staticmethod
    def _containing_ns_from_metadata(
        metadata: dict[str, Any] | None,
    ) -> tuple[str, ...]:
        """Return the namespace of the subgraph that contains this tool call.

        `langgraph_checkpoint_ns` on a tool's callback metadata ends with
        the `node_name:task_id` segment of the node that invoked the
        tool. Dropping that segment gives the subgraph's own namespace,
        which matches what other `tools` / `lifecycle` / `messages`
        emitters use.
        """
        if not metadata:
            return ()
        nskey = metadata.get("langgraph_checkpoint_ns")
        if not nskey:
            return ()
        return tuple(cast(str, nskey).split(NS_SEP))[:-1]

    def _start(
        self,
        serialized: dict[str, Any] | None,
        input_str: str,
        *,
        run_id: UUID,
        metadata: dict[str, Any] | None,
        inputs: dict[str, Any] | None,
        kwargs: dict[str, Any],
    ) -> None:
        tool_call_id = cast("str | None", kwargs.get("tool_call_id")) or str(run_id)
        tool_name = (
            (serialized or {}).get("name")
            or cast("str | None", kwargs.get("name"))
            or ""
        )
        ns = self._containing_ns_from_metadata(metadata)

        def writer(delta: Any) -> None:
            self.stream(
                (
                    ns,
                    "tools",
                    {
                        "event": "tool-output-delta",
                        "tool_call_id": tool_call_id,
                        "delta": delta,
                    },
                )
            )

        token = _tool_call_writer.set(writer)
        self._run_to_call[run_id] = (ns, tool_call_id, token)

        payload: dict[str, Any] = {
            "event": "tool-started",
            "tool_call_id": tool_call_id,
            "tool_name": tool_name,
        }
        if inputs is not None:
            payload["input"] = inputs
        self.stream((ns, "tools", payload))

    def _end(self, output: Any, *, run_id: UUID) -> None:
        info = self._run_to_call.pop(run_id, None)
        if info is None:
            return
        ns, tool_call_id, token = info
        self._reset_writer(token)
        self.stream(
            (
                ns,
                "tools",
                {
                    "event": "tool-finished",
                    "tool_call_id": tool_call_id,
                    "output": output,
                },
            )
        )

    def _error(self, error: BaseException, *, run_id: UUID) -> None:
        info = self._run_to_call.pop(run_id, None)
        if info is None:
            return
        ns, tool_call_id, token = info
        self._reset_writer(token)
        self.stream(
            (
                ns,
                "tools",
                {
                    "event": "tool-error",
                    "tool_call_id": tool_call_id,
                    "message": str(error),
                },
            )
        )

    def tap_output_aiter(
        self, run_id: UUID, output: AsyncIterator[T]
    ) -> AsyncIterator[T]:
        """Pass-through — required by the `_StreamingCallbackHandler` protocol."""
        return output

    def tap_output_iter(self, run_id: UUID, output: Iterator[T]) -> Iterator[T]:
        """Pass-through — sync counterpart to `tap_output_aiter`."""
        return output

    @staticmethod
    def _reset_writer(token: Token[ToolCallWriter | None]) -> None:
        # Token is invalid if `on_tool_end` runs in a different context
        # than `on_tool_start` (e.g. langchain may hand off to a thread
        # worker without copying the context). Swallow that case; the
        # ContextVar lifetime is bounded by the enclosing task anyway.
        try:
            _tool_call_writer.reset(token)
        except ValueError:
            pass

    # ------------------------------------------------------------------
    # Sync callbacks
    # ------------------------------------------------------------------

    def on_tool_start(
        self,
        serialized: dict[str, Any],
        input_str: str,
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        tags: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
        inputs: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> Any:
        self._start(
            serialized,
            input_str,
            run_id=run_id,
            metadata=metadata,
            inputs=inputs,
            kwargs=kwargs,
        )

    def on_tool_end(
        self,
        output: Any,
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        **kwargs: Any,
    ) -> Any:
        self._end(output, run_id=run_id)

    def on_tool_error(
        self,
        error: BaseException,
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        **kwargs: Any,
    ) -> Any:
        self._error(error, run_id=run_id)
