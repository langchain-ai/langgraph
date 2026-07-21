from __future__ import annotations

from collections.abc import AsyncIterator, Callable, Iterator
from contextvars import ContextVar, Token
from typing import Any, TypeVar, cast
from uuid import UUID

from langchain_core.callbacks import BaseCallbackHandler

from langgraph._internal._constants import NS_SEP
from langgraph.constants import TAG_NOSTREAM
from langgraph.pregel.protocol import StreamChunk

try:
    from langchain_core.tracers._streaming import _StreamingCallbackHandler
except ImportError:
    _StreamingCallbackHandler = object  # type: ignore[assignment,misc]


T = TypeVar("T")

ToolCallWriter = Callable[[Any], None]
"""A closure bound to a single tool call that emits `tool-output-delta` events."""

_tool_call_writer: ContextVar[ToolCallWriter | None] = ContextVar(
    "langgraph_tool_call_writer", default=None
)
"""ContextVar holding the writer for the currently-executing tool call.

Set by `StreamToolCallHandler.on_tool_start` and reset on end/error.
Read by `ToolRuntime.emit_output_delta` (in `langgraph.prebuilt`).
"""


class StreamToolCallHandler(BaseCallbackHandler, _StreamingCallbackHandler):
    """Callback handler that emits tool-call lifecycle events on the stream.

    Fires on LangChain's `on_tool_*` callbacks and pushes to the `tools`
    stream mode. Emits `tool-started` / `tool-output-delta` /
    `tool-finished` / `tool-error` payloads keyed by `tool_call_id`.

    While a tool is executing, this handler sets `_tool_call_writer` to a
    closure bound to that call's namespace and `tool_call_id`.
    `ToolRuntime.emit_output_delta` reads that ContextVar so tool bodies
    can stream partial output without threading the writer through their
    own signature.

    Attached by `Pregel.stream` / `astream` when `"tools"` is in
    `stream_modes`. `run_inline = True` keeps event ordering
    deterministic.
    """

    run_inline = True

    def __init__(
        self,
        stream: Callable[[StreamChunk], None],
        subgraphs: bool,
        *,
        parent_ns: tuple[str, ...] | None = None,
    ) -> None:
        """Configure the handler to stream tool-call events.

        Args:
            stream: Callable that accepts a `StreamChunk` tuple
                `(namespace, mode, payload)` and enqueues it.
            subgraphs: Whether to emit events from tools called inside
                nested subgraphs. When False, only tools at the
                handler's own scope (`parent_ns`) emit.
            parent_ns: Namespace where the handler was attached.
                Mirrors the `StreamMessagesHandler` escape hatch:
                tools whose containing namespace equals `parent_ns`
                still emit even with `subgraphs=False`, so a node that
                explicitly streams a subgraph with `stream_mode="tools"`
                sees its own tools.
        """
        self.stream = stream
        self.subgraphs = subgraphs
        self.parent_ns = parent_ns
        # run_id → (namespace, tool_call_id, ContextVar token)
        # `on_tool_end` does not receive `tool_call_id` in kwargs, so
        # we correlate by `run_id` which is present on every callback.
        self._run_to_call: dict[
            UUID, tuple[tuple[str, ...], str, Token[ToolCallWriter | None]]
        ] = {}

    def _ns_for_emit(
        self,
        metadata: dict[str, Any] | None,
        tags: list[str] | None,
    ) -> tuple[str, ...] | None:
        """Resolve the namespace this tool call should emit at, or `None` to skip.

        Mirrors `StreamMessagesHandler.on_chat_model_start`'s namespace
        derivation: parses `langgraph_checkpoint_ns` (which ends with
        the `node_name:task_id` of the calling node), drops that
        trailing segment, and returns the containing subgraph's own
        namespace. Returns `None` when the call should be silently
        suppressed:

        - `metadata` is missing — handler is attached to a context
          without Pregel routing info.
        - `TAG_NOSTREAM` is in `tags` — caller explicitly opted out.
        - Tool runs in a subgraph (`len(ns) > 0`) and the handler was
          attached with `subgraphs=False` and a different `parent_ns`
          than the call's containing subgraph.
        """
        if not metadata:
            return None
        if tags and TAG_NOSTREAM in tags:
            return None
        nskey = metadata.get("langgraph_checkpoint_ns")
        if not nskey:
            ns: tuple[str, ...] = ()
        else:
            ns = tuple(cast(str, nskey).split(NS_SEP))[:-1]
        if not self.subgraphs and len(ns) > 0 and ns != self.parent_ns:
            return None
        return ns

    def _start(
        self,
        serialized: dict[str, Any] | None,
        input_str: str,
        *,
        run_id: UUID,
        metadata: dict[str, Any] | None,
        tags: list[str] | None,
        inputs: dict[str, Any] | None,
        kwargs: dict[str, Any],
    ) -> None:
        ns = self._ns_for_emit(metadata, tags)
        if ns is None:
            return
        tool_call_id = cast("str | None", kwargs.get("tool_call_id")) or str(run_id)
        tool_name = (
            (serialized or {}).get("name")
            or cast("str | None", kwargs.get("name"))
            or ""
        )

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
            tags=tags,
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
