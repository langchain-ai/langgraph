"""In-process handle for a single tool call's streaming execution.

Mirrors the shape of `ChatModelStream` from langchain-core but simpler —
a tool has one output channel, no content-block multiplexing. Populated
by `ToolCallTransformer` as `tool-started` / `tool-output-delta` /
`tool-finished` / `tool-error` events flow in on the `tools` channel.
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Iterator
from typing import Any

from langgraph.stream.stream_channel import StreamChannel


class ToolCallStream:
    """Scoped view of a single tool call's lifecycle.

    Yielded on `run.tool_calls` once per `tool-started` event. Fields
    are populated as events arrive:

    - `tool_call_id`, `tool_name`, `input`: stable from the start event.
    - `output_deltas`: a `StreamChannel` of delta chunks. Iterate (sync or
      async) to consume partial output in arrival order.
    - `output`: terminal payload from `tool-finished`, or `None` if the
      call failed or is still in flight.
    - `error`: terminal error string from `tool-error`, or `None` if the
      call succeeded or is still in flight.
    - `completed`: True once a terminal event (`tool-finished` or
      `tool-error`) has been observed.

    `ToolCallStream` is not meant to be constructed by end users — it's
    produced by `ToolCallTransformer` as events flow through the mux.
    """

    def __init__(
        self,
        tool_call_id: str,
        tool_name: str,
        input: dict[str, Any] | None = None,
    ) -> None:
        """Initialize a fresh handle for a tool call.

        Args:
            tool_call_id: The `tool_call_id` from the AIMessage.
            tool_name: The tool's name.
            input: The tool's input arguments (as reported by
                `on_tool_start`), or `None` if none were captured.
        """
        self.tool_call_id = tool_call_id
        self.tool_name = tool_name
        self.input = input
        self._output_deltas: StreamChannel[Any] = StreamChannel()
        self.output: Any = None
        self.error: str | None = None
        self.completed = False

    @property
    def output_deltas(self) -> StreamChannel[Any]:
        """The channel of streamed `tool-output-delta` payloads.

        Iterate (sync or async depending on how the run was started)
        to consume partial output in arrival order. The log closes when
        the tool finishes or errors.
        """
        return self._output_deltas

    def _bind(self, *, is_async: bool) -> None:
        """Bind the deltas log to sync or async iteration.

        Called by `ToolCallTransformer` when constructing this handle so
        the log matches the enclosing mux's mode.
        """
        self._output_deltas._bind(is_async=is_async)

    def _push_delta(self, delta: Any) -> None:
        self._output_deltas.push(delta)

    def _finish(self, output: Any) -> None:
        self.output = output
        self.completed = True
        self._output_deltas.close()

    def _fail(self, message: str) -> None:
        self.error = message
        self.completed = True
        self._output_deltas.close()

    def __iter__(self) -> Iterator[Any]:
        """Iterate delta chunks synchronously.

        Equivalent to `iter(self.output_deltas)`. Raises `TypeError` if
        the underlying log is bound to async mode.
        """
        return iter(self._output_deltas)

    def __aiter__(self) -> AsyncIterator[Any]:
        """Iterate delta chunks asynchronously.

        Equivalent to `aiter(self.output_deltas)`. Raises `TypeError`
        if the underlying log is bound to sync mode.
        """
        return self._output_deltas.__aiter__()

    def __repr__(self) -> str:
        status = (
            "completed"
            if self.completed and self.error is None
            else "failed"
            if self.completed
            else "running"
        )
        return (
            f"ToolCallStream(tool_call_id={self.tool_call_id!r}, "
            f"tool_name={self.tool_name!r}, status={status})"
        )
