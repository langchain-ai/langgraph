from __future__ import annotations

import re
from collections.abc import AsyncIterator, Callable, Iterator, Sequence
from dataclasses import fields, is_dataclass
from typing import (
    Any,
    TypeVar,
    cast,
)
from uuid import UUID, uuid4

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.messages import BaseMessage, ToolMessage
from langchain_core.messages.utils import convert_to_messages
from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, LLMResult
from pydantic import BaseModel

from langgraph._internal._config import filter_to_user_tags
from langgraph._internal._constants import NS_SEP
from langgraph.constants import TAG_HIDDEN, TAG_NOSTREAM
from langgraph.pregel.protocol import StreamChunk
from langgraph.types import Command

# Patterns that indicate content likely originated from a tool call that the
# provider's streaming parser failed to classify.  When a chunk matches one of
# these and carries no `tool_call_chunks`, the content is buffered rather than
# emitted as user-visible text.  On `on_llm_end` the buffer is discarded if the
# finalized message contains tool calls (the buffered text was a parse error)
# or flushed as genuine assistant text otherwise.
_TOOL_CALL_LEAK_PATTERNS: list[re.Pattern[str]] = [
    # Anthropic-style tool-call boundary: "to=functions.tool_name"
    re.compile(r"^\s*to=functions\."),
    # Raw JSON that looks like a serialised tool-call dict
    re.compile(r'^\s*\{[^}]*"name"\s*:'),
    # Tool-call arguments block starting mid-stream
    re.compile(r'^\s*\{[^}]*"arguments"\s*:'),
]

try:
    from langchain_core.tracers._streaming import _StreamingCallbackHandler
except ImportError:
    _StreamingCallbackHandler = object  # type: ignore

try:
    from langchain_core.tracers._streaming import _V2StreamingCallbackHandler
except ImportError:
    _V2StreamingCallbackHandler = object  # type: ignore

T = TypeVar("T")
Meta = tuple[tuple[str, ...], dict[str, Any]]


def _state_values(obj: Any) -> Sequence[Any]:
    """Extract top-level field values from a state object (dict, BaseModel, or dataclass)."""
    if isinstance(obj, dict):
        return list(obj.values())
    elif isinstance(obj, BaseModel):
        return [getattr(obj, k) for k in type(obj).model_fields]
    elif is_dataclass(obj) and not isinstance(obj, type):
        return [getattr(obj, f.name) for f in fields(obj)]
    return ()


def _is_tool_call_like_content(content: str) -> bool:
    """Return True if *content* resembles a tool-call payload that the
    provider's streaming parser may have failed to classify.

    This is a best-effort heuristic keyed on common provider-specific
    tool-call boundary markers (e.g. Anthropic's ``to=functions.``
    prefix) and JSON shape signatures (``{"name":``, ``{"arguments":``).
    False positives (genuine JSON output or code snippets) are buffered
    temporarily and flushed on ``on_llm_end`` when the finalized message
    contains no tool calls, so they are not permanently dropped.
    """
    return any(p.search(content) for p in _TOOL_CALL_LEAK_PATTERNS)


class StreamMessagesHandler(BaseCallbackHandler, _StreamingCallbackHandler):
    """A callback handler that implements stream_mode=messages.

    Collects messages from:
    (1) chat model stream events; and
    (2) node outputs.
    """

    run_inline = True
    """We want this callback to run in the main thread to avoid order/locking issues."""

    def __init__(
        self,
        stream: Callable[[StreamChunk], None],
        subgraphs: bool,
        *,
        parent_ns: tuple[str, ...] | None = None,
    ) -> None:
        """Configure the handler to stream messages from LLMs and nodes.

        Args:
            stream: A callable that takes a StreamChunk and emits it.
            subgraphs: Whether to emit messages from subgraphs.
            parent_ns: The namespace where the handler was created.
                We keep track of this namespace to allow calls to subgraphs that
                were explicitly requested as a stream with `messages` mode
                configured.

        Example:
            parent_ns is used to handle scenarios where the subgraph is explicitly
            streamed with `stream_mode="messages"`.

            ```python
            def parent_graph_node():
                # This node is in the parent graph.
                async for event in some_subgraph(..., stream_mode="messages"):
                    do something with event # <-- these events will be emitted
                return ...

            parent_graph.invoke(subgraphs=False)
            ```
        """
        self.stream = stream
        self.subgraphs = subgraphs
        self.metadata: dict[UUID, Meta] = {}
        self.seen: set[int | str] = set()
        self.parent_ns = parent_ns
        # Per-run buffer for content that looks like leaked tool-call syntax.
        # Maps run_id → list of buffered content fragments (strings).
        self._tc_buffer: dict[UUID, list[str]] = {}
        # Tracks run_ids where a tool_call_chunk has been observed so we can
        # discriminate buffered-into-tool-call vs genuine text-abandoned.
        self._tc_seen: set[UUID] = set()

    def _emit(self, meta: Meta, message: BaseMessage, *, dedupe: bool = False) -> None:
        if dedupe and message.id in self.seen:
            return
        else:
            if message.id is None:
                message.id = str(uuid4())
            self.seen.add(message.id)
            self.stream((meta[0], "messages", (message, meta[1])))

    def _flush_tc_buffer(
        self, meta: Meta, template_msg: BaseMessage, run_id: UUID
    ) -> None:
        """Flush buffered tool-call-like content as synthetic text chunks.

        Emits the concatenated buffer content so consumers see the complete
        text stream when the buffered content turned out to be genuine
        assistant prose (not a malformed tool call).
        """
        buf = self._tc_buffer.pop(run_id, [])
        if not buf:
            return
        joined = "".join(buf)
        if not joined:
            return
        # Create a synthetic message chunk from the buffered content so the
        # emitter can assign an id and add it to the dedupe set.
        synthetic = template_msg.__class__(content=joined)
        if synthetic.id is None:
            synthetic.id = str(uuid4())
        self._emit(meta, synthetic)

    def _find_and_emit_messages(self, meta: Meta, response: Any) -> None:
        if isinstance(response, BaseMessage):
            self._emit(meta, response, dedupe=True)
        elif isinstance(response, Sequence):
            for value in response:
                if isinstance(value, BaseMessage):
                    self._emit(meta, value, dedupe=True)
        else:
            for value in _state_values(response):
                if isinstance(value, BaseMessage):
                    self._emit(meta, value, dedupe=True)
                elif isinstance(value, Sequence):
                    for item in value:
                        if isinstance(item, BaseMessage):
                            self._emit(meta, item, dedupe=True)

    def tap_output_aiter(
        self, run_id: UUID, output: AsyncIterator[T]
    ) -> AsyncIterator[T]:
        return output

    def tap_output_iter(self, run_id: UUID, output: Iterator[T]) -> Iterator[T]:
        return output

    def on_chat_model_start(
        self,
        serialized: dict[str, Any],
        messages: list[list[BaseMessage]],
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        tags: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> Any:
        if metadata and (not tags or (TAG_NOSTREAM not in tags)):
            ns = tuple(cast(str, metadata["langgraph_checkpoint_ns"]).split(NS_SEP))[
                :-1
            ]
            if not self.subgraphs and len(ns) > 0 and ns != self.parent_ns:
                return
            if (filtered_tags := filter_to_user_tags(tags)) is not None:
                metadata["tags"] = filtered_tags
            self.metadata[run_id] = (ns, metadata)
        # Clean up any stale buffer from a previous invocation for the same run_id.
        self._tc_buffer.pop(run_id, None)
        self._tc_seen.discard(run_id)

    def on_llm_new_token(
        self,
        token: str,
        *,
        chunk: ChatGenerationChunk | None = None,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        tags: list[str] | None = None,
        **kwargs: Any,
    ) -> Any:
        if not isinstance(chunk, ChatGenerationChunk):
            return
        if meta := self.metadata.get(run_id):
            msg = chunk.message

            # If this chunk carries tool_call_chunks, the provider's streaming
            # parser correctly classified it as a tool call.  Discard any
            # previously-buffered content that was likely stray preamble text.
            if msg.tool_call_chunks:
                self._tc_seen.add(run_id)
                self._tc_buffer.pop(run_id, None)
                self._emit(meta, msg)
                return

            content = msg.content
            if isinstance(content, str) and content:
                if _is_tool_call_like_content(content):
                    # Buffer this content — it may be leaked tool-call syntax.
                    self._tc_buffer.setdefault(run_id, []).append(content)
                else:
                    # Flush any previously buffered content before emitting
                    # this chunk, so the ordering is preserved.
                    if run_id in self._tc_buffer:
                        self._flush_tc_buffer(meta, msg, run_id)
                    self._emit(meta, msg)
            elif isinstance(content, list):
                # Multi-modal content — emit as-is (cannot be a tool-call leak).
                self._emit(meta, msg)

    def on_llm_end(
        self,
        response: LLMResult,
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        **kwargs: Any,
    ) -> Any:
        if meta := self.metadata.get(run_id):
            if response.generations and response.generations[0]:
                gen = response.generations[0][0]
                if isinstance(gen, ChatGeneration):
                    final_msg = gen.message
                    has_tool_calls = bool(
                        final_msg.tool_calls or final_msg.invalid_tool_calls
                    )
                    if has_tool_calls and run_id in self._tc_buffer:
                        # The final message contains tool calls, so any
                        # buffered content was leaked tool-call syntax that
                        # the streaming parser failed to classify.  Discard
                        # the buffer and strip the leaked content from the
                        # finalized message so it does not appear in
                        # conversation history.
                        self._tc_buffer.pop(run_id, None)
                        self._emit(meta, final_msg, dedupe=True)
                    elif run_id in self._tc_buffer:
                        # No tool calls in the final message — the buffered
                        # content was genuine assistant text (e.g. the model
                        # chose to output JSON as prose).  Flush the buffer
                        # so consumers see the complete text.
                        self._flush_tc_buffer(meta, final_msg, run_id)
                        self._emit(meta, final_msg, dedupe=True)
                    else:
                        self._emit(meta, final_msg, dedupe=True)
        self._tc_buffer.pop(run_id, None)
        self._tc_seen.discard(run_id)
        self.metadata.pop(run_id, None)

    def on_llm_error(
        self,
        error: BaseException,
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        **kwargs: Any,
    ) -> Any:
        self._tc_buffer.pop(run_id, None)
        self._tc_seen.discard(run_id)
        self.metadata.pop(run_id, None)

    def on_chain_start(
        self,
        serialized: dict[str, Any],
        inputs: dict[str, Any],
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        tags: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> Any:
        if (
            metadata
            and kwargs.get("name") == metadata.get("langgraph_node")
            and (not tags or TAG_HIDDEN not in tags)
        ):
            ns = tuple(cast(str, metadata["langgraph_checkpoint_ns"]).split(NS_SEP))[
                :-1
            ]
            if not self.subgraphs and len(ns) > 0:
                return
            self.metadata[run_id] = (ns, metadata)
            for value in _state_values(inputs):
                if isinstance(value, BaseMessage):
                    if value.id is not None:
                        self.seen.add(value.id)
                elif isinstance(value, Sequence) and not isinstance(value, str):
                    for item in value:
                        if isinstance(item, BaseMessage):
                            if item.id is not None:
                                self.seen.add(item.id)

    def on_chain_end(
        self,
        response: Any,
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        **kwargs: Any,
    ) -> Any:
        if meta := self.metadata.pop(run_id, None):
            # Handle Command node updates
            if isinstance(response, Command):
                self._find_and_emit_messages(meta, response.update)
            # Handle list of Command updates
            elif isinstance(response, Sequence) and any(
                isinstance(value, Command) for value in response
            ):
                for value in response:
                    if isinstance(value, Command):
                        self._find_and_emit_messages(meta, value.update)
                    else:
                        self._find_and_emit_messages(meta, value)
            # Handle basic updates / streaming
            else:
                self._find_and_emit_messages(meta, response)

    def on_chain_error(
        self,
        error: BaseException,
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        **kwargs: Any,
    ) -> Any:
        self.metadata.pop(run_id, None)


class StreamMessagesHandlerV2(StreamMessagesHandler, _V2StreamingCallbackHandler):
    """v2 variant of `StreamMessagesHandler`.

    Declaring `_V2StreamingCallbackHandler` as a base flips
    `BaseChatModel.invoke` to route through `_stream_chat_model_events`
    (firing `on_stream_event`) instead of `_stream` (firing
    `on_llm_new_token`). Inherits `on_stream_event` from the parent,
    which forwards protocol events onto the messages stream channel.

    Pregel attaches this class instead of the v1 handler only when
    `StreamingHandler` opts in via the internal
    `CONFIG_KEY_STREAM_MESSAGES_V2` config key; direct
    `graph.stream(stream_mode="messages")` callers keep the v1
    AIMessageChunk shape.
    """

    def on_llm_new_token(
        self,
        token: str,
        *,
        chunk: ChatGenerationChunk | None = None,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        tags: list[str] | None = None,
        **kwargs: Any,
    ) -> Any:
        """Intentional no-op — v1 chunks are not used on v2-flagged runs.

        The v2 marker already steers `invoke` to the event generator, so
        `on_llm_new_token` should not fire under normal routing. This
        override stays a pass-through (no call to `super()`) to make
        the intent explicit and to guard against any caller (e.g. a
        node that calls `model.stream()` directly, which still fires
        the v1 callback) leaking AIMessageChunks onto a v2-flagged
        messages stream.
        """
        # Intentionally empty: v2 handler does not forward v1 chunks.

    def __init__(
        self,
        stream: Callable[[StreamChunk], None],
        subgraphs: bool,
        *,
        parent_ns: tuple[str, ...] | None = None,
    ) -> None:
        super().__init__(stream, subgraphs, parent_ns=parent_ns)
        self._streamed_run_ids: set[UUID] = set()

    def _find_and_emit_messages(self, meta: Meta, response: Any) -> None:
        """Like the v1 handler, but skip ToolMessage from node outputs.

        Tool results belong on the tools channel / state in v3; v2-flagged streams
        must not replay finalized ToolMessages as chat tokens (see MessagesTransformer).
        Legacy v1-only `stream_mode="messages"` still emits ToolMessages (see subgraph
        streaming tests).
        """
        if isinstance(response, BaseMessage) and not isinstance(response, ToolMessage):
            self._emit(meta, response, dedupe=True)
        elif isinstance(response, Sequence):
            for value in response:
                if isinstance(value, BaseMessage) and not isinstance(
                    value, ToolMessage
                ):
                    self._emit(meta, value, dedupe=True)
        else:
            for value in _state_values(response):
                if isinstance(value, BaseMessage) and not isinstance(
                    value, ToolMessage
                ):
                    self._emit(meta, value, dedupe=True)
                elif isinstance(value, Sequence):
                    for item in value:
                        if isinstance(item, BaseMessage) and not isinstance(
                            item, ToolMessage
                        ):
                            self._emit(meta, item, dedupe=True)

    def on_llm_end(
        self,
        response: LLMResult,
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        **kwargs: Any,
    ) -> Any:
        if meta := self.metadata.get(run_id):
            if response.generations and response.generations[0]:
                gen = response.generations[0][0]
                if isinstance(gen, ChatGeneration):
                    final_msg = gen.message
                    has_tool_calls = bool(
                        final_msg.tool_calls or final_msg.invalid_tool_calls
                    )
                    if has_tool_calls and run_id in self._tc_buffer:
                        # Discard buffered content that was leaked tool-call
                        # syntax; the final message carries the real tool calls.
                        self._tc_buffer.pop(run_id, None)
                    elif run_id in self._tc_buffer:
                        # No tool calls — flush buffered content as text.
                        self._flush_tc_buffer(meta, final_msg, run_id)

                    if run_id in self._streamed_run_ids:
                        if final_msg.id is None:
                            final_msg.id = str(uuid4())
                        self.seen.add(final_msg.id)
                    else:
                        self._emit(meta, final_msg, dedupe=True)
        self._tc_buffer.pop(run_id, None)
        self._tc_seen.discard(run_id)
        self._streamed_run_ids.discard(run_id)
        self.metadata.pop(run_id, None)

    def on_llm_error(
        self,
        error: BaseException,
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        **kwargs: Any,
    ) -> Any:
        self._tc_buffer.pop(run_id, None)
        self._tc_seen.discard(run_id)
        self._streamed_run_ids.discard(run_id)
        super().on_llm_error(
            error,
            run_id=run_id,
            parent_run_id=parent_run_id,
            **kwargs,
        )

    def on_stream_event(
        self,
        event: dict[str, Any],
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        tags: list[str] | None = None,
        **kwargs: Any,
    ) -> Any:
        """Forward a protocol event from `stream_events(version="v3")` as a messages stream part.

        Fires once per `MessagesData` event (`message-start`, per-block
        `content-block-*`, `message-finish`). The transformer layer
        correlates events back to a single `ChatModelStream` via
        `metadata["run_id"]` — attached here so the v1
        `stream_mode="messages"` output (which emits
        `(AIMessageChunk, metadata)` via `on_llm_new_token`) keeps its
        original metadata shape.

        Lives on the v2 handler rather than the v1 base: content-block
        events are a v2-only concept, and forwarding them only when the
        v2 handler is attached keeps the message channel's shape
        predictable for v1 callers.
        """
        if meta := self.metadata.get(run_id):
            # Record message_id on message-start so on_chain_end's
            # dedupe skips the finalized AIMessage the node returns
            # (otherwise the messages projection double-counts: once
            # from streaming, once from the chain output).
            if event.get("event") == "message-start":
                self._streamed_run_ids.add(run_id)
                msg_id = event.get("message_id")
                if msg_id:
                    self.seen.add(msg_id)
            v2_meta = {**meta[1], "run_id": str(run_id)}
            self.stream((meta[0], "messages", (event, v2_meta)))


# Known role values (OpenAI-style) and type values (LangChain serialisation)
# that identify a dict as a message. Checked before coercing to BaseMessage so
# we don't accidentally touch unrelated dicts that happen to have a "role" key.
_MESSAGE_ROLES: frozenset[str] = frozenset(
    {"user", "human", "assistant", "ai", "tool", "system", "function"}
)
_MESSAGE_TYPES: frozenset[str] = frozenset(
    {"human", "ai", "tool", "system", "function", "remove"}
)


def _is_message_dict(item: dict) -> bool:
    return item.get("role") in _MESSAGE_ROLES or item.get("type") in _MESSAGE_TYPES


def ensure_message_ids(value: Any) -> None:
    """Coerce message-like write values to typed BaseMessages with stable IDs.

    Called in put_writes() before DeltaChannel writes are submitted to the
    checkpointer. Without this the checkpoint may store raw dicts or id=None
    BaseMessages; every get_state() replay then produces a different UUID and
    the same message appears with a different ID in each LangSmith trace.

    Handles three input shapes:
    - BaseMessage objects: assign a UUID if id is None.
    - Dicts with a known "role" (OpenAI-style) or "type" (LangChain format) at
      the root level: stamp "id" into the dict in-place. The reducer's
      convert_to_messages call will forward the id to the resulting BaseMessage.
    - Lists of the above: apply the same logic to each element, replacing dict
      items with coerced BaseMessages so the shared list reference seen by
      checkpoint_pending_writes and the background thread both get typed messages.

    Mutating synchronously here (before the background thread is submitted) is
    safe: the serialised bytes always reflect the post-coercion state.
    """
    if isinstance(value, BaseMessage):
        if value.id is None:
            value.id = str(uuid4())
    elif isinstance(value, dict) and _is_message_dict(value):
        if not value.get("id"):
            value["id"] = str(uuid4())
    elif isinstance(value, list):
        for i, item in enumerate(value):
            if isinstance(item, BaseMessage):
                if item.id is None:
                    item.id = str(uuid4())
            elif isinstance(item, dict) and _is_message_dict(item):
                msg = convert_to_messages([item])[0]
                if msg.id is None:
                    msg.id = str(uuid4())
                value[i] = msg
