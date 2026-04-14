"""Protocol-native content-block message handler for StreamingHandler.

Emits structured content-block lifecycle events (message-start,
content-block-start/delta/finish, message-finish) instead of raw
``(AIMessageChunk, metadata)`` tuples.  The existing
:class:`~langgraph.pregel._messages.StreamMessagesHandler` is NOT
modified — this handler is only activated when
``__protocol_messages_stream`` is ``True`` in the run's configurable.
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator, Callable, Iterator, Sequence
from dataclasses import dataclass, field
from typing import Any, TypeVar, cast
from uuid import UUID, uuid4

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.messages import AIMessageChunk, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, LLMResult

from langgraph._internal._constants import NS_SEP
from langgraph.constants import TAG_HIDDEN, TAG_NOSTREAM
from langgraph.pregel.protocol import StreamChunk
from langgraph.stream._types import (
    ContentBlockDeltaData,
    ContentBlockFinishData,
    ContentBlockStartData,
    FinishReason,
    InvalidToolCallBlock,
    MessageErrorData,
    MessageStartData,
    ReasoningBlock,
    TextBlock,
    ToolCallBlock,
    UsageInfo,
)

try:
    from langchain_core.tracers._streaming import _StreamingCallbackHandler
except ImportError:
    _StreamingCallbackHandler = object  # type: ignore

T = TypeVar("T")
Meta = tuple[tuple[str, ...], dict[str, Any]]

PROTOCOL_MESSAGES_STREAM_KEY = "__protocol_messages_stream"

# ---------------------------------------------------------------------------
# Content-block accumulation helpers
# ---------------------------------------------------------------------------

# A "compatible content block" is a dict matching one of the protocol block
# TypedDicts (TextBlock, ReasoningBlock, ToolCallChunkBlock, etc.).
CompatBlock = dict[str, Any]


@dataclass
class _ProtocolRunState:
    """Per-run state for tracking the active message lifecycle."""

    message_id: str | None = None
    started: bool = False
    blocks: dict[int, CompatBlock] = field(default_factory=dict)
    usage: dict[str, Any] | None = None


def _accumulate_block(accumulated: CompatBlock, delta: CompatBlock) -> CompatBlock:
    """Merge *delta* into *accumulated*, returning the updated block."""
    btype = accumulated.get("type", "text")
    if btype == "text" and delta.get("type", "text") == "text":
        accumulated["text"] = accumulated.get("text", "") + delta.get("text", "")
    elif btype == "reasoning" and delta.get("type") == "reasoning":
        accumulated["reasoning"] = accumulated.get("reasoning", "") + delta.get(
            "reasoning", ""
        )
    elif btype == "tool_call_chunk" and delta.get("type") == "tool_call_chunk":
        accumulated["args"] = accumulated.get("args", "") + delta.get("args", "")
        if delta.get("id") is not None:
            accumulated["id"] = delta["id"]
        if delta.get("name") is not None:
            accumulated["name"] = delta["name"]
    return accumulated


def _delta_block(previous: CompatBlock, current: CompatBlock) -> CompatBlock | None:
    """Compute the delta between *previous* and *current*.

    Returns ``None`` if there is nothing new to emit.
    """
    btype = current.get("type", "text")
    if btype == "text":
        prev_text = previous.get("text", "")
        cur_text = current.get("text", "")
        delta_text = cur_text[len(prev_text) :]
        if not delta_text:
            return None
        return TextBlock(type="text", text=delta_text)
    elif btype == "reasoning":
        prev_r = previous.get("reasoning", "")
        cur_r = current.get("reasoning", "")
        delta_r = cur_r[len(prev_r) :]
        if not delta_r:
            return None
        return ReasoningBlock(type="reasoning", reasoning=delta_r)
    elif btype == "tool_call_chunk":
        prev_args = previous.get("args", "")
        cur_args = current.get("args", "")
        delta_args = cur_args[len(prev_args) :]
        has_meta = current.get("id") is not None or current.get("name") is not None
        if not delta_args and not has_meta:
            return None
        result: CompatBlock = {"type": "tool_call_chunk", "args": delta_args}
        if current.get("id") is not None and previous.get("id") is None:
            result["id"] = current["id"]
        if current.get("name") is not None and previous.get("name") is None:
            result["name"] = current["name"]
        return result
    # Unrecognized block type — pass through unchanged
    return current


def _finalize_block(block: CompatBlock) -> CompatBlock:
    """Convert a ``tool_call_chunk`` block to a finalized ``tool_call`` or
    ``invalid_tool_call`` block.  Other block types pass through unchanged.
    """
    if block.get("type") != "tool_call_chunk":
        return block
    raw_args = block.get("args", "{}")
    try:
        parsed_args = json.loads(raw_args) if raw_args else {}
        return ToolCallBlock(
            type="tool_call",
            id=block.get("id", ""),
            name=block.get("name", ""),
            args=parsed_args,
        )
    except (json.JSONDecodeError, TypeError):
        return InvalidToolCallBlock(
            type="invalid_tool_call",
            id=block.get("id"),
            name=block.get("name"),
            args=raw_args,
            error="Failed to parse tool call arguments as JSON",
        )


def _normalize_finish_reason(value: Any) -> FinishReason:
    """Map provider-specific stop reasons to protocol finish reasons."""
    if value == "length":
        return "length"
    if value == "content_filter":
        return "content_filter"
    if value in ("tool_use", "tool_calls"):
        return "tool_use"
    # "end_turn", "stop", None, and anything else → "stop"
    return "stop"


def _accumulate_usage(
    current: dict[str, Any] | None, delta: Any
) -> dict[str, Any] | None:
    """Accumulate usage metadata from streamed chunks."""
    if not isinstance(delta, dict):
        return current
    if current is None:
        return dict(delta)
    for key in ("input_tokens", "output_tokens", "total_tokens", "cached_tokens"):
        if key in delta:
            current[key] = current.get(key, 0) + delta[key]
    # Merge detail dicts
    for detail_key in ("input_token_details", "output_token_details"):
        if detail_key in delta and isinstance(delta[detail_key], dict):
            if detail_key not in current:
                current[detail_key] = {}
            current[detail_key].update(delta[detail_key])
    return current


def _to_protocol_usage(usage: dict[str, Any] | None) -> UsageInfo | None:
    """Convert LangChain usage metadata to protocol ``UsageInfo``."""
    if usage is None:
        return None
    result: dict[str, Any] = {}
    if "input_tokens" in usage:
        result["input_tokens"] = usage["input_tokens"]
    if "output_tokens" in usage:
        result["output_tokens"] = usage["output_tokens"]
    if "total_tokens" in usage:
        result["total_tokens"] = usage["total_tokens"]
    if "cached_tokens" in usage:
        result["cached_tokens"] = usage["cached_tokens"]
    return UsageInfo(**result) if result else None


# ---------------------------------------------------------------------------
# Extracting content blocks from LangChain messages
# ---------------------------------------------------------------------------


def _extract_blocks_from_chunk(msg: AIMessageChunk) -> list[tuple[int, CompatBlock]]:
    """Extract ``(index, block)`` pairs from an ``AIMessageChunk``.

    LangChain stores content in several places:
    - ``content: str`` — a single text block at index 0
    - ``content: list[dict]`` — explicit content blocks with their own types
    - ``tool_call_chunks`` — separate list for streamed tool call deltas
    """
    blocks: list[tuple[int, CompatBlock]] = []
    content = msg.content
    if isinstance(content, str) and content:
        blocks.append((0, dict(TextBlock(type="text", text=content))))
    elif isinstance(content, list):
        for i, item in enumerate(content):
            if not isinstance(item, dict):
                continue
            ctype = item.get("type", "")
            if ctype == "text" and item.get("text"):
                blocks.append(
                    (
                        item.get("index", i),
                        dict(TextBlock(type="text", text=item["text"])),
                    )
                )
            elif ctype in ("reasoning_content", "reasoning", "thinking"):
                reasoning_text = (
                    item.get("reasoning_content")
                    or item.get("reasoning")
                    or item.get("thinking", "")
                )
                if reasoning_text:
                    blocks.append(
                        (
                            item.get("index", i),
                            dict(
                                ReasoningBlock(
                                    type="reasoning", reasoning=reasoning_text
                                )
                            ),
                        )
                    )

    # Tool call chunks live in a separate field
    for tc in msg.tool_call_chunks or []:
        idx = tc.get("index")
        if idx is None:
            # Assign indices after text content blocks
            idx = len(blocks)
        block: CompatBlock = {"type": "tool_call_chunk", "args": tc.get("args", "")}
        if tc.get("id") is not None:
            block["id"] = tc["id"]
        if tc.get("name") is not None:
            block["name"] = tc["name"]
        blocks.append((idx, block))

    return blocks


# ---------------------------------------------------------------------------
# The handler
# ---------------------------------------------------------------------------


class StreamProtocolMessagesHandler(BaseCallbackHandler, _StreamingCallbackHandler):
    """Callback handler that emits content-block protocol events.

    Activated when ``__protocol_messages_stream`` is ``True`` in the run's
    configurable metadata.  Emits ``StreamChunk`` tuples of the form
    ``(namespace, "messages", data)`` where *data* is one of the
    ``MessagesData`` event types (``message-start``, ``content-block-start``,
    etc.).
    """

    run_inline = True

    def __init__(
        self,
        stream: Callable[[StreamChunk], None],
        subgraphs: bool,
        *,
        parent_ns: tuple[str, ...] | None = None,
    ) -> None:
        self.stream = stream
        self.subgraphs = subgraphs
        self.parent_ns = parent_ns
        # Per-run metadata: run_id → (namespace, metadata_dict)
        self.metadata: dict[UUID, Meta] = {}
        # Per-run protocol state for streamed messages
        self.protocol_runs: dict[UUID, _ProtocolRunState] = {}
        # Stable message ID mapping: run_id → message_id
        self.stable_message_ids: dict[UUID, str] = {}
        # Seen message IDs for deduplication of chain-emitted messages
        self.seen: set[str | int] = set()

    def _emit(self, meta: Meta, data: Any) -> None:
        """Emit a protocol event as a StreamChunk.

        The node name from *meta* is embedded at ``"__node__"`` so the
        stream pump can lift it into ``params.node`` without changing the
        ``StreamChunk`` tuple shape.
        """
        node = meta[1].get("langgraph_node")
        if node and isinstance(data, dict):
            data = {**data, "__node__": node}
        self.stream((meta[0], "messages", data))

    # -- Chat model callbacks -----------------------------------------------

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
            if tags:
                if filtered := [t for t in tags if not t.startswith("seq:step")]:
                    metadata["tags"] = filtered
            self.metadata[run_id] = (ns, metadata)
            self.protocol_runs[run_id] = _ProtocolRunState()

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
        meta = self.metadata.get(run_id)
        if meta is None:
            return
        state = self.protocol_runs.get(run_id)
        if state is None:
            return

        msg = chunk.message
        if not isinstance(msg, AIMessageChunk):
            return

        # Emit message-start on first token
        if not state.started:
            message_id = self._normalize_message_id(msg, run_id)
            state.message_id = message_id
            state.started = True
            start_data = dict(
                MessageStartData(
                    event="message-start",
                    role="ai",
                )
            )
            if message_id:
                start_data["message_id"] = message_id
            self._emit(meta, start_data)

        # Extract content blocks from this chunk
        extracted = _extract_blocks_from_chunk(msg)
        for idx, delta_block in extracted:
            if idx not in state.blocks:
                # New block — emit content-block-start
                state.blocks[idx] = dict(delta_block)
                # Start block has empty content placeholder
                start_block = _make_start_block(delta_block)
                self._emit(
                    meta,
                    ContentBlockStartData(
                        event="content-block-start",
                        index=idx,
                        content_block=start_block,
                    ),
                )
                # Then emit the first delta
                first_delta = _delta_block(
                    _make_start_block(delta_block), state.blocks[idx]
                )
                if first_delta is not None:
                    self._emit(
                        meta,
                        ContentBlockDeltaData(
                            event="content-block-delta",
                            index=idx,
                            content_block=first_delta,
                        ),
                    )
            else:
                # Existing block — compute delta, accumulate, emit
                previous = dict(state.blocks[idx])
                state.blocks[idx] = _accumulate_block(state.blocks[idx], delta_block)
                delta = _delta_block(previous, state.blocks[idx])
                if delta is not None:
                    self._emit(
                        meta,
                        ContentBlockDeltaData(
                            event="content-block-delta",
                            index=idx,
                            content_block=delta,
                        ),
                    )

        # Accumulate usage from chunk
        if msg.usage_metadata:
            state.usage = _accumulate_usage(state.usage, msg.usage_metadata)

    def on_llm_end(
        self,
        response: LLMResult,
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        **kwargs: Any,
    ) -> Any:
        meta = self.metadata.pop(run_id, None)
        state = self.protocol_runs.pop(run_id, None)
        if meta is None or state is None:
            return

        # Extract finish reason and usage from the final generation
        finish_reason: FinishReason = "stop"
        final_usage = state.usage

        if response.generations and response.generations[0]:
            gen = response.generations[0][0]
            if isinstance(gen, ChatGeneration):
                final_msg = gen.message
                # Get finish reason from response_metadata
                rm = getattr(final_msg, "response_metadata", {}) or {}
                raw_reason = rm.get("finish_reason") or rm.get("stop_reason")
                if raw_reason:
                    finish_reason = _normalize_finish_reason(raw_reason)
                # If we have tool calls in the final message, infer tool_use
                if (
                    finish_reason == "stop"
                    and hasattr(final_msg, "tool_calls")
                    and final_msg.tool_calls
                ):
                    finish_reason = "tool_use"
                # Get usage from final message if not accumulated from chunks
                if final_usage is None and hasattr(final_msg, "usage_metadata"):
                    final_usage = (
                        dict(final_msg.usage_metadata)
                        if final_msg.usage_metadata
                        else None
                    )

                # If we never got streaming tokens (non-streamed model call),
                # emit the full message lifecycle now
                if not state.started:
                    self._emit_full_message(meta, final_msg, finish_reason, final_usage)
                    return

        # Close out any open content blocks
        for idx in sorted(state.blocks):
            finalized = _finalize_block(state.blocks[idx])
            self._emit(
                meta,
                ContentBlockFinishData(
                    event="content-block-finish",
                    index=idx,
                    content_block=finalized,
                ),
            )

        # Emit message-finish
        finish_data: dict[str, Any] = {
            "event": "message-finish",
            "reason": finish_reason,
        }
        usage_info = _to_protocol_usage(final_usage)
        if usage_info is not None:
            finish_data["usage"] = usage_info
        self._emit(meta, finish_data)

        # Track the message as seen for dedup
        if state.message_id:
            self.seen.add(state.message_id)

    def on_llm_error(
        self,
        error: BaseException,
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        **kwargs: Any,
    ) -> Any:
        meta = self.metadata.pop(run_id, None)
        state = self.protocol_runs.pop(run_id, None)
        self.stable_message_ids.pop(run_id, None)
        if meta is None or state is None:
            return
        if state.started:
            self._emit(
                meta,
                MessageErrorData(
                    event="error",
                    message=str(error),
                ),
            )

    # -- Chain callbacks (for node-level message dedup) ---------------------

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
            # Record input message IDs for deduplication
            self._record_seen_messages(inputs)

    def on_chain_end(
        self,
        response: Any,
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        **kwargs: Any,
    ) -> Any:
        meta = self.metadata.pop(run_id, None)
        if meta is None:
            return
        # Emit protocol events for any new messages in the node's output
        self._emit_chain_messages(meta, response)

    def on_chain_error(
        self,
        error: BaseException,
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        **kwargs: Any,
    ) -> Any:
        self.metadata.pop(run_id, None)

    # -- Iterator taps (required by _StreamingCallbackHandler) ---------------

    def tap_output_aiter(
        self, run_id: UUID, output: AsyncIterator[T]
    ) -> AsyncIterator[T]:
        return output

    def tap_output_iter(self, run_id: UUID, output: Iterator[T]) -> Iterator[T]:
        return output

    # -- Internal helpers ---------------------------------------------------

    def _normalize_message_id(self, msg: BaseMessage, run_id: UUID) -> str | None:
        """Return a stable message ID for this run, creating one if needed."""
        msg_id = msg.id
        if msg_id is None:
            msg_id = self.stable_message_ids.get(run_id)
        if msg_id is None:
            msg_id = f"run-{run_id}"
        self.stable_message_ids[run_id] = msg_id
        # Mutate the message for consistency downstream
        if msg.id != msg_id:
            msg.id = msg_id
        return msg_id

    def _emit_full_message(
        self,
        meta: Meta,
        msg: BaseMessage,
        finish_reason: FinishReason,
        usage: dict[str, Any] | None,
        role: str = "ai",
    ) -> None:
        """Emit a complete message lifecycle for a non-streamed model call."""
        message_id = msg.id or str(uuid4())
        if message_id in self.seen:
            return
        self.seen.add(message_id)

        # message-start
        start_data = dict(
            MessageStartData(
                event="message-start",
                role=role,
            )
        )
        start_data["message_id"] = message_id
        self._emit(meta, start_data)

        # Extract all blocks from the final message
        blocks = _extract_final_blocks(msg)
        for idx, block in blocks:
            # content-block-start with the full content
            self._emit(
                meta,
                ContentBlockStartData(
                    event="content-block-start",
                    index=idx,
                    content_block=_make_start_block(block),
                ),
            )
            # content-block-delta with the full content
            delta = _delta_block(_make_start_block(block), block)
            if delta is not None:
                self._emit(
                    meta,
                    ContentBlockDeltaData(
                        event="content-block-delta",
                        index=idx,
                        content_block=delta,
                    ),
                )
            # content-block-finish
            finalized = _finalize_block(block)
            self._emit(
                meta,
                ContentBlockFinishData(
                    event="content-block-finish",
                    index=idx,
                    content_block=finalized,
                ),
            )

        # message-finish
        finish_data: dict[str, Any] = {
            "event": "message-finish",
            "reason": finish_reason,
        }
        usage_info = _to_protocol_usage(usage)
        if usage_info is not None:
            finish_data["usage"] = usage_info
        self._emit(meta, finish_data)

    def _record_seen_messages(self, obj: Any) -> None:
        """Record message IDs from node inputs for deduplication."""
        if isinstance(obj, BaseMessage):
            if obj.id is not None:
                self.seen.add(obj.id)
        elif isinstance(obj, dict):
            for value in obj.values():
                self._record_seen_messages(value)
        elif isinstance(obj, Sequence) and not isinstance(obj, (str, bytes)):
            for item in obj:
                self._record_seen_messages(item)

    def _emit_chain_messages(self, meta: Meta, response: Any) -> None:
        """Emit protocol events for messages found in chain output."""
        from langgraph.types import Command

        if isinstance(response, Command):
            self._emit_chain_messages(meta, response.update)
        elif isinstance(response, BaseMessage):
            self._emit_message_from_chain(meta, response)
        elif isinstance(response, Sequence) and not isinstance(response, (str, bytes)):
            for item in response:
                if isinstance(item, Command):
                    self._emit_chain_messages(meta, item.update)
                elif isinstance(item, BaseMessage):
                    self._emit_message_from_chain(meta, item)
        elif isinstance(response, dict):
            for value in response.values():
                if isinstance(value, BaseMessage):
                    self._emit_message_from_chain(meta, value)
                elif isinstance(value, Sequence) and not isinstance(
                    value, (str, bytes)
                ):
                    for item in value:
                        if isinstance(item, BaseMessage):
                            self._emit_message_from_chain(meta, item)

    def _emit_message_from_chain(self, meta: Meta, msg: BaseMessage) -> None:
        """Emit a full message lifecycle for a message from a chain output,
        deduplicating against previously-seen messages."""
        if msg.id is not None and msg.id in self.seen:
            return
        if msg.id is None:
            msg.id = str(uuid4())

        # Determine role and finish reason
        role = "ai"
        if hasattr(msg, "type"):
            if msg.type == "human":
                role = "human"
            elif msg.type == "system":
                role = "system"

        finish_reason: FinishReason = "stop"
        rm = getattr(msg, "response_metadata", {}) or {}
        raw_reason = rm.get("finish_reason") or rm.get("stop_reason")
        if raw_reason:
            finish_reason = _normalize_finish_reason(raw_reason)
        if finish_reason == "stop" and hasattr(msg, "tool_calls") and msg.tool_calls:
            finish_reason = "tool_use"

        raw_usage = getattr(msg, "usage_metadata", None)
        usage = dict(raw_usage) if raw_usage else None

        self._emit_full_message(meta, msg, finish_reason, usage, role=role)


# ---------------------------------------------------------------------------
# Block extraction for finalized (non-streamed) messages
# ---------------------------------------------------------------------------


def _extract_final_blocks(msg: BaseMessage) -> list[tuple[int, CompatBlock]]:
    """Extract ``(index, block)`` pairs from a finalized ``AIMessage``."""
    blocks: list[tuple[int, CompatBlock]] = []
    content = msg.content

    if isinstance(content, str) and content:
        blocks.append((0, dict(TextBlock(type="text", text=content))))
    elif isinstance(content, list):
        for i, item in enumerate(content):
            if not isinstance(item, dict):
                continue
            ctype = item.get("type", "")
            if ctype == "text" and item.get("text"):
                blocks.append((i, dict(TextBlock(type="text", text=item["text"]))))
            elif ctype in ("reasoning_content", "reasoning", "thinking"):
                reasoning_text = (
                    item.get("reasoning_content")
                    or item.get("reasoning")
                    or item.get("thinking", "")
                )
                if reasoning_text:
                    blocks.append(
                        (
                            i,
                            dict(
                                ReasoningBlock(
                                    type="reasoning", reasoning=reasoning_text
                                )
                            ),
                        )
                    )

    # Finalized tool calls (already parsed, not chunks)
    for tc in getattr(msg, "tool_calls", None) or []:
        idx = len(blocks)
        blocks.append(
            (
                idx,
                dict(
                    ToolCallBlock(
                        type="tool_call",
                        id=tc.get("id", ""),
                        name=tc.get("name", ""),
                        args=tc.get("args", {}),
                    )
                ),
            )
        )

    return blocks


def _make_start_block(block: CompatBlock) -> CompatBlock:
    """Create an empty start placeholder for a content block."""
    btype = block.get("type", "text")
    if btype == "text":
        return TextBlock(type="text", text="")
    elif btype == "reasoning":
        return ReasoningBlock(type="reasoning", reasoning="")
    elif btype == "tool_call_chunk":
        result: CompatBlock = {"type": "tool_call_chunk", "args": ""}
        if "id" in block:
            result["id"] = block["id"]
        if "name" in block:
            result["name"] = block["name"]
        return result
    elif btype == "tool_call":
        # Already finalized — return as-is for start event
        return ToolCallBlock(
            type="tool_call",
            id=block.get("id", ""),
            name=block.get("name", ""),
            args=block.get("args", {}),
        )
    return dict(block)


__all__ = ["PROTOCOL_MESSAGES_STREAM_KEY", "StreamProtocolMessagesHandler"]
