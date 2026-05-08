from __future__ import annotations

import hashlib
import json
import logging
import re
import time
from collections.abc import AsyncIterator, Callable, Iterator, Sequence
from dataclasses import fields, is_dataclass
from typing import (
    Any,
    TypeVar,
    cast,
)
from uuid import UUID, uuid4

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.messages import BaseMessage
from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, LLMResult
from pydantic import BaseModel

from langgraph._internal._constants import NS_SEP
from langgraph.constants import TAG_HIDDEN, TAG_NOSTREAM
from langgraph.pregel.protocol import StreamChunk
from langgraph.types import Command

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

_audit_logger = logging.getLogger("langgraph.pregel.messages.audit")

# Allowlist of metadata keys permitted to be forwarded to stream consumers.
_METADATA_ALLOWLIST: frozenset[str] = frozenset(
    {
        "langgraph_node",
        "langgraph_step",
        "langgraph_task_idx",
        "langgraph_triggers",
        "tags",
        "run_id",
    }
)

# Patterns that indicate dynamic code execution primitives in LLM output.
_DANGEROUS_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"\beval\s*\(", re.IGNORECASE),
    re.compile(r"\bexec\s*\(", re.IGNORECASE),
    re.compile(r"\bsubprocess\s*\.", re.IGNORECASE),
    re.compile(r"\bos\.system\s*\(", re.IGNORECASE),
    re.compile(r"\bos\.popen\s*\(", re.IGNORECASE),
    re.compile(r"\b__import__\s*\(", re.IGNORECASE),
    re.compile(r"\bcompile\s*\(", re.IGNORECASE),
    re.compile(r"\bexecfile\s*\(", re.IGNORECASE),
)


def _filter_metadata(metadata: dict[str, Any]) -> dict[str, Any]:
    """Return a copy of metadata containing only allowlisted keys."""
    return {k: v for k, v in metadata.items() if k in _METADATA_ALLOWLIST}


def _contains_dangerous_content(text: str) -> bool:
    """Return True if the text contains dynamic code execution primitives."""
    for pattern in _DANGEROUS_PATTERNS:
        if pattern.search(text):
            return True
    return False


def _validate_message_content(message: BaseMessage) -> bool:
    """Validate a message for dangerous content. Returns True if safe."""
    content = message.content
    if isinstance(content, str):
        if _contains_dangerous_content(content):
            _audit_logger.warning(
                "Blocked LLM output containing potential code execution primitive",
                extra={
                    "message_id": message.id,
                    "message_type": message.type,
                    "event": "llm_output_blocked",
                    "timestamp": time.time(),
                },
            )
            return False
    elif isinstance(content, list):
        for block in content:
            if isinstance(block, dict):
                block_text = block.get("text", "")
                if isinstance(block_text, str) and _contains_dangerous_content(block_text):
                    _audit_logger.warning(
                        "Blocked LLM output block containing potential code execution primitive",
                        extra={
                            "message_id": message.id,
                            "message_type": message.type,
                            "event": "llm_output_blocked",
                            "timestamp": time.time(),
                        },
                    )
                    return False
    return True


def _attach_provenance(message: BaseMessage, meta: Meta) -> None:
    """Attach synthetic content provenance labels to a message's additional_kwargs."""
    if not hasattr(message, "additional_kwargs"):
        return
    provenance: dict[str, Any] = {
        "ai_generated": True,
        "langgraph_node": meta[1].get("langgraph_node"),
        "langgraph_step": meta[1].get("langgraph_step"),
    }
    message.additional_kwargs.setdefault("__provenance__", provenance)


def _hash_content(content: Any) -> str:
    """Return a SHA-256 hex digest of the serialized content."""
    try:
        serialized = json.dumps(content, default=str, sort_keys=True)
    except Exception:
        serialized = str(content)
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def _state_values(obj: Any) -> Sequence[Any]:
    """Extract top-level field values from a state object (dict, BaseModel, or dataclass)."""
    if isinstance(obj, dict):
        return list(obj.values())
    elif isinstance(obj, BaseModel):
        return [getattr(obj, k) for k in type(obj).model_fields]
    elif is_dataclass(obj) and not isinstance(obj, type):
        return [getattr(obj, f.name) for f in fields(obj)]
    return ()


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

    def _emit(self, meta: Meta, message: BaseMessage, *, dedupe: bool = False) -> None:
        if not _validate_message_content(message):
            return
        if dedupe and message.id in self.seen:
            return
        else:
            if message.id is None:
                message.id = str(uuid4())
            self.seen.add(message.id)
            _attach_provenance(message, meta)
            filtered_meta = _filter_metadata(meta[1])
            filtered_meta["ai_generated"] = True
            _audit_logger.info(
                "Emitting AI-generated message",
                extra={
                    "event": "llm_message_emit",
                    "message_id": message.id,
                    "message_type": message.type,
                    "content_hash": _hash_content(message.content),
                    "langgraph_node": meta[1].get("langgraph_node"),
                    "langgraph_step": meta[1].get("langgraph_step"),
                    "timestamp": time.time(),
                },
            )
            self.stream((meta[0], "messages", (message, filtered_meta)))

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
            if tags:
                if filtered_tags := [t for t in tags if not t.startswith("seq:step")]:
                    metadata["tags"] = filtered_tags
            self.metadata[run_id] = (ns, metadata)
            model_id = serialized.get("name") or serialized.get("id", ["unknown"])
            if isinstance(model_id, list):
                model_id = model_id[-1] if model_id else "unknown"
            input_hashes = [
                _hash_content([m.content for m in msg_list])
                for msg_list in messages
            ]
            _audit_logger.info(
                "Chat model started",
                extra={
                    "event": "chat_model_start",
                    "run_id": str(run_id),
                    "parent_run_id": str(parent_run_id) if parent_run_id else None,
                    "model_id": model_id,
                    "langgraph_node": metadata.get("langgraph_node"),
                    "langgraph_step": metadata.get("langgraph_step"),
                    "input_hashes": input_hashes,
                    "timestamp": time.time(),
                },
            )

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
        if not _validate_message_content(chunk.message):
            return
        if meta := self.metadata.get(run_id):
            self._emit(meta, chunk.message)

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
                    if _validate_message_content(gen.message):
                        _audit_logger.info(
                            "LLM generation completed",
                            extra={
                                "event": "llm_end",
                                "run_id": str(run_id),
                                "parent_run_id": str(parent_run_id) if parent_run_id else None,
                                "message_id": gen.message.id,
                                "message_type": gen.message.type,
                                "content_hash": _hash_content(gen.message.content),
                                "langgraph_node": meta[1].get("langgraph_node"),
                                "langgraph_step": meta[1].get("langgraph_step"),
                                "timestamp": time.time(),
                            },
                        )
                        self._emit(meta, gen.message, dedupe=True)
        self.metadata.pop(run_id, None)

    def on_llm_error(
        self,
        error: BaseException,
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        **kwargs: Any,
    ) -> Any:
        _audit_logger.warning(
            "LLM error encountered",
            extra={
                "event": "llm_error",
                "run_id": str(run_id),
                "parent_run_id": str(parent_run_id) if parent_run_id else None,
                "error_type": type(error).__name__,
                "timestamp": time.time(),
            },
        )
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
            _audit_logger.info(
                "Chain started",
                extra={
                    "event": "chain_start",
                    "run_id": str(run_id),
                    "parent_run_id": str(parent_run_id) if parent_run_id else None,
                    "langgraph_node": metadata.get("langgraph_node"),
                    "langgraph_step": metadata.get("langgraph_step"),
                    "input_hash": _hash_content(inputs),
                    "timestamp": time.time(),
                },
            )
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
            _audit_logger.info(
                "Chain ended",
                extra={
                    "event": "chain_end",
                    "run_id": str(run_id),
                    "parent_run_id": str(parent_run_id) if parent_run_id else None,
                    "langgraph_node": meta[1].get("langgraph_node"),
                    "langgraph_step": meta[1].get("langgraph_step"),
                    "output_hash": _hash_content(str(type(response).__name__)),
                    "timestamp": time.time(),
                },
            )
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
        _audit_logger.warning(
            "Chain error encountered",
            extra={
                "event": "chain_error",
                "run_id": str(run_id),
                "parent_run_id": str(parent_run_id) if parent_run_id else None,
                "error_type": type(error).__name__,
                "timestamp": time.time(),
            },
        )
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
                    if run_id in self._streamed_run_ids:
                        if gen.message.id is None:
                            gen.message.id = str(uuid4())
                        self.seen.add(gen.message.id)
                    else:
                        if _validate_message_content(gen.message):
                            _audit_logger.info(
                                "LLM generation completed (v2)",
                                extra={
                                    "event": "llm_end_v2",
                                    "run_id": str(run_id),
                                    "parent_run_id": str(parent_run_id) if parent_run_id else None,
                                    "message_id": gen.message.id,
                                    "message_type": gen.message.type,
                                    "content_hash": _hash_content(gen.message.content),
                                    "langgraph_node": meta[1].get("langgraph_node"),
                                    "langgraph_step": meta[1].get("langgraph_step"),
                                    "timestamp": time.time(),
                                },
                            )
                            self._emit(meta, gen.message, dedupe=True)
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
            # Validate event content for dangerous patterns before forwarding.
            event_text = event.get("content", "") or event.get("text", "") or ""
            if isinstance(event_text, str) and _contains_dangerous_content(event_text):
                _audit_logger.warning(
                    "Blocked stream event containing potential code execution primitive",
                    extra={
                        "event_type": event.get("event"),
                        "run_id": str(run_id),
                        "event": "stream_event_blocked",
                        "timestamp": time.time(),
                    },
                )
                return
            # Record message_id on message-start so on_chain_end's
            # dedupe skips the finalized AIMessage the node returns
            # (otherwise the messages projection double-counts: once
            # from streaming, once from the chain output).
            if event.get("event") == "message-start":
                self._streamed_run_ids.add(run_id)
                msg_id = event.get("message_id")
                if msg_id:
                    self.seen.add(msg_id)
            filtered_meta = _filter_metadata(meta[1])
            filtered_meta["run_id"] = str(run_id)
            filtered_meta["ai_generated"] = True
            _audit_logger.info(
                "Forwarding stream event",
                extra={
                    "event": "stream_event_forward",
                    "run_id": str(run_id),
                    "parent_run_id": str(parent_run_id) if parent_run_id else None,
                    "stream_event_type": event.get("event"),
                    "langgraph_node": meta[1].get("langgraph_node"),
                    "langgraph_step": meta[1].get("langgraph_step"),
                    "timestamp": time.time(),
                },
            )
            self.stream((meta[0], "messages", (event, filtered_meta)))