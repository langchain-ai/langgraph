from __future__ import annotations

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
_USAGE_DETAIL_KEYS = ("input_token_details", "output_token_details")


def _state_values(obj: Any) -> Sequence[Any]:
    """Extract top-level field values from a state object (dict, BaseModel, or dataclass)."""
    if isinstance(obj, dict):
        return list(obj.values())
    elif isinstance(obj, BaseModel):
        return [getattr(obj, k) for k in type(obj).model_fields]
    elif is_dataclass(obj) and not isinstance(obj, type):
        return [getattr(obj, f.name) for f in fields(obj)]
    return ()


def _usage_detail_fields(usage: Any) -> dict[str, dict[str, Any]]:
    """Return usage detail dictionaries that are lost by older protocol bridges."""
    if not isinstance(usage, dict):
        return {}
    return {
        key: dict(value)
        for key in _USAGE_DETAIL_KEYS
        if isinstance((value := usage.get(key)), dict)
    }


def _merge_usage_details(
    usage: Any, details: dict[str, dict[str, Any]]
) -> dict[str, Any] | None:
    if not details:
        return cast(dict[str, Any] | None, usage if isinstance(usage, dict) else None)

    merged: dict[str, Any] = dict(usage) if isinstance(usage, dict) else {}
    for key, value in details.items():
        existing = merged.get(key)
        if isinstance(existing, dict):
            merged[key] = {**existing, **value}
        else:
            merged[key] = dict(value)
    return merged


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
        if dedupe and message.id in self.seen:
            return
        else:
            if message.id is None:
                message.id = str(uuid4())
            self.seen.add(message.id)
            self.stream((meta[0], "messages", (message, meta[1])))

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

    def __init__(
        self,
        stream: Callable[[StreamChunk], None],
        subgraphs: bool,
        *,
        parent_ns: tuple[str, ...] | None = None,
    ) -> None:
        super().__init__(stream, subgraphs, parent_ns=parent_ns)
        self._streamed_run_ids: set[UUID] = set()
        self._usage_details_by_run: dict[UUID, dict[str, dict[str, Any]]] = {}

    def _capture_usage_details(
        self, run_id: UUID, usage: Any
    ) -> dict[str, dict[str, Any]]:
        details = _usage_detail_fields(usage)
        if not details:
            return {}
        stored = self._usage_details_by_run.setdefault(run_id, {})
        for key, value in details.items():
            if key in stored:
                stored[key].update(value)
            else:
                stored[key] = dict(value)
        return stored

    def _patch_usage_details(
        self, run_id: UUID, usage: Any
    ) -> dict[str, Any] | None:
        details = self._usage_details_by_run.get(run_id)
        if not details:
            return cast(dict[str, Any] | None, usage if isinstance(usage, dict) else None)
        return _merge_usage_details(usage, details)

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
        """Capture v1 chunk usage details without emitting v1 chunks.

        The langchain-core compat bridge converts chunks to v3 protocol events,
        but older versions narrowed `usage_metadata` before the final
        `message-finish` event. Keep the detail dictionaries so the finalized
        message exposed through `on_llm_end` remains faithful to v2.
        """
        if isinstance(chunk, ChatGenerationChunk):
            self._capture_usage_details(run_id, chunk.message.usage_metadata)

    def on_llm_end(
        self,
        response: LLMResult,
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        **kwargs: Any,
    ) -> Any:
        if response.generations and response.generations[0]:
            gen = response.generations[0][0]
            if isinstance(gen, ChatGeneration):
                patched_usage = self._patch_usage_details(
                    run_id, gen.message.usage_metadata
                )
                if patched_usage is not None:
                    gen.message.usage_metadata = patched_usage
                if meta := self.metadata.get(run_id):
                    if run_id in self._streamed_run_ids:
                        if gen.message.id is None:
                            gen.message.id = str(uuid4())
                        self.seen.add(gen.message.id)
                    else:
                        self._emit(meta, gen.message, dedupe=True)
        self._streamed_run_ids.discard(run_id)
        self._usage_details_by_run.pop(run_id, None)
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
        self._usage_details_by_run.pop(run_id, None)
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
            elif event.get("event") == "message-finish":
                patched_usage = self._patch_usage_details(run_id, event.get("usage"))
                if patched_usage is not None:
                    event["usage"] = patched_usage
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
