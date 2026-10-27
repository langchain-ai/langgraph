from __future__ import annotations

from collections.abc import (
    AsyncIterator,
    Callable,
    Iterator,
    Mapping,
    MutableSequence,
    Sequence,
)
from dataclasses import fields, is_dataclass
from typing import (
    Any,
    TypeVar,
    cast,
)
from uuid import UUID, uuid4

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.messages import AnyMessage, BaseMessage, RemoveMessage, ToolMessage
from langchain_core.messages.utils import convert_to_messages
from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, LLMResult
from pydantic import BaseModel

from langgraph._internal._config import filter_to_user_tags
from langgraph._internal._constants import NS_SEP
from langgraph.channels.binop import _get_overwrite
from langgraph.channels.delta import DeltaChannel
from langgraph.constants import TAG_HIDDEN, TAG_NOSTREAM
from langgraph.pregel.protocol import StreamChunk
from langgraph.types import Command, Overwrite

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
            if not self.subgraphs and len(ns) > 0 and ns != self.parent_ns:
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
                    if run_id in self._streamed_run_ids:
                        if gen.message.id is None:
                            gen.message.id = str(uuid4())
                        self.seen.add(gen.message.id)
                    else:
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


# Defined here rather than in `langgraph.graph.message`, which re-exports it:
# `ensure_message_ids` has to recognize it, and importing that module from the
# loop would be an import cycle.
def _messages_delta_reducer(
    state: list[AnyMessage], writes: list[list[AnyMessage]]
) -> list[AnyMessage]:
    """**Experimental.** Batch reducer for use with `DeltaChannel`.

    Processes all writes in one pass — dedup by ID, `RemoveMessage`
    tombstoning — without calling `add_messages`.

    This reducer is batching-invariant, as required by `DeltaChannel`:
    `reducer(reducer(state, xs), ys) == reducer(state, xs + ys)`.

    Raw dict / string / tuple inputs are coerced to typed `BaseMessage`
    objects so that HTTP-driven graphs work without a separate coercion
    step. This is not full `add_messages` parity — `REMOVE_ALL_MESSAGES`,
    unknown-id `RemoveMessage` errors, missing-id UUID assignment, and
    `BaseMessageChunk` conversion are not handled here.

    Example::

        from typing import Annotated
        from langgraph.channels.delta import DeltaChannel
        from langgraph.graph.message import _messages_delta_reducer

        class State(TypedDict):
            messages: Annotated[list, DeltaChannel(_messages_delta_reducer)]
    """

    # Each write is either a list of message-likes or a single message-like
    # (BaseMessage / dict / str / tuple). Only lists flatten; everything
    # else is one message.
    flat: list[Any] = []
    for w in writes:
        if isinstance(w, list):
            flat.extend(w)
        else:
            flat.append(w)
    # Steady state: the reducer's own output is already typed, so skip
    # `convert_to_messages` on state when the first element is a BaseMessage.
    # Only raw input (initial dicts, deserialized blobs) hits the slow path.
    if state and isinstance(state[0], BaseMessage):
        state_msgs = state
    else:
        state_msgs = cast("list[AnyMessage]", convert_to_messages(state))
    msgs = cast("list[AnyMessage]", convert_to_messages(flat))

    index: dict[str, int] = {
        m.id: i for i, m in enumerate(state_msgs) if m.id is not None
    }
    result: list[AnyMessage | None] = list(state_msgs)
    for msg in msgs:
        mid = msg.id
        if mid is None:
            result.append(msg)
        elif isinstance(msg, RemoveMessage):
            if mid in index:
                result[index[mid]] = None
                del index[mid]
        elif mid in index:
            result[index[mid]] = msg
        else:
            index[mid] = len(result)
            result.append(msg)
    return [m for m in result if m is not None]


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


def ensure_message_ids(
    channels: Mapping[str, Any], writes: Sequence[tuple[str, Any]]
) -> None:
    """Give the messages written to DeltaChannels stable ids.

    A `DeltaChannel` replays its saved writes on every read, and its reducer
    may not assign missing ids (`_messages_delta_reducer` doesn't), so a
    message saved without an id comes back without one, or with a new one on
    every read, and nothing can target it by id, for example `RemoveMessage`.

    In a channel whose reducer is `_messages_delta_reducer`, every message-like
    it accepts (a message, a dict, a string, a `(role, content)` tuple; alone,
    in a list or in an `Overwrite`) becomes a message with an id. Other
    DeltaChannels may hold other data, so there only message objects get ids,
    plus message dicts outside an `Overwrite`: one in a list becomes a message.
    Anything that can't become a message is left as it is, so it fails where
    it always has: a message dict in a list raises here, anything else when
    the reducer reads it. An empty id counts as missing.

    A write that changes is replaced in `writes`: lists and dicts the caller
    passed in are never changed, only the ids of message objects.
    """
    for i, (ch, value) in enumerate(writes):
        spec = channels.get(ch)
        if not isinstance(spec, DeltaChannel):
            continue
        if spec.reducer is _messages_delta_reducer:
            new_value = _with_message_ids(value)
        else:
            new_value = _with_base_message_ids(value)
        if new_value is not value:
            cast(MutableSequence[tuple[str, Any]], writes)[i] = (ch, new_value)


def _with_message_ids(value: Any) -> Any:
    is_overwrite, overwrite_value = _get_overwrite(value)
    if not is_overwrite:
        if isinstance(value, list):
            return _map_messages(value, _list_item_with_id)
        return _message_with_id(value)
    if isinstance(overwrite_value, (list, tuple)):
        items = _map_messages(overwrite_value, _message_with_id)
        if items is not overwrite_value:
            return Overwrite(items)
    return value


def _with_base_message_ids(value: Any) -> Any:
    is_overwrite, overwrite_value = _get_overwrite(value)
    if is_overwrite:
        if isinstance(overwrite_value, (list, tuple)):
            for item in overwrite_value:
                _stamp_message_id(item)
        return value
    if isinstance(value, list):
        return _map_messages(value, _base_list_item_with_id)
    if isinstance(value, dict) and _is_message_dict(value) and not value.get("id"):
        return {**value, "id": str(uuid4())}
    _stamp_message_id(value)
    return value


def _map_messages(items: list | tuple, fn: Callable[[Any], Any]) -> list | tuple:
    mapped = [fn(item) for item in items]
    if all(new is old for new, old in zip(mapped, items)):
        return items
    return mapped if isinstance(items, list) else tuple(mapped)


def _list_item_with_id(item: Any) -> Any:
    if isinstance(item, dict) and _is_message_dict(item):
        return _message_with_id(item, strict=True)
    return _message_with_id(item)


def _base_list_item_with_id(item: Any) -> Any:
    if isinstance(item, dict) and _is_message_dict(item):
        return _message_with_id(item, strict=True)
    _stamp_message_id(item)
    return item


def _message_with_id(item: Any, *, strict: bool = False) -> Any:
    if isinstance(item, BaseMessage):
        _stamp_message_id(item)
        return item
    try:
        message = convert_to_messages([item])[0]
    except Exception:
        if strict:
            raise
        return item
    if not message.id:
        message.id = str(uuid4())
    return message


def _stamp_message_id(item: Any) -> None:
    if isinstance(item, BaseMessage) and item.id is None:
        item.id = str(uuid4())
