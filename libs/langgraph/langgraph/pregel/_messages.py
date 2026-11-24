from __future__ import annotations

import copy
import uuid
from collections.abc import (
    AsyncIterator,
    Callable,
    Iterator,
    Mapping,
    MutableSequence,
    Sequence,
)
from dataclasses import fields, is_dataclass
from functools import cache, partial
from typing import (
    Any,
    TypeGuard,
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
from langgraph._internal._constants import NS_SEP, OVERWRITE, PUSH, ROOT
from langgraph.channels.binop import BinaryOperatorAggregate, _get_overwrite
from langgraph.channels.delta import DeltaChannel
from langgraph.constants import START, TAG_HIDDEN, TAG_NOSTREAM
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
    return [v for _, v in _state_items(obj)]


def _state_items(obj: object) -> Sequence[tuple[str, object]]:
    if isinstance(obj, dict):
        return list(obj.items())
    elif isinstance(obj, BaseModel):
        return [(k, getattr(obj, k)) for k in type(obj).model_fields]
    elif is_dataclass(obj) and not isinstance(obj, type):
        return [(f.name, getattr(obj, f.name)) for f in fields(obj)]
    return ()


def graph_namespace(metadata: Mapping[str, Any]) -> tuple[str, ...]:
    """Namespace of the graph a task runs in, from the task's run metadata.

    Drops the task's own segment, plus one per enclosing `@task` call: a call
    runs in its caller's graph but nests its segment under the caller's. A
    call's `langgraph_path` is `(PUSH, <caller's path>, <index>, True)`, while
    a `Send` task's ends in `False`.
    """
    ns = tuple(cast(str, metadata["langgraph_checkpoint_ns"]).split(NS_SEP))
    path = metadata.get("langgraph_path", ())
    drop = 1
    while len(path) == 4 and path[0] == PUSH and path[-1] is True:
        drop += 1
        path = path[1]
    return ns[:-drop]


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
            ns = graph_namespace(metadata)
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
            ns = graph_namespace(metadata)
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
REMOVE_ALL_MESSAGES = "__remove_all__"


def _messages_delta_reducer(
    state: list[AnyMessage], writes: list[list[AnyMessage]]
) -> list[AnyMessage]:
    """**Experimental.** Batch reducer for use with `DeltaChannel`.

    Processes all writes in one pass — dedup by ID, `RemoveMessage`
    tombstoning, `REMOVE_ALL_MESSAGES` reset — without calling `add_messages`.

    This reducer is batching-invariant, as required by `DeltaChannel`:
    `reducer(reducer(state, xs), ys) == reducer(state, xs + ys)`.

    Raw dict / string / tuple inputs are coerced to typed `BaseMessage`
    objects so that HTTP-driven graphs work without a separate coercion
    step. This is not full `add_messages` parity — unknown-id
    `RemoveMessage` errors, missing-id UUID assignment, and
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

    for idx in range(len(msgs) - 1, -1, -1):
        m = msgs[idx]
        if isinstance(m, RemoveMessage) and m.id == REMOVE_ALL_MESSAGES:
            state_msgs = []
            msgs = msgs[idx + 1 :]
            break

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
    channels: Mapping[str, object], writes: Sequence[tuple[str, object]]
) -> None:
    """Give the messages written to messages channels stable ids.

    A message saved without an id gets one later, when a reducer reads it, so
    the `updates` stream, the saved write and the state can each show a
    different id, and a replay or fork from that checkpoint mints a new one.
    A `DeltaChannel` replays its saved writes on every read, so there the id
    can change on every read.

    A `StateGraph` input is one write to `START`; the values in it get ids by
    the channel they go to, so the input checkpoint saves them. In a graph
    with one root channel, the whole input goes to that channel.

    A write that changes is replaced in `writes`: lists and dicts the caller
    passed in are never changed, only the ids of message objects.
    """
    for i, (ch, value) in enumerate(writes):
        if ch == START:
            new_value = _input_with_message_ids(channels, value)
        else:
            new_value = _channel_value_with_message_ids(channels.get(ch), value)
        if new_value is not value:
            cast(MutableSequence[tuple[str, object]], writes)[i] = (ch, new_value)


def _channel_value_with_message_ids(spec: object, value: object) -> object:
    if isinstance(spec, DeltaChannel):
        if spec.reducer is _messages_delta_reducer:
            return _delta_messages_ids(value)
        return _delta_ids(value)
    if isinstance(spec, BinaryOperatorAggregate) and _is_add_messages(spec.operator):
        return _add_messages_ids(value)
    return value


def _input_with_message_ids(channels: Mapping[str, object], value: object) -> object:
    if ROOT in channels:
        return _channel_value_with_message_ids(channels[ROOT], value)
    changed = {
        name: new
        for name, old in _state_items(value)
        if (new := _channel_value_with_message_ids(channels.get(name), old)) is not old
    }
    if not changed:
        return value
    if isinstance(value, dict):
        return {k: changed.get(k, v) for k, v in value.items()}
    if isinstance(value, BaseModel):
        return value.model_copy(update=changed)
    # Not `dataclasses.replace`, which reruns `__post_init__`.
    new_value = copy.copy(value)
    for name, field_value in changed.items():
        object.__setattr__(new_value, name, field_value)
    return new_value


@cache
def _add_messages_reducers() -> tuple[object, object]:
    # Imported here: `langgraph.graph.message` imports this module.
    from langgraph.graph.message import add_messages

    return add_messages, cast(Callable[[], partial], add_messages)().func


def _is_add_messages(operator: object) -> bool:
    add_messages, func = _add_messages_reducers()
    if isinstance(operator, partial):
        return operator.func is func
    return operator is add_messages


def _add_messages_ids(value: object) -> object:
    is_overwrite, items = _unwrap(value)
    if _is_many(items, is_overwrite):
        new = _map_messages(items, _add_messages_item)
    else:
        new = _add_messages_item(items)
    return _rewrap(value, is_overwrite, items, new)


def _delta_messages_ids(value: object) -> object:
    is_overwrite, items = _unwrap(value)
    if not _is_many(items, is_overwrite):
        new = _message(items)
        if is_overwrite and isinstance(new, BaseMessage):
            new = [new]
    elif is_overwrite:
        new = _map_messages(items, _message)
    else:
        new = _map_messages(items, _delta_messages_list_item)
    return _rewrap(value, is_overwrite, items, new)


def _delta_ids(value: object) -> object:
    """Other DeltaChannels may hold other data, so less is converted."""
    is_overwrite, items = _unwrap(value)
    if is_overwrite:
        many = _is_many(items, is_overwrite)
        new = _map_messages(items, _stamped) if many else _stamped(items)
    elif isinstance(items, list):
        new = _map_messages(items, _delta_list_item)
    else:
        new = _dict_with_id(items)
    return _rewrap(value, is_overwrite, items, new)


def _unwrap(value: object) -> tuple[bool, object]:
    is_overwrite, items = _get_overwrite(value)
    return is_overwrite, items if is_overwrite else value


def _rewrap(value: object, is_overwrite: bool, items: object, new: object) -> object:
    if new is items:
        return value
    if not is_overwrite:
        return new
    if isinstance(value, dict):
        return {OVERWRITE: new} if OVERWRITE in value else {**value, "value": new}
    return Overwrite(new)


def _is_many(
    items: object, is_overwrite: bool
) -> TypeGuard[list[object] | tuple[object, ...]]:
    """In an `Overwrite`, a tuple is many messages unless it's a `(role, content)` pair."""
    if isinstance(items, list):
        return True
    return is_overwrite and isinstance(items, tuple) and not _is_message_pair(items)


def _is_message_pair(items: tuple[object, ...]) -> bool:
    if len(items) != 2 or not isinstance(items[0], str):
        return False
    try:
        convert_to_messages([items])
    except Exception:
        return False
    return True


def _map_messages(
    items: list[object] | tuple[object, ...], step: Callable[[object], object]
) -> list[object] | tuple[object, ...]:
    mapped = [step(item) for item in items]
    if all(new is old for new, old in zip(mapped, items)):
        return items
    return mapped if isinstance(items, list) else tuple(mapped)


def _new_id() -> str:
    return str(uuid.uuid4())


def _needs_id(id: object, *, keep_empty: bool) -> bool:
    return id is None or (not keep_empty and not id)


def _stamped(item: T, *, keep_empty: bool = False) -> T:
    if isinstance(item, BaseMessage) and _needs_id(item.id, keep_empty=keep_empty):
        item.id = _new_id()
    return item


def _dict_with_id(item: object, *, keep_empty: bool = False) -> object:
    if not isinstance(item, dict):
        return _stamped(item, keep_empty=keep_empty)
    if _is_message_dict(item) and _needs_id(item.get("id"), keep_empty=keep_empty):
        return {**item, "id": _new_id()}
    return item


def _message(item: object, *, strict: bool = False, keep_empty: bool = False) -> object:
    if not isinstance(item, BaseMessage):
        try:
            item = convert_to_messages([item])[0]
        except Exception:
            if strict:
                raise
            return item
    return _stamped(item, keep_empty=keep_empty)


def _is_list_message_dict(item: object) -> TypeGuard[dict[str, object]]:
    # Raises here, before it is saved, if it can't become a message.
    return isinstance(item, dict) and _is_message_dict(item)


def _delta_messages_list_item(item: object) -> object:
    return _message(item, strict=_is_list_message_dict(item))


def _delta_list_item(item: object) -> object:
    if _is_list_message_dict(item):
        return _message(item, strict=True)
    return _dict_with_id(item)


def _add_messages_item(item: object) -> object:
    if isinstance(item, dict):
        return _dict_with_id(item, keep_empty=True)
    return _message(item, keep_empty=True)
