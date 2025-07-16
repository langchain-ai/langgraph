from __future__ import annotations

from collections.abc import AsyncIterator, Iterator, Sequence
from typing import (
    Any,
    Callable,
    TypeVar,
    cast,
)
from uuid import UUID, uuid4

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.messages import BaseMessage
from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, LLMResult

from langgraph._internal._constants import NS_SEP
from langgraph.constants import TAG_HIDDEN, TAG_NOSTREAM
from langgraph.pregel.protocol import StreamChunk
from langgraph.types import Command

try:
    from langchain_core.tracers._streaming import _StreamingCallbackHandler
except ImportError:
    _StreamingCallbackHandler = object  # type: ignore

T = TypeVar("T")
Meta = tuple[tuple[str, ...], dict[str, Any]]


class StreamMessagesHandler(BaseCallbackHandler, _StreamingCallbackHandler):
    """A callback handler that implements stream_mode=messages.
    Collects messages from (1) chat model stream events and (2) node outputs."""

    run_inline = True
    """We want this callback to run in the main thread, to avoid order/locking issues."""

    def __init__(self, stream: Callable[[StreamChunk], None], subgraphs: bool):
        self.stream = stream
        self.subgraphs = subgraphs
        self.metadata: dict[UUID, Meta] = {}
        self.seen: set[int | str] = set()

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
        elif isinstance(response, dict):
            for value in response.values():
                if isinstance(value, BaseMessage):
                    self._emit(meta, value, dedupe=True)
                elif isinstance(value, Sequence):
                    for item in value:
                        if isinstance(item, BaseMessage):
                            self._emit(meta, item, dedupe=True)
        elif hasattr(response, "__dir__") and callable(response.__dir__):
            for key in dir(response):
                try:
                    value = getattr(response, key)
                    if isinstance(value, BaseMessage):
                        self._emit(meta, value, dedupe=True)
                    elif isinstance(value, Sequence):
                        for item in value:
                            if isinstance(item, BaseMessage):
                                self._emit(meta, item, dedupe=True)
                except AttributeError:
                    pass

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
            if not self.subgraphs and len(ns) > 0:
                return
            if tags:
                if filtered_tags := [t for t in tags if not t.startswith("seq:step")]:
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
            if isinstance(inputs, dict):
                for key, value in inputs.items():
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
