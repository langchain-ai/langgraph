from typing import (
    Any,
    AsyncIterator,
    Callable,
    Dict,
    Iterator,
    List,
    Optional,
    Sequence,
    Union,
    cast,
)
from uuid import UUID, uuid4

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.messages import BaseMessage
from langchain_core.outputs import ChatGenerationChunk, LLMResult
from langchain_core.tracers._streaming import T, _StreamingCallbackHandler

from langgraph.constants import NS_SEP
from langgraph.pregel.loop import StreamChunk

Meta = tuple[tuple[str, ...], dict[str, Any]]


class StreamMessagesHandler(BaseCallbackHandler, _StreamingCallbackHandler):
    """A callback handler that implements stream_mode=messages.
    Collects messages from (1) chat model stream events and (2) node outputs."""

    def __init__(self, stream: Callable[[StreamChunk], None]):
        self.stream = stream
        self.metadata: dict[UUID, Meta] = {}
        self.seen: set[Union[int, str]] = set()

    def _emit(self, meta: Meta, message: BaseMessage, *, dedupe: bool = False) -> None:
        ident = id(message)
        if dedupe and message.id in self.seen:
            return
        elif ident in self.seen:
            return
        else:
            if message.id is None:
                message.id = str(uuid4())
            self.seen.add(ident)
            self.seen.add(message.id)
            self.stream((meta[0], "messages", (message, meta[1])))

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
        parent_run_id: Optional[UUID] = None,
        tags: Optional[list[str]] = None,
        metadata: Optional[dict[str, Any]] = None,
        **kwargs: Any,
    ) -> Any:
        if metadata:
            self.metadata[run_id] = (
                tuple(cast(str, metadata["langgraph_checkpoint_ns"]).split(NS_SEP)),
                metadata,
            )

    def on_llm_new_token(
        self,
        token: str,
        *,
        chunk: Optional[ChatGenerationChunk] = None,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
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
        parent_run_id: Optional[UUID] = None,
        **kwargs: Any,
    ) -> Any:
        self.metadata.pop(run_id, None)

    def on_llm_error(
        self,
        error: BaseException,
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        **kwargs: Any,
    ) -> Any:
        self.metadata.pop(run_id, None)

    def on_chain_start(
        self,
        serialized: Dict[str, Any],
        inputs: Dict[str, Any],
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        tags: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> Any:
        if metadata and kwargs.get("name") == metadata.get("langgraph_node"):
            self.metadata[run_id] = (
                tuple(cast(str, metadata["langgraph_checkpoint_ns"]).split(NS_SEP)),
                metadata,
            )

    def on_chain_end(
        self,
        response: Any,
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        **kwargs: Any,
    ) -> Any:
        if meta := self.metadata.pop(run_id, None):
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

    def on_chain_error(
        self,
        error: BaseException,
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        **kwargs: Any,
    ) -> Any:
        self.metadata.pop(run_id, None)
