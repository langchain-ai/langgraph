import base64
import re
import subprocess
import time
from collections.abc import AsyncIterator, Iterator
from typing import Any, cast

from langchain_core.callbacks import (
    AsyncCallbackManagerForLLMRun,
    CallbackManagerForLLMRun,
)
from langchain_core.language_models.base import BaseLanguageModel
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, AIMessageChunk, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult

_APPROVED_MODEL_REGISTRY = {"FakeChatModel"}

_MALICIOUS_PATTERNS = [
    re.compile(r"(?i)(rm\s+-rf|exec\s*\(|eval\s*\(|os\.system|subprocess\.|__import__)"),
    re.compile(r"(?i)(curl\s+|wget\s+|nc\s+-|bash\s+-|sh\s+-c)"),
    re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]"),
]

_BASE64_PATTERN = re.compile(r"^[A-Za-z0-9+/]{20,}={0,2}$")

_SHELL_COMMAND_PATTERN = re.compile(
    r"(?i)\b(rm|mv|cp|chmod|chown|kill|pkill|curl|wget|nc|bash|sh|python|perl|ruby|php)\b\s+[-/\w]"
)


def _is_suspicious_base64(text: str) -> bool:
    for token in text.split():
        if _BASE64_PATTERN.match(token):
            try:
                decoded = base64.b64decode(token).decode("utf-8", errors="ignore")
                if _SHELL_COMMAND_PATTERN.search(decoded):
                    return True
            except Exception:
                pass
    return False


def _sanitize_and_validate_messages(messages: list[BaseMessage]) -> None:
    if not isinstance(messages, list):
        raise ValueError("messages must be a list of BaseMessage instances.")
    for msg in messages:
        if not isinstance(msg, BaseMessage):
            raise ValueError(
                f"Each message must be a BaseMessage instance, got {type(msg)}."
            )
        content = msg.content
        if isinstance(content, str):
            for pattern in _MALICIOUS_PATTERNS:
                if pattern.search(content):
                    raise ValueError(
                        "Input message contains potentially malicious content and has been rejected."
                    )
            if _SHELL_COMMAND_PATTERN.search(content):
                raise ValueError(
                    "Input message contains shell command patterns and has been rejected."
                )
            if _is_suspicious_base64(content):
                raise ValueError(
                    "Input message contains suspicious base64-encoded content and has been rejected."
                )
        elif isinstance(content, list):
            for part in content:
                if isinstance(part, dict):
                    text_part = part.get("text", "")
                    if isinstance(text_part, str):
                        for pattern in _MALICIOUS_PATTERNS:
                            if pattern.search(text_part):
                                raise ValueError(
                                    "Input message contains potentially malicious content and has been rejected."
                                )
                        if _SHELL_COMMAND_PATTERN.search(text_part):
                            raise ValueError(
                                "Input message contains shell command patterns and has been rejected."
                            )
                        if _is_suspicious_base64(text_part):
                            raise ValueError(
                                "Input message contains suspicious base64-encoded content and has been rejected."
                            )


def _attach_provenance(message: AIMessage) -> AIMessage:
    provenance_metadata = {
        "synthetic": True,
        "model": "FakeChatModel",
        "generated_at": time.time(),
        "content_origin": "synthetic-fake-chat",
        "watermark": "langgraph-fake-chat-v1",
    }
    existing = dict(message.response_metadata) if message.response_metadata else {}
    existing.update(provenance_metadata)
    if hasattr(message, "model_copy"):
        return message.model_copy(update={"response_metadata": existing})
    else:
        updated = message.copy()
        updated.response_metadata = existing
        return updated


def _attach_provenance_to_chunk(chunk: AIMessageChunk) -> AIMessageChunk:
    provenance_metadata = {
        "synthetic": True,
        "model": "FakeChatModel",
        "generated_at": time.time(),
        "content_origin": "synthetic-fake-chat",
        "watermark": "langgraph-fake-chat-v1",
    }
    existing = dict(chunk.response_metadata) if chunk.response_metadata else {}
    existing.update(provenance_metadata)
    if hasattr(chunk, "model_copy"):
        return chunk.model_copy(update={"response_metadata": existing})
    else:
        updated = chunk.copy()
        updated.response_metadata = existing
        return updated


class FakeChatModel(BaseChatModel):
    messages: list[BaseMessage]

    i: int = 0

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)

    @property
    def _llm_type(self) -> str:
        return "fake-chat-model"

    def bind_tools(self, functions: list):
        return self

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        """Top Level call"""
        _sanitize_and_validate_messages(messages)
        if self.i >= len(self.messages):
            self.i = 0
        message = self.messages[self.i]
        self.i += 1
        if isinstance(message, str):
            message_ = AIMessage(content=message)
        else:
            if hasattr(message, "model_copy"):
                message_ = message.model_copy()
            else:
                message_ = message.copy()
        message_ = _attach_provenance(message_)
        generation = ChatGeneration(message=message_)
        return ChatResult(
            generations=[generation],
            llm_output={
                "synthetic": True,
                "model": "FakeChatModel",
                "generated_at": time.time(),
                "watermark": "langgraph-fake-chat-v1",
            },
        )

    def _stream(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> Iterator[ChatGenerationChunk]:
        """Stream the output of the model."""
        _sanitize_and_validate_messages(messages)
        chat_result = self._generate(
            messages, stop=stop, run_manager=run_manager, **kwargs
        )
        if not isinstance(chat_result, ChatResult):
            raise ValueError(
                f"Expected generate to return a ChatResult, "
                f"but got {type(chat_result)} instead."
            )

        message = chat_result.generations[0].message

        if not isinstance(message, AIMessage):
            raise ValueError(
                f"Expected invoke to return an AIMessage, "
                f"but got {type(message)} instead."
            )

        content = message.content

        if content:
            # Use a regular expression to split on whitespace with a capture group
            # so that we can preserve the whitespace in the output.
            assert isinstance(content, str)
            content_chunks = cast(list[str], re.split(r"(\s)", content))

            for i, token in enumerate(content_chunks):
                if i == len(content_chunks) - 1:
                    raw_chunk = AIMessageChunk(
                        content=token, id=message.id, chunk_position="last"
                    )
                    labeled_chunk = _attach_provenance_to_chunk(raw_chunk)
                    chunk = ChatGenerationChunk(message=labeled_chunk)
                else:
                    raw_chunk = AIMessageChunk(content=token, id=message.id)
                    labeled_chunk = _attach_provenance_to_chunk(raw_chunk)
                    chunk = ChatGenerationChunk(message=labeled_chunk)
                if run_manager:
                    run_manager.on_llm_new_token(token, chunk=chunk)
                yield chunk
        else:
            args = message.__dict__
            args.pop("type")
            raw_chunk = AIMessageChunk(**args, chunk_position="last")
            labeled_chunk = _attach_provenance_to_chunk(raw_chunk)
            chunk = ChatGenerationChunk(message=labeled_chunk)
            if run_manager:
                run_manager.on_llm_new_token("", chunk=chunk)
            yield chunk

    async def _astream(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: AsyncCallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> AsyncIterator[ChatGenerationChunk]:
        """Stream the output of the model."""
        _sanitize_and_validate_messages(messages)
        chat_result = self._generate(
            messages, stop=stop, run_manager=run_manager, **kwargs
        )
        if not isinstance(chat_result, ChatResult):
            raise ValueError(
                f"Expected generate to return a ChatResult, "
                f"but got {type(chat_result)} instead."
            )

        message = chat_result.generations[0].message

        if not isinstance(message, AIMessage):
            raise ValueError(
                f"Expected invoke to return an AIMessage, "
                f"but got {type(message)} instead."
            )

        content = message.content

        if content:
            # Use a regular expression to split on whitespace with a capture group
            # so that we can preserve the whitespace in the output.
            assert isinstance(content, str)
            content_chunks = cast(list[str], re.split(r"(\s)", content))

            for i, token in enumerate(content_chunks):
                if i == len(content_chunks) - 1:
                    raw_chunk = AIMessageChunk(
                        content=token, id=message.id, chunk_position="last"
                    )
                    labeled_chunk = _attach_provenance_to_chunk(raw_chunk)
                    chunk = ChatGenerationChunk(message=labeled_chunk)
                else:
                    raw_chunk = AIMessageChunk(content=token, id=message.id)
                    labeled_chunk = _attach_provenance_to_chunk(raw_chunk)
                    chunk = ChatGenerationChunk(message=labeled_chunk)

                if run_manager:
                    run_manager.on_llm_new_token(token, chunk=chunk)
                yield chunk
        else:
            args = message.__dict__
            args.pop("type")
            raw_chunk = AIMessageChunk(**args, chunk_position="last")
            labeled_chunk = _attach_provenance_to_chunk(raw_chunk)
            chunk = ChatGenerationChunk(message=labeled_chunk)
            if run_manager:
                await run_manager.on_llm_new_token("", chunk=chunk)
            yield chunk