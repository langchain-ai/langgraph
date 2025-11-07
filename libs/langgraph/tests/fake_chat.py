import re
from collections.abc import AsyncIterator, Iterator
from typing import Any, cast

from langchain_core.callbacks import (
    AsyncCallbackManagerForLLMRun,
    CallbackManagerForLLMRun,
)
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage, AIMessageChunk, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult


class FakeChatModel(GenericFakeChatModel):
    messages: list[BaseMessage]

    i: int = 0

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
        generation = ChatGeneration(message=message_)
        return ChatResult(generations=[generation])

    def _stream(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> Iterator[ChatGenerationChunk]:
        """Stream the output of the model."""
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
                    chunk = ChatGenerationChunk(
                        message=AIMessageChunk(
                            content=token, id=message.id, chunk_position="last"
                        )
                    )
                else:
                    chunk = ChatGenerationChunk(
                        message=AIMessageChunk(content=token, id=message.id)
                    )
                if run_manager:
                    run_manager.on_llm_new_token(token, chunk=chunk)
                yield chunk
        else:
            args = message.__dict__
            args.pop("type")
            chunk = ChatGenerationChunk(
                message=AIMessageChunk(**args, chunk_position="last")
            )
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
                    chunk = ChatGenerationChunk(
                        message=AIMessageChunk(
                            content=token, id=message.id, chunk_position="last"
                        )
                    )
                else:
                    chunk = ChatGenerationChunk(
                        message=AIMessageChunk(content=token, id=message.id)
                    )

                if run_manager:
                    run_manager.on_llm_new_token(token, chunk=chunk)
                yield chunk
        else:
            args = message.__dict__
            args.pop("type")
            chunk = ChatGenerationChunk(
                message=AIMessageChunk(**args, chunk_position="last")
            )
            if run_manager:
                await run_manager.on_llm_new_token("", chunk=chunk)
            yield chunk
