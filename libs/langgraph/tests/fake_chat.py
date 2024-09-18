import re
from typing import Any, Iterator, List, Optional, cast

from langchain_core.callbacks import CallbackManagerForLLMRun
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
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
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
            message_ = message
        generation = ChatGeneration(message=message_)
        return ChatResult(generations=[generation])

    def _stream(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
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

            for token in content_chunks:
                chunk = ChatGenerationChunk(
                    message=AIMessageChunk(content=token, id=message.id)
                )
                if run_manager:
                    run_manager.on_llm_new_token(token, chunk=chunk)
                yield chunk
        else:
            args = message.__dict__
            args.pop("type")
            chunk = ChatGenerationChunk(message=AIMessageChunk(**args))
            if run_manager:
                run_manager.on_llm_new_token("", chunk=chunk)
            yield chunk
