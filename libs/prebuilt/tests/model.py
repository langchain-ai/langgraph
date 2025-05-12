from typing import (
    Any,
    Callable,
    Dict,
    List,
    Literal,
    Optional,
    Sequence,
    Type,
    Union,
)

from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models import BaseChatModel, LanguageModelInput
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    ToolCall,
)
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.runnables import Runnable, RunnableLambda
from langchain_core.tools import BaseTool
from pydantic import BaseModel

from langgraph.prebuilt.chat_agent_executor import StructuredResponse


class FakeToolCallingModel(BaseChatModel):
    tool_calls: Optional[list[list[ToolCall]]] = None
    structured_response: Optional[StructuredResponse] = None
    index: int = 0
    tool_style: Literal["openai", "anthropic"] = "openai"

    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        """Top Level call"""
        messages_string = "-".join([m.content for m in messages])
        tool_calls = (
            self.tool_calls[self.index % len(self.tool_calls)]
            if self.tool_calls
            else []
        )
        message = AIMessage(
            content=messages_string, id=str(self.index), tool_calls=tool_calls.copy()
        )
        self.index += 1
        return ChatResult(generations=[ChatGeneration(message=message)])

    @property
    def _llm_type(self) -> str:
        return "fake-tool-call-model"

    def with_structured_output(
        self, schema: Type[BaseModel]
    ) -> Runnable[LanguageModelInput, StructuredResponse]:
        if self.structured_response is None:
            raise ValueError("Structured response is not set")

        return RunnableLambda(lambda x: self.structured_response)

    def bind_tools(
        self,
        tools: Sequence[Union[Dict[str, Any], Type[BaseModel], Callable, BaseTool]],
        **kwargs: Any,
    ) -> Runnable[LanguageModelInput, BaseMessage]:
        if len(tools) == 0:
            raise ValueError("Must provide at least one tool")

        tool_dicts = []
        for tool in tools:
            if not isinstance(tool, BaseTool):
                raise TypeError(
                    "Only BaseTool is supported by FakeToolCallingModel.bind_tools"
                )

            # NOTE: this is a simplified tool spec for testing purposes only
            if self.tool_style == "openai":
                tool_dicts.append(
                    {
                        "type": "function",
                        "function": {
                            "name": tool.name,
                        },
                    }
                )
            elif self.tool_style == "anthropic":
                tool_dicts.append(
                    {
                        "name": tool.name,
                    }
                )

        return self.bind(tools=tool_dicts)
