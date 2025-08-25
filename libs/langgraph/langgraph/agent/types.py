from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Annotated, Any, Self

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AnyMessage
from langchain_core.tools import BaseTool
from typing_extensions import TypedDict

from langgraph.graph.message import Messages, add_messages


@dataclass
class ModelRequest:
    model: BaseChatModel
    system_prompt: str
    messages: Sequence[AnyMessage]  # excluding system prompt
    tool_choice: Any
    tools: Sequence[BaseTool]


class AgentMiddleware:
    def __init__(self) -> None:
        pass

    def __copy__(self) -> Self:
        return self.__class__(**self.__dict__)

    def before_model(self, state: AgentState) -> AgentState | None:
        pass

    def modify_model_request(self, request: ModelRequest) -> ModelRequest:
        return request

    def after_model(self, state: AgentState) -> AgentState | None:
        pass


class AgentInput(TypedDict):
    messages: Messages


@dataclass
class AgentState:
    messages: Annotated[list[AnyMessage], add_messages]
