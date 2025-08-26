from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Annotated, Any, Literal, Self

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AnyMessage
from langchain_core.tools import BaseTool
from pydantic import BaseModel
from typing_extensions import TypedDict

from langgraph.channels.ephemeral_value import EphemeralValue
from langgraph.graph.message import Messages, add_messages

ResponseFormat = dict | type[BaseModel]
GoTo = Literal["tools", "model", "__end__"]


@dataclass
class ModelRequest:
    model: BaseChatModel
    system_prompt: str
    messages: Sequence[AnyMessage]  # excluding system prompt
    tool_choice: Any
    tools: Sequence[BaseTool]
    response_format: ResponseFormat | None


class AgentMiddleware:
    def __init__(self) -> None:
        pass

    def __copy__(self) -> Self:
        return self.__class__(**self.__dict__)

    def before_model(self, state: AgentState) -> AgentUpdate | AgentGoTo | None:
        pass

    def modify_model_request(
        self, request: ModelRequest, state: AgentState
    ) -> ModelRequest:
        return request

    def after_model(self, state: AgentState) -> AgentUpdate | AgentGoTo | None:
        pass


class AgentUpdate(TypedDict, total=False):
    messages: Messages
    response: dict


class AgentGoTo(TypedDict, total=False):
    messages: Messages
    goto: GoTo


@dataclass
class AgentState:
    messages: Annotated[list[AnyMessage], add_messages]
    goto: Annotated[GoTo | None, EphemeralValue] = None
    response: dict | None = None
