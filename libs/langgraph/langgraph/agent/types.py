from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Annotated, Any, Literal

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AnyMessage
from langchain_core.tools import BaseTool
from pydantic import BaseModel
from typing_extensions import TypedDict

from langgraph.channels.ephemeral_value import EphemeralValue
from langgraph.graph.message import Messages, add_messages

ResponseFormat = dict | type[BaseModel]
JumpTo = Literal["tools", "model", "__end__"]


@dataclass
class ModelRequest:
    model: BaseChatModel
    system_prompt: str
    messages: Sequence[AnyMessage]  # excluding system prompt
    tool_choice: Any
    tools: Sequence[BaseTool]
    response_format: ResponseFormat | None


@dataclass
class AgentState:
    messages: Annotated[list[AnyMessage], add_messages]
    model_request: Annotated[ModelRequest | None, EphemeralValue]
    jump_to: Annotated[JumpTo | None, EphemeralValue] = None
    response: dict | None = None


class AgentMiddleware:
    class State(AgentState):
        pass

    def before_model(self, state: State) -> AgentUpdate | AgentJump | None:
        pass

    def modify_model_request(self, request: ModelRequest, state: State) -> ModelRequest:
        return request

    def after_model(self, state: State) -> AgentUpdate | AgentJump | None:
        pass


class AgentUpdate(TypedDict, total=False):
    messages: Messages
    response: dict


class AgentJump(TypedDict, total=False):
    messages: Messages
    jump_to: JumpTo
