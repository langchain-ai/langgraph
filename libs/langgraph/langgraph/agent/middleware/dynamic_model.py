from dataclasses import dataclass
from typing import Literal

from langchain_core.language_models.chat_models import BaseChatModel

from langgraph.agent.types import (
    AgentJump,
    AgentMiddleware,
    AgentState,
    AgentUpdate,
    ModelRequest,
)


class DynamicModelMiddleware(AgentMiddleware):
    """Selects different models based on task complexity"""

    def __init__(
        self,
        basic_model: BaseChatModel,
        complex_model: BaseChatModel,
        message_threshold: int = 5,
    ):
        self.basic_model = basic_model
        self.complex_model = complex_model
        self.message_threshold = message_threshold

    def modify_model_request(
        self, request: ModelRequest, state: AgentState
    ) -> ModelRequest:
        if len(state.messages) > self.message_threshold:
            request.model = self.complex_model
        else:
            request.model = self.basic_model
        return request
