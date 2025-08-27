from langgraph.agent.types import AgentMiddleware, AgentState, ModelRequest
from typing import Dict, Any, List, Optional, Union
from langgraph.types import interrupt


class SwarmMiddleWare(AgentMiddleware):

    def __init__(self, model_configs: dict[str, dict]):
        super().__init__()

    def modify_model_request(
        self, request: ModelRequest, state: AgentState
    ) -> ModelRequest: