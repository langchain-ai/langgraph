import operator
from dataclasses import dataclass
from typing import Annotated

from langgraph.agent.types import AgentJump, AgentMiddleware, AgentState, AgentUpdate


class ModelRequestLimitMiddleware(AgentMiddleware):
    """Terminates after N model requests"""

    @dataclass
    class State(AgentMiddleware.State):
        model_request_count: Annotated[int, operator.add] = 0

    def __init__(self, max_requests: int = 10):
        self.max_requests = max_requests

    def before_model(self, state: State) -> AgentUpdate | AgentJump | None:
        # TODO: want to be able to configure end behavior here
        if state.model_request_count == self.max_requests:
            return {"jump_to": "__end__"}

        return {"model_request_count": 1}
