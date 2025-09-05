from dataclasses import dataclass, field
from typing import Annotated, Any, Dict, List, cast

from langchain_core.messages import AIMessage
from typing_extensions import Annotated
from typing import ClassVar
from operator import add

from langgraph.agent import create_agent
from langgraph.agent.types import AgentJump, AgentMiddleware, AgentState, AgentUpdate, ModelRequest

class AcceptInput:
    ...

class ExposeOutput:
    ...

# other ideas
# state_extensions: ClassVar[dict[str, type | type[Annotated]]] = {
#         "int1": Annotated[int, add],
#         "int2": Annotated[int, add, AcceptInput],
#         "int3": Annotated[int, add, ExposeOutput],
#         "int4": Annotated[int, add, AcceptInput, ExposeOutput],
# }

class State(AgentState):
    int1: Annotated[int, add]
    int2: Annotated[int, add, AcceptInput]
    int3: Annotated[int, add, ExposeOutput]
    int4: Annotated[int, add, AcceptInput, ExposeOutput]

class StateModMidleware(AgentMiddleware[State]):
    """Terminates after a specific tool is called N times."""

    state_schema: type[State] = State

    def __init__(self):
        pass

    def before_model(self, state: State) -> AgentUpdate | AgentJump | None:
        return {"int1": 1, "int2": 1, "int3": 1, "int4": 1}

    def modify_model_request(self, request: ModelRequest, state: State) -> ModelRequest:
        return request

agent = create_agent(
    model="gpt-4o",
    tools=[],
    system_prompt="You are a helpful assistant.",
    middleware=[StateModMidleware()],
)

