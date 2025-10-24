from typing import Literal

from pydantic import BaseModel


# define these objects to avoid importing langchain_core.agents
# and therefore avoid relying on core Pydantic version
class AgentAction(BaseModel):
    """
    Represents a request to execute an action by an agent.

    The action consists of the name of the tool to execute and the input to pass
    to the tool. The log is used to pass along extra information about the action.
    """

    tool: str
    tool_input: str | dict
    log: str
    type: Literal["AgentAction"] = "AgentAction"


class AgentFinish(BaseModel):
    """Final return value of an ActionAgent.

    Agents return an AgentFinish when they have reached a stopping condition.
    """

    return_values: dict
    log: str
    type: Literal["AgentFinish"] = "AgentFinish"
