from typing import Literal, Union

from pydantic import BaseModel


# define these objects to avoid importing langchain_core.agents
# and therefore avoid relying on core Pydantic version
class AgentAction(BaseModel):
    tool: str
    tool_input: Union[str, dict]
    log: str
    type: Literal["AgentAction"] = "AgentAction"

    model_config = {
        "json_schema_extra": {
            "description": (
                """Represents a request to execute an action by an agent.

The action consists of the name of the tool to execute and the input to pass
to the tool. The log is used to pass along extra information about the action."""
            )
        }
    }


class AgentFinish(BaseModel):
    """Final return value of an ActionAgent.

    Agents return an AgentFinish when they have reached a stopping condition.
    """

    return_values: dict
    log: str
    type: Literal["AgentFinish"] = "AgentFinish"
    model_config = {
        "json_schema_extra": {
            "description": (
                """Final return value of an ActionAgent.

Agents return an AgentFinish when they have reached a stopping condition."""
            )
        }
    }
