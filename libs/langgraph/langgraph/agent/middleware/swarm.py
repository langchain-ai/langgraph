from langgraph.agent.types import AgentMiddleware, ModelRequest, AgentJump
from langchain_core.tools import BaseTool, tool
from langgraph.agent import create_agent
from langchain_core.messages import AIMessage, ToolMessage, HumanMessage
from dataclasses import dataclass
from typing import cast


@dataclass
class SwarmAgent:
    name: str
    system_prompt: str
    tools: list[BaseTool]


class SwarmMiddleware(AgentMiddleware):
    """Swarm middleware.

    TODOs:
    * Support create_agent for handoffs
    * Support handoff customization
      * do we want to include handoff messages / enable togglging
      * default active agent
      * handoff tool naming / descriptions
    """

    class State(AgentMiddleware.State):
        active_agent: str | None = None

    @staticmethod
    def _create_handoff_tools(agents: list[SwarmAgent]) -> list[BaseTool]:
        handoff_tools: list[BaseTool] = []

        for agent in agents:

            def handoff_tool() -> str:
                return f"Handing off to {agent.name}"

            handoff_tools.append(
                tool(
                    f"handoff_to_{agent.name}",
                    description=f"Handoff tool to trigger a handoff to {agent.name}",
                )(handoff_tool)
            )

        return handoff_tools

    def __init__(self, agents: list[SwarmAgent]):
        self.agents: dict[str, SwarmAgent] = {agent.name: agent for agent in agents}
        self.handoff_tools = self._create_handoff_tools(agents)

    def modify_model_request(self, request: ModelRequest, state: State) -> ModelRequest:
        if (active_agent := getattr(state, "active_agent", None)) is not None:
            agent = self.agents[active_agent]
            request.system_prompt = agent.system_prompt
            request.tools = agent.tools

        request.tools.extend(self.handoff_tools)
        return request

    def after_model(self, state) -> State | None:
        # TODO: handle parallel handoffs, we don't do this currently

        ai_msg: AIMessage = cast(AIMessage, state.messages[-1])
        if ai_msg.tool_calls:
            for call in ai_msg.tool_calls:
                if call["name"].startswith("handoff_to_"):
                    active_agent = call["name"].replace("handoff_to_", "")
                    return {
                        "messages": [
                            ToolMessage(
                                name=call["name"],
                                content=f"Successfully transferred to {active_agent}",
                                tool_call_id=call["id"],
                            )
                        ],
                        "active_agent": active_agent,
                        "jump_to": "model",
                    }
            return None