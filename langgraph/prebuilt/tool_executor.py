from langchain_core.runnables import RunnableBinding, RunnableLambda
from typing import Sequence, Any
from langchain_core.tools import BaseTool
from langchain_core.agents import AgentAction
INVALID_TOOL_MSG_TEMPLATE = (
    "{requested_tool_name} is not a valid tool, "
    "try one of [{available_tool_names_str}]."
)

class ToolExecutor(RunnableBinding):

    tools: Sequence[BaseTool]
    tool_map: dict
    invalid_tool_msg_template: str
    def __init__(self, tools: Sequence[BaseTool], invalid_tool_msg_template: str = INVALID_TOOL_MSG_TEMPLATE, **kwargs: Any) -> None:

        bound = RunnableLambda(self._execute, afunc=self._aexecute)
        super().__init__(bound=bound, tools=tools, tool_map ={t.name: t for t in tools}, invalid_tool_msg_template=invalid_tool_msg_template, **kwargs)

    def _execute(self, tool_invocation: AgentAction) -> Any:
        if tool_invocation.tool not in self.tool_map:
            return self.invalid_tool_msg_template.format(
                requested_tool_name=tool_invocation.tool,
                available_tool_names_str=", ".join([t.name for t in self.tools])
            )
        else:
            tool = self.tool_map[tool_invocation.tool]
            output = tool.invoke(tool_invocation.tool_input)
            return output

    async def _aexecute(self, tool_invocation: AgentAction) -> Any:
        if tool_invocation.tool not in self.tool_map:
            return self.invalid_tool_msg_template.format(
                requested_tool_name=tool_invocation.tool,
                available_tool_names_str=", ".join([t.name for t in self.tools])
            )
        else:
            tool = self.tool_map[tool_invocation.tool]
            output = await tool.ainvoke(tool_invocation.tool_input)
            return output