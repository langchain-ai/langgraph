from typing import Any, Sequence, Union

from langchain_core.load.serializable import Serializable
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import BaseTool

from langgraph.utils import RunnableCallable

INVALID_TOOL_MSG_TEMPLATE = (
    "{requested_tool_name} is not a valid tool, "
    "try one of [{available_tool_names_str}]."
)


class ToolInvocationInterface:
    """Interface for invoking a tool"""

    tool: str
    tool_input: Union[str, dict]


class ToolInvocation(Serializable):
    """Information about how to invoke a tool."""

    tool: str
    """The name of the Tool to execute."""
    tool_input: Union[str, dict]
    """The input to pass in to the Tool."""


class ToolExecutor(RunnableCallable):
    def __init__(
        self,
        tools: Sequence[BaseTool],
        *,
        invalid_tool_msg_template: str = INVALID_TOOL_MSG_TEMPLATE,
    ) -> None:
        super().__init__(self._execute, afunc=self._aexecute, trace=False)
        self.tools = tools
        self.tool_map = {t.name: t for t in tools}
        self.invalid_tool_msg_template = invalid_tool_msg_template

    def _execute(
        self, tool_invocation: ToolInvocationInterface, config: RunnableConfig
    ) -> Any:
        if tool_invocation.tool not in self.tool_map:
            return self.invalid_tool_msg_template.format(
                requested_tool_name=tool_invocation.tool,
                available_tool_names_str=", ".join([t.name for t in self.tools]),
            )
        else:
            tool = self.tool_map[tool_invocation.tool]
            output = tool.invoke(tool_invocation.tool_input, config)
            return output

    async def _aexecute(
        self, tool_invocation: ToolInvocationInterface, config: RunnableConfig
    ) -> Any:
        if tool_invocation.tool not in self.tool_map:
            return self.invalid_tool_msg_template.format(
                requested_tool_name=tool_invocation.tool,
                available_tool_names_str=", ".join([t.name for t in self.tools]),
            )
        else:
            tool = self.tool_map[tool_invocation.tool]
            output = await tool.ainvoke(tool_invocation.tool_input, config)
            return output
