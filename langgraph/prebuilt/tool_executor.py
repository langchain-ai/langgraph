import json
from typing import Any, List, Optional, Sequence, Union

from langchain_core.load.serializable import Serializable
from langchain_core.runnables import RunnableBinding, RunnableConfig, RunnableLambda
from langchain_core.tools import BaseTool
from langchain_core.messages import AIMessage

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
    id: Optional[str] = None
    """The tool invocation ID"""


def create_tool_invocations(message: AIMessage) -> List[ToolInvocation]:
    if message.additional_kwargs.get("function_call"):
        return [
            ToolInvocation(
                tool=message.additional_kwargs["function_call"]["name"],
                tool_input=json.loads(
                    message.additional_kwargs["function_call"]["arguments"]
                ),
            )
        ]
    if message.additional_kwargs.get("tool_calls"):
        return [
            ToolInvocation(
                tool=tool["function"]["name"],
                tool_input=json.loads(tool["function"]["arguments"]),
                id=tool["id"],
            )
            for tool in message.additional_kwargs.get("tool_calls")
            if tool["type"] == "function"
        ]
    return []


class ToolExecutor(RunnableBinding):
    tools: Sequence[BaseTool]
    tool_map: dict
    invalid_tool_msg_template: str

    def __init__(
        self,
        tools: Sequence[BaseTool],
        *,
        invalid_tool_msg_template: str = INVALID_TOOL_MSG_TEMPLATE,
        **kwargs: Any,
    ) -> None:
        bound = RunnableLambda(self._execute, afunc=self._aexecute)
        super().__init__(
            bound=bound,
            tools=tools,
            tool_map={t.name: t for t in tools},
            invalid_tool_msg_template=invalid_tool_msg_template,
            **kwargs,
        )

    def _execute(
        self, tool_invocation: ToolInvocationInterface, *, config: RunnableConfig
    ) -> Any:
        if tool_invocation.tool not in self.tool_map:
            return self.invalid_tool_msg_template.format(
                requested_tool_name=tool_invocation.tool,
                available_tool_names_str=", ".join([t.name for t in self.tools]),
            )
        else:
            tool = self.tool_map[tool_invocation.tool]
            output = tool.invoke(tool_invocation.tool_input, config=config)
            return output

    async def _aexecute(
        self, tool_invocation: ToolInvocationInterface, *, config: RunnableConfig
    ) -> Any:
        if tool_invocation.tool not in self.tool_map:
            return self.invalid_tool_msg_template.format(
                requested_tool_name=tool_invocation.tool,
                available_tool_names_str=", ".join([t.name for t in self.tools]),
            )
        else:
            tool = self.tool_map[tool_invocation.tool]
            output = await tool.ainvoke(tool_invocation.tool_input, config=config)
            return output
