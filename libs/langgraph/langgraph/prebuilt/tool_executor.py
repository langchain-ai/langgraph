from typing import Any, Callable, Sequence, Union

from langchain_core.load.serializable import Serializable
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import BaseTool
from langchain_core.tools import tool as create_tool

from langgraph._api.deprecation import deprecated
from langgraph.utils.runnable import RunnableCallable

INVALID_TOOL_MSG_TEMPLATE = (
    "{requested_tool_name} is not a valid tool, "
    "try one of [{available_tool_names_str}]."
)


@deprecated("0.2.0", "langgraph.prebuilt.ToolNode", removal="0.3.0")
class ToolInvocationInterface:
    """Interface for invoking a tool.

    Attributes:
        tool (str): The name of the tool to invoke.
        tool_input (Union[str, dict]): The input to pass to the tool.

    """

    tool: str
    tool_input: Union[str, dict]


@deprecated("0.2.0", "langgraph.prebuilt.ToolNode", removal="0.3.0")
class ToolInvocation(Serializable):
    """Information about how to invoke a tool.

    Attributes:
        tool (str): The name of the Tool to execute.
        tool_input (Union[str, dict]): The input to pass in to the Tool.

    Examples:
        Basic usage:
        ```pycon
        >>> invocation = ToolInvocation(
        ...    tool="search",
        ...     tool_input="What is the capital of France?"
        ... )
        ```
    """

    tool: str
    tool_input: Union[str, dict]


@deprecated("0.2.0", "langgraph.prebuilt.ToolNode", removal="0.3.0")
class ToolExecutor(RunnableCallable):
    """Executes a tool invocation.

    Args:
        tools (Sequence[BaseTool]): A sequence of tools that can be invoked.
        invalid_tool_msg_template (str, optional): The template for the error message
            when an invalid tool is requested. Defaults to INVALID_TOOL_MSG_TEMPLATE.

    Examples:
        Basic usage:

        ```pycon
        >>> from langchain_core.tools import tool
        >>> from langgraph.prebuilt.tool_executor import ToolExecutor, ToolInvocation
        ...
        ...
        >>> @tool
        ... def search(query: str) -> str:
        ...     \"\"\"Search engine.\"\"\"
        ...     return f"Searching for: {query}"
        ...
        ...
        >>> tools = [search]
        >>> executor = ToolExecutor(tools)
        ...
        >>> invocation = ToolInvocation(tool="search", tool_input="What is the capital of France?")
        >>> result = executor.invoke(invocation)
        >>> print(result)
        "Searching for: What is the capital of France?"
        ```
        Handling invalid tool:

        ```pycon
        >>> invocation = ToolInvocation(
        ...     tool="nonexistent", tool_input="What is the capital of France?"
        ... )
        >>> result = executor.invoke(invocation)
        >>> print(result)
        "nonexistent is not a valid tool, try one of [search]."
        ```
    """

    def __init__(
        self,
        tools: Sequence[Union[BaseTool, Callable]],
        *,
        invalid_tool_msg_template: str = INVALID_TOOL_MSG_TEMPLATE,
    ) -> None:
        super().__init__(self._execute, afunc=self._aexecute, trace=False)
        tools_ = [
            tool if isinstance(tool, BaseTool) else create_tool(tool) for tool in tools
        ]
        self.tools = tools_
        self.tool_map = {t.name: t for t in tools_}
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
