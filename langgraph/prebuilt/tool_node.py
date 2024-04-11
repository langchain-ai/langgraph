import asyncio
import json
from typing import Any, Sequence, Union

from langchain_core.messages import AIMessage, AnyMessage, ToolCall, ToolMessage
from langchain_core.runnables import RunnableConfig
from langchain_core.runnables.config import get_executor_for_config
from langchain_core.tools import BaseTool

from langgraph.utils import RunnableCallable


def str_output(output: Any) -> str:
    if isinstance(output, str):
        return output
    else:
        try:
            return json.dumps(output)
        except Exception:
            return str(output)


class ToolNode(RunnableCallable):
    """
    A node that runs the tools requested in the last AIMessage. It can be used
    either in StateGraph with a "messages" key or in MessageGraph. If multiple
    tool calls are requested, they will be run in parallel. The output will be
    a list of ToolMessages, one for each tool call.
    """

    def __init__(
        self,
        tools: Sequence[BaseTool],
        *,
        name: str = "tools",
        tags: list[str] | None = None,
    ) -> None:
        super().__init__(self._func, self._afunc, name=name, tags=tags, trace=False)
        self.tools_by_name = {tool.name: tool for tool in tools}

    def _func(
        self, input: Union[list[AnyMessage], dict[str, Any]], config: RunnableConfig
    ) -> Any:
        if isinstance(input, list):
            output_type = "list"
            message: AnyMessage = input[-1]
        elif messages := input.get("messages", []):
            output_type = "dict"
            message = messages[-1]
        else:
            raise ValueError("No message found in input")

        if not isinstance(message, AIMessage):
            raise ValueError("Last message is not an AIMessage")

        def run_one(call: ToolCall):
            output = self.tools_by_name[call["name"]].invoke(call["args"], config)
            return ToolMessage(
                content=str_output(output), name=call["name"], tool_call_id=call["id"]
            )

        with get_executor_for_config(config) as executor:
            outputs = [*executor.map(run_one, message.tool_calls)]
            if output_type == "list":
                return outputs
            else:
                return {"messages": outputs}

    async def _afunc(
        self, input: Union[list[AnyMessage], dict[str, Any]], config: RunnableConfig
    ) -> Any:
        if isinstance(input, list):
            output_type = "list"
            message: AnyMessage = input[-1]
        elif messages := input.get("messages", []):
            output_type = "dict"
            message = messages[-1]
        else:
            raise ValueError("No message found in input")

        if not isinstance(message, AIMessage):
            raise ValueError("Last message is not an AIMessage")

        async def run_one(call: ToolCall):
            output = await self.tools_by_name[call["name"]].ainvoke(
                call["args"], config
            )
            return ToolMessage(
                content=str_output(output), name=call["name"], tool_call_id=call["id"]
            )

        outputs = await asyncio.gather(*(run_one(call) for call in message.tool_calls))
        if output_type == "list":
            return outputs
        else:
            return {"messages": outputs}
