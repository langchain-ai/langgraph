from _typeshed import Incomplete
from langchain_core.load.serializable import Serializable
from langchain_core.runnables import RunnableConfig as RunnableConfig
from langchain_core.tools import BaseTool
from langgraph.utils.runnable import RunnableCallable
from typing import Callable, Sequence

INVALID_TOOL_MSG_TEMPLATE: str

class ToolInvocationInterface:
    tool: str
    tool_input: str | dict

class ToolInvocation(Serializable):
    tool: str
    tool_input: str | dict

class ToolExecutor(RunnableCallable):
    tools: Incomplete
    tool_map: Incomplete
    invalid_tool_msg_template: Incomplete
    def __init__(self, tools: Sequence[BaseTool | Callable], *, invalid_tool_msg_template: str = ...) -> None: ...
