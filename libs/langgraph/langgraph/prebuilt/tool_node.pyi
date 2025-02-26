from _typeshed import Incomplete
from langchain_core.messages import AnyMessage as AnyMessage, ToolCall as ToolCall
from langchain_core.runnables import RunnableConfig as RunnableConfig
from langchain_core.tools import BaseTool, InjectedToolArg
from langgraph.store.base import BaseStore as BaseStore
from langgraph.utils.runnable import RunnableCallable
from pydantic import BaseModel as BaseModel
from typing import Any, Callable, Literal, Sequence

INVALID_TOOL_NAME_ERROR_TEMPLATE: str
TOOL_CALL_ERROR_TEMPLATE: str

def msg_content_output(output: Any) -> str | list[dict]: ...

class ToolNode(RunnableCallable):
    name: str
    tools_by_name: Incomplete
    tool_to_state_args: Incomplete
    tool_to_store_arg: Incomplete
    handle_tool_errors: Incomplete
    messages_key: Incomplete
    def __init__(self, tools: Sequence[BaseTool | Callable], *, name: str = 'tools', tags: list[str] | None = None, handle_tool_errors: bool | str | Callable[..., str] | tuple[type[Exception], ...] = True, messages_key: str = 'messages') -> None: ...
    def inject_tool_args(self, tool_call: ToolCall, input: list[AnyMessage] | dict[str, Any] | BaseModel, store: BaseStore | None) -> ToolCall: ...

def tools_condition(state: list[AnyMessage] | dict[str, Any] | BaseModel, messages_key: str = 'messages') -> Literal['tools', '__end__']: ...

class InjectedState(InjectedToolArg):
    field: Incomplete
    def __init__(self, field: str | None = None) -> None: ...

class InjectedStore(InjectedToolArg): ...
