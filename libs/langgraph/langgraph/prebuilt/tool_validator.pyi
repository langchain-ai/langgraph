from _typeshed import Incomplete
from langchain_core.messages import AnyMessage as AnyMessage, ToolCall as ToolCall
from langchain_core.runnables import RunnableConfig as RunnableConfig
from langchain_core.tools import BaseTool
from langgraph.utils.runnable import RunnableCallable
from pydantic import BaseModel
from typing import Callable, Sequence

class ValidationNode(RunnableCallable):
    schemas_by_name: Incomplete
    def __init__(self, schemas: Sequence[BaseTool | type[BaseModel] | Callable], *, format_error: Callable[[BaseException, ToolCall, type[BaseModel]], str] | None = None, name: str = 'validation', tags: list[str] | None = None) -> None: ...
