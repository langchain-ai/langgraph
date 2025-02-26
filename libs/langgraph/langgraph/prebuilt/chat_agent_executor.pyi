from langchain_core.language_models import LanguageModelInput, LanguageModelLike
from langchain_core.messages import BaseMessage, SystemMessage
from langchain_core.runnables import Runnable
from langchain_core.tools import BaseTool
from langgraph.graph.graph import CompiledGraph
from langgraph.graph.message import add_messages
from langgraph.managed import IsLastStep, RemainingSteps
from langgraph.prebuilt.tool_executor import ToolExecutor
from langgraph.prebuilt.tool_node import ToolNode
from langgraph.store.base import BaseStore
from langgraph.types import Checkpointer
from pydantic import BaseModel
from typing import Any, Callable, Literal, Sequence, TypeVar
from typing_extensions import Annotated, TypedDict

__all__ = ['create_react_agent', 'create_tool_calling_executor', 'AgentState']

StructuredResponse = dict | BaseModel
StructuredResponseSchema = dict | type[BaseModel]
F = TypeVar('F', bound=Callable[..., Any])

class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]
    is_last_step: IsLastStep
    remaining_steps: RemainingSteps

class AgentStateWithStructuredResponse(AgentState):
    structured_response: StructuredResponse
StateSchema = TypeVar('StateSchema', bound=AgentState)
StateSchemaType = type[StateSchema]
MessagesModifier = SystemMessage | str | Callable[[Sequence[BaseMessage]], LanguageModelInput] | Runnable[Sequence[BaseMessage], LanguageModelInput]
Prompt = SystemMessage | str | Callable[[StateSchema], LanguageModelInput] | Runnable[StateSchema, LanguageModelInput]

def create_react_agent(model: str | LanguageModelLike, tools: ToolExecutor | Sequence[BaseTool] | ToolNode, *, prompt: Prompt | None = None, response_format: StructuredResponseSchema | tuple[str, StructuredResponseSchema] | None = None, state_schema: StateSchemaType | None = None, config_schema: type[Any] | None = None, checkpointer: Checkpointer | None = None, store: BaseStore | None = None, interrupt_before: list[str] | None = None, interrupt_after: list[str] | None = None, debug: bool = False, version: Literal['v1', 'v2'] = 'v1', name: str | None = None) -> CompiledGraph: ...
create_tool_calling_executor = create_react_agent
