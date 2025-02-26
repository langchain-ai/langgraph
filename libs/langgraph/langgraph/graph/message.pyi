from langchain_core.messages import AnyMessage, BaseMessage as BaseMessage, MessageLikeRepresentation
from langgraph.graph.state import StateGraph
from typing import Annotated, Literal
from typing_extensions import TypedDict

Messages = list[MessageLikeRepresentation] | MessageLikeRepresentation

def add_messages(left: Messages, right: Messages, *, format: Literal['langchain-openai'] | None = None) -> Messages: ...

class MessageGraph(StateGraph):
    def __init__(self) -> None: ...

class MessagesState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]
