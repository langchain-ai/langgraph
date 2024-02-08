from typing import Annotated, Union

from langchain_core.messages import AnyMessage

from langgraph.graph.state import StateGraph

Messages = Union[list[AnyMessage], AnyMessage]


def add_messages(left: Messages, right: Messages) -> Messages:
    if not isinstance(left, list):
        left = [left]
    if not isinstance(right, list):
        right = [right]
    return left + right


class MessageGraph(StateGraph):
    """A StateGraph where every node
    - receives a list of messages as input
    - returns one or more messages as output."""

    def __init__(self) -> None:
        super().__init__(Annotated[list[AnyMessage], add_messages])
