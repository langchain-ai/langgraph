import uuid
from typing import Annotated, Union

from langchain_core.messages import (
    AnyMessage,
    convert_to_messages,
    message_chunk_to_message,
)

from langgraph.graph.state import StateGraph

Messages = Union[list[AnyMessage], AnyMessage]


def add_messages(left: Messages, right: Messages) -> Messages:
    # coerce to list
    if not isinstance(left, list):
        left = [left]
    if not isinstance(right, list):
        right = [right]
    # coerce to message
    left = [message_chunk_to_message(m) for m in convert_to_messages(left)]
    right = [message_chunk_to_message(m) for m in convert_to_messages(right)]
    # assign missing ids
    for m in left:
        if m.id is None:
            m.id = str(uuid.uuid4())
    for m in right:
        if m.id is None:
            m.id = str(uuid.uuid4())
    # merge
    left_idx_by_id = {m.id: i for i, m in enumerate(left)}
    merged = left.copy()
    for m in right:
        if (existing_idx := left_idx_by_id.get(m.id)) is not None:
            merged[existing_idx] = m
        else:
            merged.append(m)
    return merged


class MessageGraph(StateGraph):
    """A StateGraph where every node
    - receives a list of messages as input
    - returns one or more messages as output."""

    def __init__(self) -> None:
        super().__init__(Annotated[list[AnyMessage], add_messages])
