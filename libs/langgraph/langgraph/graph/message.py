import uuid
from typing import (
    Annotated,
    Any,
    Literal,
    Optional,
    Protocol,
    Sequence,
    Union,
    runtime_checkable,
)

from pydantic import BaseModel
from typing_extensions import TypedDict


@runtime_checkable
class MessageProtocol(Protocol):
    content: Union[str, list]
    id: Optional[str]


class Message(BaseModel, extra="allow"):
    role: str
    content: Union[str, list]
    id: Optional[str] = None


MessageLike = Union[MessageProtocol, list[str], tuple[str, str], str, dict[str, Any]]

Messages = Union[list[MessageLike], MessageLike]


def convert_to_message(
    message: MessageLike,
) -> MessageProtocol:
    if isinstance(message, MessageProtocol):
        return message
    elif isinstance(message, str):
        return Message(role="user", content=message)
    elif isinstance(message, Sequence):
        return Message(role=message[0], content=message[1])
    elif isinstance(message, dict):
        return Message(**message)
    else:
        raise TypeError(f"Expected a message-like object, but got {type(message)}")


def add_messages(
    left: Messages,
    right: Messages,
    *,
    format: Optional[Literal["langchain-openai"]] = None,
) -> Messages:
    """Merges two lists of messages, updating existing messages by ID.

    By default, this ensures the state is "append-only", unless the
    new message has the same ID as an existing message.

    Args:
        left: The base list of messages.
        right: The list of messages (or single message) to merge
            into the base list.
        format: The format to return messages in. If None then messages will be
            returned as is. If 'langchain-openai' then messages will be returned as
            BaseMessage objects with their contents formatted to match OpenAI message
            format, meaning contents can be string, 'text' blocks, or 'image_url' blocks
            and tool responses are returned as their own ToolMessages.

            **REQUIREMENT**: Must have ``langchain-core>=0.3.11`` installed to use this
            feature.

    Returns:
        A new list of messages with the messages from `right` merged into `left`.
        If a message in `right` has the same ID as a message in `left`, the
        message from `right` will replace the message from `left`.

    Examples:
        ```pycon
        >>> from langchain_core.messages import AIMessage, HumanMessage
        >>> msgs1 = [HumanMessage(content="Hello", id="1")]
        >>> msgs2 = [AIMessage(content="Hi there!", id="2")]
        >>> add_messages(msgs1, msgs2)
        [HumanMessage(content='Hello', id='1'), AIMessage(content='Hi there!', id='2')]

        >>> msgs1 = [HumanMessage(content="Hello", id="1")]
        >>> msgs2 = [HumanMessage(content="Hello again", id="1")]
        >>> add_messages(msgs1, msgs2)
        [HumanMessage(content='Hello again', id='1')]

        >>> from typing import Annotated
        >>> from typing_extensions import TypedDict
        >>> from langgraph.graph import StateGraph
        >>>
        >>> class State(TypedDict):
        ...     messages: Annotated[list, add_messages]
        ...
        >>> builder = StateGraph(State)
        >>> builder.add_node("chatbot", lambda state: {"messages": [("assistant", "Hello")]})
        >>> builder.set_entry_point("chatbot")
        >>> builder.set_finish_point("chatbot")
        >>> graph = builder.compile()
        >>> graph.invoke({})
        {'messages': [AIMessage(content='Hello', id=...)]}

        >>> from typing import Annotated
        >>> from typing_extensions import TypedDict
        >>> from langgraph.graph import StateGraph, add_messages
        >>>
        >>> class State(TypedDict):
        ...     messages: Annotated[list, add_messages(format='langchain-openai')]
        ...
        >>> def chatbot_node(state: State) -> list:
        ...     return {"messages": [
        ...         {
        ...             "role": "user",
        ...             "content": [
        ...                 {
        ...                     "type": "text",
        ...                     "text": "Here's an image:",
        ...                     "cache_control": {"type": "ephemeral"},
        ...                 },
        ...                 {
        ...                     "type": "image",
        ...                     "source": {
        ...                         "type": "base64",
        ...                         "media_type": "image/jpeg",
        ...                         "data": "1234",
        ...                     },
        ...                 },
        ...             ]
        ...         },
        ...     ]}
        >>> builder = StateGraph(State)
        >>> builder.add_node("chatbot", chatbot_node)
        >>> builder.set_entry_point("chatbot")
        >>> builder.set_finish_point("chatbot")
        >>> graph = builder.compile()
        >>> graph.invoke({"messages": []})
        {
            'messages': [
                HumanMessage(
                    content=[
                        {"type": "text", "text": "Here's an image:"},
                        {
                            "type": "image_url",
                            "image_url": {"url": "data:image/jpeg;base64,1234"},
                        },
                    ],
                ),
            ]
        }
        ```

    ..versionchanged:: 0.2.61

        Support for 'format="langchain-openai"' flag added.
    """
    # coerce to list
    if not isinstance(left, list):
        left = [left]  # type: ignore[assignment]
    if not isinstance(right, list):
        right = [right]  # type: ignore[assignment]
    # coerce to message
    left = [convert_to_message(m) for m in left]
    right = [convert_to_message(m) for m in right]
    # assign missing ids
    for m in left:
        if m.id is None:
            m.id = str(uuid.uuid4())
    for m in right:
        if m.id is None:
            m.id = str(uuid.uuid4())
    # merge
    merged = left.copy()
    merged_by_id = {m.id: i for i, m in enumerate(merged)}
    ids_to_remove = set()
    for m in right:
        if (existing_idx := merged_by_id.get(m.id)) is not None:
            ids_to_remove.discard(m.id)
            merged[existing_idx] = m
        else:
            merged_by_id[m.id] = len(merged)
            merged.append(m)
    merged = [m for m in merged if m.id not in ids_to_remove]
    return merged


class MessagesState(TypedDict):
    messages: Annotated[list[MessageProtocol], add_messages]
