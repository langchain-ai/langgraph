import uuid
import warnings
from functools import partial
from typing import (
    Annotated,
    Any,
    Callable,
    Literal,
    Optional,
    Sequence,
    Union,
    cast,
)

from langchain_core.messages import (
    AnyMessage,
    BaseMessage,
    BaseMessageChunk,
    MessageLikeRepresentation,
    RemoveMessage,
    convert_to_messages,
    message_chunk_to_message,
)
from typing_extensions import TypedDict

from langgraph.graph.state import StateGraph

Messages = Union[list[MessageLikeRepresentation], MessageLikeRepresentation]


def _add_messages_wrapper(func: Callable) -> Callable[[Messages, Messages], Messages]:
    def _add_messages(
        left: Optional[Messages] = None, right: Optional[Messages] = None, **kwargs: Any
    ) -> Union[Messages, Callable[[Messages, Messages], Messages]]:
        if left is not None and right is not None:
            return func(left, right, **kwargs)
        elif left is not None or right is not None:
            msg = (
                f"Must specify non-null arguments for both 'left' and 'right'. Only "
                f"received: '{'left' if left else 'right'}'."
            )
            raise ValueError(msg)
        else:
            return partial(func, **kwargs)

    _add_messages.__doc__ = func.__doc__
    return cast(Callable[[Messages, Messages], Messages], _add_messages)


@_add_messages_wrapper
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
    left = [
        message_chunk_to_message(cast(BaseMessageChunk, m))
        for m in convert_to_messages(left)
    ]
    right = [
        message_chunk_to_message(cast(BaseMessageChunk, m))
        for m in convert_to_messages(right)
    ]
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
            if isinstance(m, RemoveMessage):
                ids_to_remove.add(m.id)
            else:
                ids_to_remove.discard(m.id)
                merged[existing_idx] = m
        else:
            if isinstance(m, RemoveMessage):
                raise ValueError(
                    f"Attempting to delete a message with an ID that doesn't exist ('{m.id}')"
                )

            merged_by_id[m.id] = len(merged)
            merged.append(m)
    merged = [m for m in merged if m.id not in ids_to_remove]

    if format == "langchain-openai":
        merged = _format_messages(merged)
    elif format:
        msg = f"Unrecognized {format=}. Expected one of 'langchain-openai', None."
        raise ValueError(msg)
    else:
        pass

    return merged


class MessageGraph(StateGraph):
    """A StateGraph where every node receives a list of messages as input and returns one or more messages as output.

    MessageGraph is a subclass of StateGraph whose entire state is a single, append-only* list of messages.
    Each node in a MessageGraph takes a list of messages as input and returns zero or more
    messages as output. The `add_messages` function is used to merge the output messages from each node
    into the existing list of messages in the graph's state.

    Examples:
        ```pycon
        >>> from langgraph.graph.message import MessageGraph
        ...
        >>> builder = MessageGraph()
        >>> builder.add_node("chatbot", lambda state: [("assistant", "Hello!")])
        >>> builder.set_entry_point("chatbot")
        >>> builder.set_finish_point("chatbot")
        >>> builder.compile().invoke([("user", "Hi there.")])
        [HumanMessage(content="Hi there.", id='...'), AIMessage(content="Hello!", id='...')]
        ```

        ```pycon
        >>> from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
        >>> from langgraph.graph.message import MessageGraph
        ...
        >>> builder = MessageGraph()
        >>> builder.add_node(
        ...     "chatbot",
        ...     lambda state: [
        ...         AIMessage(
        ...             content="Hello!",
        ...             tool_calls=[{"name": "search", "id": "123", "args": {"query": "X"}}],
        ...         )
        ...     ],
        ... )
        >>> builder.add_node(
        ...     "search", lambda state: [ToolMessage(content="Searching...", tool_call_id="123")]
        ... )
        >>> builder.set_entry_point("chatbot")
        >>> builder.add_edge("chatbot", "search")
        >>> builder.set_finish_point("search")
        >>> builder.compile().invoke([HumanMessage(content="Hi there. Can you search for X?")])
        {'messages': [HumanMessage(content="Hi there. Can you search for X?", id='b8b7d8f4-7f4d-4f4d-9c1d-f8b8d8f4d9c1'),
                     AIMessage(content="Hello!", id='f4d9c1d8-8d8f-4d9c-b8b7-d8f4f4d9c1d8'),
                     ToolMessage(content="Searching...", id='d8f4f4d9-c1d8-4f4d-b8b7-d8f4f4d9c1d8', tool_call_id="123")]}
        ```
    """

    def __init__(self) -> None:
        super().__init__(Annotated[list[AnyMessage], add_messages])  # type: ignore[arg-type]


class MessagesState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]


def _format_messages(messages: Sequence[BaseMessage]) -> list[BaseMessage]:
    try:
        from langchain_core.messages import convert_to_openai_messages
    except ImportError:
        msg = (
            "Must have langchain-core>=0.3.11 installed to use automatic message "
            "formatting (format='langchain-openai'). Please update your langchain-core "
            "version or remove the 'format' flag. Returning un-formatted "
            "messages."
        )
        warnings.warn(msg)
        return list(messages)
    else:
        return convert_to_messages(convert_to_openai_messages(messages))
