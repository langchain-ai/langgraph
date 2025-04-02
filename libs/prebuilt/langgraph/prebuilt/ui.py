from typing import Any, Literal, Optional, Union
from uuid import uuid4

from langchain_core.messages import AnyMessage
from typing_extensions import TypedDict

from langgraph.constants import CONF, CONFIG_KEY_SEND
from langgraph.utils.config import get_config, get_stream_writer


class UIMessage(TypedDict):
    type: Literal["ui"]
    id: str
    name: str
    props: dict[str, Any]
    metadata: dict[str, Any]


class RemoveUIMessage(TypedDict):
    type: Literal["remove-ui"]
    id: str


AnyUIMessage = Union[UIMessage, RemoveUIMessage]


def push_ui_message(
    name: str,
    props: dict[str, Any],
    *,
    id: Optional[str] = None,
    metadata: Optional[dict[str, Any]] = None,
    message: Optional[AnyMessage] = None,
    state_key: str = "ui",
) -> UIMessage:
    """Push a new UI message."""
    writer = get_stream_writer()
    config = get_config()

    message_id = None
    if message:
        if isinstance(message, dict) and "id" in message:
            message_id = message.get("id")
        elif hasattr(message, "id"):
            message_id = message.id

    evt: UIMessage = {
        "type": "ui",
        "id": id or str(uuid4()),
        "name": name,
        "props": props,
        "metadata": {
            **(config.get("metadata") or {}),
            "tags": config.get("tags", None),
            "name": config.get("run_name", None),
            "run_id": config.get("run_id", None),
            **(metadata or {}),
            **({"message_id": message_id} if message_id else {}),
        },
    }

    writer(evt)
    config[CONF][CONFIG_KEY_SEND]([(state_key, evt)])

    return evt


def remove_ui_message(id: str, *, state_key: str = "ui") -> RemoveUIMessage:
    """Delete a UI message by ID."""
    writer = get_stream_writer()
    config = get_config()

    evt: RemoveUIMessage = {"type": "remove-ui", "id": id}

    writer(evt)
    config[CONF][CONFIG_KEY_SEND]([(state_key, evt)])

    return evt


def reduce_ui_messages(
    left: Union[list[AnyUIMessage], AnyUIMessage],
    right: Union[list[AnyUIMessage], AnyUIMessage],
) -> list[AnyUIMessage]:
    """Merge two lists of UI messages, supporting removing UI messages."""
    if not isinstance(left, list):
        left = [left]

    if not isinstance(right, list):
        right = [right]

    new_state = left.copy()
    for m in right:
        if m.get("type") == "remove-ui":
            new_state = [m for m in new_state if m.get("id") != m.get("id")]
        else:
            new_state.append(m)

    return new_state
