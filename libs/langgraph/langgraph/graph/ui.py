from __future__ import annotations

from typing import Any, Literal, cast
from uuid import uuid4

from langchain_core.messages import AnyMessage
from typing_extensions import TypedDict

from langgraph.config import get_config, get_stream_writer
from langgraph.constants import CONF

__all__ = (
    "UIMessage",
    "RemoveUIMessage",
    "AnyUIMessage",
    "push_ui_message",
    "delete_ui_message",
    "ui_message_reducer",
)


class UIMessage(TypedDict):
    """A message type for UI updates in LangGraph.

    This TypedDict represents a UI message that can be sent to update the UI state.
    It contains information about the UI component to render and its properties.

    Attributes:
        type: Literal type indicating this is a UI message.
        id: Unique identifier for the UI message.
        name: Name of the UI component to render.
        props: Properties to pass to the UI component.
        metadata: Additional metadata about the UI message.
    """

    type: Literal["ui"]
    id: str
    name: str
    props: dict[str, Any]
    metadata: dict[str, Any]


class RemoveUIMessage(TypedDict):
    """A message type for removing UI components in LangGraph.

    This TypedDict represents a message that can be sent to remove a UI component
    from the current state.

    Attributes:
        type: Literal type indicating this is a remove-ui message.
        id: Unique identifier of the UI message to remove.
    """

    type: Literal["remove-ui"]
    id: str


AnyUIMessage = UIMessage | RemoveUIMessage


def push_ui_message(
    name: str,
    props: dict[str, Any],
    *,
    id: str | None = None,
    metadata: dict[str, Any] | None = None,
    message: AnyMessage | None = None,
    state_key: str | None = "ui",
    merge: bool = False,
) -> UIMessage:
    """Push a new UI message to update the UI state.

    This function creates and sends a UI message that will be rendered in the UI.
    It also updates the graph state with the new UI message.

    Args:
        name: Name of the UI component to render.
        props: Properties to pass to the UI component.
        id: Optional unique identifier for the UI message.
            If not provided, a random UUID will be generated.
        metadata: Optional additional metadata about the UI message.
        message: Optional message object to associate with the UI message.
        state_key: Key in the graph state where the UI messages are stored.
            Defaults to "ui".
        merge: Whether to merge props with existing UI message (True) or replace
            them (False). Defaults to False.

    Returns:
        The created UI message.

    Example:
        ```python
        push_ui_message(
            name="component-name",
            props={"content": "Hello world"},
        )
        ```

    """
    from langgraph._internal._constants import CONFIG_KEY_SEND

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
            "merge": merge,
            "run_id": config.get("run_id", None),
            "tags": config.get("tags", None),
            "name": config.get("run_name", None),
            **(metadata or {}),
            **({"message_id": message_id} if message_id else {}),
        },
    }

    writer(evt)
    if state_key:
        config[CONF][CONFIG_KEY_SEND]([(state_key, evt)])

    return evt


def delete_ui_message(id: str, *, state_key: str = "ui") -> RemoveUIMessage:
    """Delete a UI message by ID from the UI state.

    This function creates and sends a message to remove a UI component from the current state.
    It also updates the graph state to remove the UI message.

    Args:
        id: Unique identifier of the UI component to remove.
        state_key: Key in the graph state where the UI messages are stored. Defaults to "ui".

    Returns:
        The remove UI message.

    Example:
        ```python
        delete_ui_message("message-123")
        ```

    """
    from langgraph._internal._constants import CONFIG_KEY_SEND

    writer = get_stream_writer()
    config = get_config()

    evt: RemoveUIMessage = {"type": "remove-ui", "id": id}

    writer(evt)
    config[CONF][CONFIG_KEY_SEND]([(state_key, evt)])

    return evt


def ui_message_reducer(
    left: list[AnyUIMessage] | AnyUIMessage,
    right: list[AnyUIMessage] | AnyUIMessage,
) -> list[AnyUIMessage]:
    """Merge two lists of UI messages, supporting removing UI messages.

    This function combines two lists of UI messages, handling both regular UI messages
    and `remove-ui` messages. When a `remove-ui` message is encountered, it removes any
    UI message with the matching ID from the current state.

    Args:
        left: First list of UI messages or single UI message.
        right: Second list of UI messages or single UI message.

    Returns:
        Combined list of UI messages with removals applied.

    Example:
        ```python
        messages = ui_message_reducer(
            [{"type": "ui", "id": "1", "name": "Chat", "props": {}}],
            {"type": "remove-ui", "id": "1"},
        )
        ```

    """
    if not isinstance(left, list):
        left = [left]

    if not isinstance(right, list):
        right = [right]

    # merge messages
    merged = left.copy()
    merged_by_id = {m.get("id"): i for i, m in enumerate(merged)}
    ids_to_remove = set()

    for msg in right:
        msg_id = msg.get("id")

        if (existing_idx := merged_by_id.get(msg_id)) is not None:
            if msg.get("type") == "remove-ui":
                ids_to_remove.add(msg_id)
            else:
                ids_to_remove.discard(msg_id)

                if cast(UIMessage, msg).get("metadata", {}).get("merge", False):
                    prev_msg = merged[existing_idx]
                    msg = msg.copy()
                    msg["props"] = {**prev_msg["props"], **msg["props"]}

                merged[existing_idx] = msg
        else:
            if msg.get("type") == "remove-ui":
                raise ValueError(
                    f"Attempting to delete an UI message with an ID that doesn't exist ('{msg_id}')"
                )

            merged_by_id[msg_id] = len(merged)
            merged.append(msg)

    merged = [m for m in merged if m.get("id") not in ids_to_remove]
    return merged
