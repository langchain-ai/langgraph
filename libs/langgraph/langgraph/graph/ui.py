from __future__ import annotations

import hashlib
import hmac
import json
import logging
import os
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

logger = logging.getLogger(__name__)

_PROVENANCE_SECRET = os.environ.get("LANGGRAPH_PROVENANCE_SECRET", "langgraph-provenance-default-secret")
_PROPS_MAX_KEYS = 32
_PROPS_MAX_VALUE_LEN = 4096
_PROPS_ALLOWED_VALUE_TYPES = (str, int, float, bool, type(None))

_HITL_DELETE_APPROVER: Any = None


def set_hitl_delete_approver(approver: Any) -> None:
    """Set a Human-in-the-Loop approver callable for delete operations.

    The approver must be a callable that accepts (id: str, state_key: str) and
    returns True if the deletion is approved, False otherwise.
    """
    global _HITL_DELETE_APPROVER
    _HITL_DELETE_APPROVER = approver


def _sign_provenance(data: dict) -> str:
    """Generate an HMAC-SHA256 signature for provenance metadata."""
    payload = json.dumps(data, sort_keys=True, default=str).encode("utf-8")
    return hmac.new(
        _PROVENANCE_SECRET.encode("utf-8"),
        payload,
        hashlib.sha256,
    ).hexdigest()


def _verify_provenance(data: dict, signature: str) -> bool:
    """Verify the HMAC-SHA256 provenance signature of a message."""
    expected = _sign_provenance(data)
    return hmac.compare_digest(expected, signature)


def _minimise_props(props: dict[str, Any]) -> dict[str, Any]:
    """Apply output data minimisation to the props dictionary.

    Enforces:
    - Maximum number of keys (_PROPS_MAX_KEYS)
    - Maximum string value length (_PROPS_MAX_VALUE_LEN)
    - Only allowed primitive value types are forwarded; complex objects are redacted.
    """
    minimised: dict[str, Any] = {}
    for i, (k, v) in enumerate(props.items()):
        if i >= _PROPS_MAX_KEYS:
            logger.warning(
                "push_ui_message: props truncated at %d keys for data minimisation.",
                _PROPS_MAX_KEYS,
            )
            break
        if isinstance(v, str):
            if len(v) > _PROPS_MAX_VALUE_LEN:
                v = v[:_PROPS_MAX_VALUE_LEN]
                logger.warning(
                    "push_ui_message: props key '%s' value truncated to %d chars.",
                    k,
                    _PROPS_MAX_VALUE_LEN,
                )
        elif not isinstance(v, _PROPS_ALLOWED_VALUE_TYPES):
            logger.warning(
                "push_ui_message: props key '%s' redacted (type %s not allowed).",
                k,
                type(v).__name__,
            )
            v = "[redacted]"
        minimised[k] = v
    return minimised


def _require_hitl_approval_for_delete(id: str, state_key: str) -> None:
    """Enforce Human-in-the-Loop approval for delete operations.

    If a HITL approver has been registered via set_hitl_delete_approver, it will
    be called with (id, state_key). If it returns False or raises, the deletion
    is blocked.

    If no approver is registered, a default interactive prompt is used when
    running in an interactive terminal, otherwise the deletion is blocked.
    """
    if _HITL_DELETE_APPROVER is not None:
        try:
            approved = _HITL_DELETE_APPROVER(id, state_key)
        except Exception as exc:
            raise PermissionError(
                f"HITL approver raised an exception for delete of UI message '{id}': {exc}"
            ) from exc
        if not approved:
            raise PermissionError(
                f"Human-in-the-Loop approval denied for delete of UI message '{id}'."
            )
        return

    # Default: interactive prompt when stdin is a tty, otherwise block.
    if os.isatty(0):
        answer = input(
            f"[HITL] Approve deletion of UI message id='{id}' from state_key='{state_key}'? [y/N]: "
        ).strip().lower()
        if answer not in ("y", "yes"):
            raise PermissionError(
                f"Human-in-the-Loop approval denied for delete of UI message '{id}'."
            )
    else:
        raise PermissionError(
            f"Human-in-the-Loop approval required for delete of UI message '{id}' "
            "but no approver is registered and no interactive terminal is available. "
            "Register an approver via set_hitl_delete_approver()."
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
        merge: Whether to merge props with existing UI message (True) or replace
            them (False).

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

    # Apply output data minimisation to props before including in the event.
    minimised_props = _minimise_props(props)

    msg_id = id or str(uuid4())

    # Build provenance metadata and sign it for AI-content labeling/watermarking.
    provenance_data: dict[str, Any] = {
        "ai_generated": True,
        "watermark": "langgraph-ui",
        "run_id": config.get("run_id", None),
        "tags": config.get("tags", None),
        "name": config.get("run_name", None),
        "merge": merge,
        **(metadata or {}),
        **({"message_id": message_id} if message_id else {}),
    }
    provenance_signature = _sign_provenance(provenance_data)

    evt: UIMessage = {
        "type": "ui",
        "id": msg_id,
        "name": name,
        "props": minimised_props,
        "metadata": {
            **provenance_data,
            "provenance_signature": provenance_signature,
        },
    }

    # Fail-safe: block serving unlabeled content if writer or send raises.
    try:
        writer(evt)
    except Exception as exc:
        raise RuntimeError(
            f"push_ui_message: failed to stream UI event '{msg_id}' — "
            "blocking to prevent unlabeled AI content from being served."
        ) from exc

    if state_key:
        try:
            config[CONF][CONFIG_KEY_SEND]([(state_key, evt)])
        except Exception as exc:
            raise RuntimeError(
                f"push_ui_message: failed to send UI event '{msg_id}' to state — "
                "blocking to prevent unlabeled AI content from being served."
            ) from exc

    return evt


def delete_ui_message(id: str, *, state_key: str = "ui") -> RemoveUIMessage:
    """Delete a UI message by ID from the UI state.

    This function creates and sends a message to remove a UI component from the current state.
    It also updates the graph state to remove the UI message.

    Requires Human-in-the-Loop (HITL) approval before executing the deletion.
    Register an approver via set_hitl_delete_approver(), or run in an interactive
    terminal to be prompted.

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

    # Enforce HITL approval before proceeding with deletion.
    _require_hitl_approval_for_delete(id, state_key)

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

    Incoming messages are verified for AI provenance markers before being accepted.

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

        # Verify AI provenance on ingest for non-remove messages.
        if msg.get("type") != "remove-ui":
            msg_metadata = cast(UIMessage, msg).get("metadata", {})
            provenance_signature = msg_metadata.get("provenance_signature")
            if provenance_signature is None:
                logger.warning(
                    "ui_message_reducer: rejecting message id='%s' — missing provenance signature.",
                    msg_id,
                )
                continue
            # Reconstruct the provenance data for verification (exclude the signature itself).
            provenance_data = {k: v for k, v in msg_metadata.items() if k != "provenance_signature"}
            if not _verify_provenance(provenance_data, provenance_signature):
                logger.warning(
                    "ui_message_reducer: rejecting message id='%s' — invalid provenance signature.",
                    msg_id,
                )
                continue

        if (existing_idx := merged_by_id.get(msg_id)) is not None:
            if msg.get("type") == "remove-ui":
                # HITL approval for reducer-driven removals is enforced here.
                _require_hitl_approval_for_delete(msg_id, "ui")
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