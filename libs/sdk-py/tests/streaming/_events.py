"""Builders for protocol `Event` payloads used in tests.

Mirrors `libs/sdk/src/client/stream/test/event-builders.ts` from the JS SDK.
"""

from __future__ import annotations

from typing import Any


def _base(seq: int, method: str, namespace: list[str], data: Any) -> dict[str, Any]:
    return {
        "type": "event",
        "method": method,
        "params": {
            "namespace": namespace,
            "data": data,
        },
        "seq": seq,
        "event_id": f"evt-{seq}",
    }


def lifecycle_event(
    seq: int = 0, namespace: list[str] | None = None, **data: Any
) -> dict[str, Any]:
    return _base(seq, "lifecycle", namespace or [], data or {"phase": "started"})


def lifecycle_started_event(
    seq: int = 0, namespace: list[str] | None = None
) -> dict[str, Any]:
    """Lifecycle event with `phase="started"`."""
    return _base(seq, "lifecycle", namespace or [], {"phase": "started"})


def lifecycle_completed_event(
    seq: int = 0, namespace: list[str] | None = None
) -> dict[str, Any]:
    """Lifecycle event with `phase="completed"`."""
    return _base(seq, "lifecycle", namespace or [], {"phase": "completed"})


def lifecycle_errored_event(
    seq: int = 0,
    namespace: list[str] | None = None,
    error: str = "run errored",
) -> dict[str, Any]:
    """Lifecycle event with `phase="errored"` and an error message."""
    return _base(
        seq, "lifecycle", namespace or [], {"phase": "errored", "error": error}
    )


def values_event(
    seq: int = 0, namespace: list[str] | None = None, **data: Any
) -> dict[str, Any]:
    return _base(seq, "values", namespace or [], data or {"values": {}})


def custom_event(
    seq: int = 0, name: str = "ext", namespace: list[str] | None = None, **data: Any
) -> dict[str, Any]:
    payload = {"name": name, **data} if name else dict(data)
    return _base(seq, "custom", namespace or [], payload)


def input_requested_event(
    seq: int = 0, namespace: list[str] | None = None
) -> dict[str, Any]:
    return _base(seq, "input.requested", namespace or [], {"interrupt_id": "i-1"})


def message_start_event(
    seq: int = 0,
    namespace: list[str] | None = None,
    *,
    message_id: str = "msg-1",
    role: str = "ai",
    run_id: str = "run-1",
    node: str = "agent",
) -> dict[str, Any]:
    return _base(
        seq,
        "messages",
        namespace or [],
        {
            "event": "message-start",
            "id": message_id,
            "role": role,
            "metadata": {"run_id": run_id, "langgraph_node": node},
        },
    )


def message_text_delta_event(
    seq: int = 0,
    namespace: list[str] | None = None,
    *,
    text: str,
    index: int = 0,
) -> dict[str, Any]:
    return _base(
        seq,
        "messages",
        namespace or [],
        {
            "event": "content-block-delta",
            "index": index,
            "delta": {"type": "text-delta", "text": text},
        },
    )


def message_text_finish_event(
    seq: int = 0,
    namespace: list[str] | None = None,
    *,
    text: str,
    index: int = 0,
) -> dict[str, Any]:
    return _base(
        seq,
        "messages",
        namespace or [],
        {
            "event": "content-block-finish",
            "index": index,
            "content": {"type": "text", "text": text},
        },
    )


def message_finish_event(
    seq: int = 0,
    namespace: list[str] | None = None,
    *,
    input_tokens: int = 1,
    output_tokens: int = 1,
) -> dict[str, Any]:
    return _base(
        seq,
        "messages",
        namespace or [],
        {
            "event": "message-finish",
            "usage": {
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "total_tokens": input_tokens + output_tokens,
            },
        },
    )


def message_error_event(
    seq: int = 0,
    namespace: list[str] | None = None,
    *,
    message: str = "model failed",
    code: str = "provider_error",
) -> dict[str, Any]:
    return _base(
        seq,
        "messages",
        namespace or [],
        {"event": "error", "message": message, "code": code},
    )


def tool_started_event(
    seq: int = 0,
    namespace: list[str] | None = None,
    *,
    tool_call_id: str = "call-1",
    tool_name: str = "search",
    input: Any = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "event": "tool-started",
        "tool_call_id": tool_call_id,
        "tool_name": tool_name,
    }
    if input is not None:
        payload["input"] = input
    return _base(seq, "tools", namespace or [], payload)


def tool_output_delta_event(
    seq: int = 0,
    namespace: list[str] | None = None,
    *,
    tool_call_id: str = "call-1",
    delta: str = "",
) -> dict[str, Any]:
    return _base(
        seq,
        "tools",
        namespace or [],
        {
            "event": "tool-output-delta",
            "tool_call_id": tool_call_id,
            "delta": delta,
        },
    )


def tool_finished_event(
    seq: int = 0,
    namespace: list[str] | None = None,
    *,
    tool_call_id: str = "call-1",
    output: Any = None,
) -> dict[str, Any]:
    return _base(
        seq,
        "tools",
        namespace or [],
        {
            "event": "tool-finished",
            "tool_call_id": tool_call_id,
            "output": output,
        },
    )


def tool_error_event(
    seq: int = 0,
    namespace: list[str] | None = None,
    *,
    tool_call_id: str = "call-1",
    message: str = "tool failed",
    code: str = "tool_error",
) -> dict[str, Any]:
    return _base(
        seq,
        "tools",
        namespace or [],
        {
            "event": "tool-error",
            "tool_call_id": tool_call_id,
            "message": message,
            "code": code,
        },
    )
