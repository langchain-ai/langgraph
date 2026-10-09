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


def _normalize_lifecycle_data(data: dict[str, Any]) -> dict[str, Any]:
    """Map test-fixture shorthand to the wire shape `langgraph-api` emits.

    The server emits the lifecycle status as `data.event` with values
    `running` / `completed` / `failed` / `interrupted` (see
    `api/langgraph_api/event_streaming/event_normalizers.py::to_lifecycle_status`).
    The fixture historically accepted `phase=` and the legacy `"errored"`
    value; translate them so tests exercise the real wire format
    without touching every call site.
    """
    normalized = dict(data)
    if "phase" in normalized and "event" not in normalized:
        normalized["event"] = normalized.pop("phase")
    if normalized.get("event") == "errored":
        normalized["event"] = "failed"
    return normalized


def lifecycle_event(
    seq: int = 0, namespace: list[str] | None = None, **data: Any
) -> dict[str, Any]:
    payload = _normalize_lifecycle_data(data) if data else {"event": "started"}
    return _base(seq, "lifecycle", namespace or [], payload)


def lifecycle_started_event(
    seq: int = 0, namespace: list[str] | None = None
) -> dict[str, Any]:
    """Lifecycle event with `event="started"`."""
    return _base(seq, "lifecycle", namespace or [], {"event": "started"})


def lifecycle_completed_event(
    seq: int = 0, namespace: list[str] | None = None
) -> dict[str, Any]:
    """Lifecycle event with `event="completed"`."""
    return _base(seq, "lifecycle", namespace or [], {"event": "completed"})


def lifecycle_errored_event(
    seq: int = 0,
    namespace: list[str] | None = None,
    error: str = "run errored",
) -> dict[str, Any]:
    """Lifecycle event with `event="failed"` and an error message."""
    return _base(seq, "lifecycle", namespace or [], {"event": "failed", "error": error})


def values_event(
    seq: int = 0, namespace: list[str] | None = None, **data: Any
) -> dict[str, Any]:
    return _base(seq, "values", namespace or [], data or {"values": {}})


def updates_event(
    seq: int = 0, namespace: list[str] | None = None, **data: Any
) -> dict[str, Any]:
    return _base(seq, "updates", namespace or [], data or {})


def checkpoints_event(
    seq: int = 0, namespace: list[str] | None = None, **data: Any
) -> dict[str, Any]:
    return _base(seq, "checkpoints", namespace or [], data or {})


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
    run_id: str | None = None,
    node: str = "agent",
) -> dict[str, Any]:
    metadata: dict[str, Any] = {"langgraph_node": node}
    if run_id is not None:
        metadata["run_id"] = run_id
    return _base(
        seq,
        "messages",
        namespace or [],
        {
            "event": "message-start",
            "id": message_id,
            "role": role,
            "metadata": metadata,
        },
    )


def message_text_delta_event(
    seq: int = 0,
    namespace: list[str] | None = None,
    *,
    text: str,
    index: int = 0,
    message_id: str | None = None,
) -> dict[str, Any]:
    data: dict[str, Any] = {
        "event": "content-block-delta",
        "index": index,
        "delta": {"type": "text-delta", "text": text},
    }
    if message_id is not None:
        data["id"] = message_id
    return _base(seq, "messages", namespace or [], data)


def message_text_finish_event(
    seq: int = 0,
    namespace: list[str] | None = None,
    *,
    text: str,
    index: int = 0,
    message_id: str | None = None,
) -> dict[str, Any]:
    data: dict[str, Any] = {
        "event": "content-block-finish",
        "index": index,
        "content": {"type": "text", "text": text},
    }
    if message_id is not None:
        data["id"] = message_id
    return _base(seq, "messages", namespace or [], data)


def message_finish_event(
    seq: int = 0,
    namespace: list[str] | None = None,
    *,
    input_tokens: int = 1,
    output_tokens: int = 1,
    message_id: str | None = None,
) -> dict[str, Any]:
    data: dict[str, Any] = {
        "event": "message-finish",
        "usage": {
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": input_tokens + output_tokens,
        },
    }
    if message_id is not None:
        data["id"] = message_id
    return _base(seq, "messages", namespace or [], data)


def message_error_event(
    seq: int = 0,
    namespace: list[str] | None = None,
    *,
    message: str = "model failed",
    code: str = "provider_error",
    message_id: str | None = None,
) -> dict[str, Any]:
    data: dict[str, Any] = {"event": "error", "message": message, "code": code}
    if message_id is not None:
        data["id"] = message_id
    return _base(seq, "messages", namespace or [], data)


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


def tasks_start_event(
    seq: int = 0,
    namespace: list[str] | None = None,
    *,
    task_id: str = "task-1",
    name: str = "node",
    input: Any = None,
) -> dict[str, Any]:
    return _base(
        seq,
        "tasks",
        namespace or [],
        {
            "id": task_id,
            "name": name,
            "input": input,
            "triggers": [],
        },
    )


def tasks_result_event(
    seq: int = 0,
    namespace: list[str] | None = None,
    *,
    task_id: str = "task-1",
    name: str = "node",
    result: Any = None,
    error: str | None = None,
    interrupts: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    return _base(
        seq,
        "tasks",
        namespace or [],
        {
            "id": task_id,
            "name": name,
            "result": result if result is not None else {},
            "error": error,
            "interrupts": interrupts or [],
        },
    )
