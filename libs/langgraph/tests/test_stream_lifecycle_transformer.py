"""Tests for LifecycleTransformer.

Consumes the `tasks` stream mode and emits subgraph lifecycle payloads
on the `lifecycle` channel for both in-process iteration via
`run.lifecycle` and wire delivery via `custom:lifecycle` protocol
events. These tests dispatch synthetic protocol events through a
`StreamMux` rather than running a real graph, to keep the inference
logic isolated.
"""

from __future__ import annotations

import time
from typing import Any

from langgraph.errors import GraphInterrupt
from langgraph.stream._mux import StreamMux
from langgraph.stream.transformers import (
    LifecyclePayload,
    LifecycleTransformer,
)

TS = int(time.time() * 1000)


def _tasks_start(
    namespace: list[str],
    *,
    task_id: str,
    name: str,
) -> dict[str, Any]:
    """Build a `tasks` ProtocolEvent carrying a TaskPayload (start)."""
    return {
        "type": "event",
        "method": "tasks",
        "params": {
            "namespace": namespace,
            "timestamp": TS,
            "data": {
                "id": task_id,
                "name": name,
                "input": None,
                "triggers": [],
            },
        },
    }


def _tasks_result(
    namespace: list[str],
    *,
    task_id: str,
    name: str,
    error: str | None = None,
    interrupts: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Build a `tasks` ProtocolEvent carrying a TaskResultPayload (finish)."""
    return {
        "type": "event",
        "method": "tasks",
        "params": {
            "namespace": namespace,
            "timestamp": TS,
            "data": {
                "id": task_id,
                "name": name,
                "error": error,
                "interrupts": interrupts or [],
                "result": {},
            },
        },
    }


def _arm(mux: StreamMux) -> None:
    """Force the lifecycle channel's log to accept pushes.

    `EventLog.push` is a no-op until a subscriber attaches. Tests that
    inspect `_items` directly need the gate flipped before any event
    is dispatched.
    """
    for transformer in mux._transformers:
        if isinstance(transformer, LifecycleTransformer):
            transformer._channel._log._subscribed = True


def _drain_lifecycle(mux: StreamMux) -> list[LifecyclePayload]:
    """Snapshot the lifecycle channel's underlying log."""
    transformer = mux.transformer_by_key("lifecycle")
    assert isinstance(transformer, LifecycleTransformer)
    return list(transformer._channel._log._items)


def _build_lifecycle_mux(*, scope: tuple[str, ...] = ()) -> StreamMux:
    mux = StreamMux([LifecycleTransformer(scope=scope)], is_async=False)
    _arm(mux)
    return mux


# ---------------------------------------------------------------------------
# LifecycleTransformer
# ---------------------------------------------------------------------------


def test_started_emitted_on_first_direct_child_task() -> None:
    mux = _build_lifecycle_mux()
    mux.push(_tasks_start(["agent:abc123"], task_id="t1", name="tool"))

    [payload] = _drain_lifecycle(mux)
    assert payload["event"] == "started"
    assert payload["namespace"] == ["agent:abc123"]
    assert payload["graph_name"] == "agent"
    assert payload["trigger_call_id"] == "abc123"


def test_started_dedup_on_repeat_namespace() -> None:
    mux = _build_lifecycle_mux()
    mux.push(_tasks_start(["agent:abc"], task_id="t1", name="a"))
    mux.push(_tasks_start(["agent:abc"], task_id="t2", name="b"))

    payloads = _drain_lifecycle(mux)
    assert [p["event"] for p in payloads] == ["started"]


def test_grandchild_namespace_ignored_at_root_scope() -> None:
    """A length-2 namespace is not a direct child of the root mux."""
    mux = _build_lifecycle_mux()
    mux.push(_tasks_start(["agent:abc", "tool:def"], task_id="t1", name="x"))

    assert _drain_lifecycle(mux) == []


def test_completed_on_parent_task_result() -> None:
    mux = _build_lifecycle_mux()
    mux.push(_tasks_start(["agent:abc"], task_id="t1", name="tool"))
    mux.push(_tasks_result([], task_id="abc", name="agent"))

    events = [p["event"] for p in _drain_lifecycle(mux)]
    assert events == ["started", "completed"]


def test_failed_on_parent_task_result_with_error() -> None:
    mux = _build_lifecycle_mux()
    mux.push(_tasks_start(["agent:abc"], task_id="t1", name="tool"))
    mux.push(_tasks_result([], task_id="abc", name="agent", error="boom"))

    payloads = _drain_lifecycle(mux)
    assert [p["event"] for p in payloads] == ["started", "failed"]
    assert payloads[1]["error"] == "boom"


def test_interrupted_on_parent_task_result_with_interrupts() -> None:
    mux = _build_lifecycle_mux()
    mux.push(_tasks_start(["agent:abc"], task_id="t1", name="tool"))
    mux.push(
        _tasks_result(
            [],
            task_id="abc",
            name="agent",
            interrupts=[{"value": "pause"}],
        )
    )

    payloads = _drain_lifecycle(mux)
    assert [p["event"] for p in payloads] == ["started", "interrupted"]


def test_interrupt_takes_precedence_over_error() -> None:
    mux = _build_lifecycle_mux()
    mux.push(_tasks_start(["agent:abc"], task_id="t1", name="tool"))
    mux.push(
        _tasks_result(
            [],
            task_id="abc",
            name="agent",
            error="should-be-suppressed",
            interrupts=[{"value": "pause"}],
        )
    )

    last = _drain_lifecycle(mux)[-1]
    assert last["event"] == "interrupted"
    assert "error" not in last


def test_finalize_completes_open_subgraphs() -> None:
    mux = _build_lifecycle_mux()
    mux.push(_tasks_start(["agent:abc"], task_id="t1", name="tool"))

    mux.close()
    payloads = _drain_lifecycle(mux)
    assert [p["event"] for p in payloads] == ["started", "completed"]


def test_fail_emits_interrupted_for_graph_interrupt() -> None:
    mux = _build_lifecycle_mux()
    mux.push(_tasks_start(["agent:abc"], task_id="t1", name="tool"))

    mux.fail(GraphInterrupt())
    payloads = _drain_lifecycle(mux)
    assert [p["event"] for p in payloads] == ["started", "interrupted"]
    assert "error" not in payloads[1]


def test_fail_emits_failed_for_other_exceptions() -> None:
    mux = _build_lifecycle_mux()
    mux.push(_tasks_start(["agent:abc"], task_id="t1", name="tool"))

    mux.fail(RuntimeError("boom"))
    payloads = _drain_lifecycle(mux)
    assert [p["event"] for p in payloads] == ["started", "failed"]
    assert payloads[1]["error"] == "boom"


def test_unrelated_methods_pass_through() -> None:
    """Non-`tasks` events are not consumed and don't emit lifecycle."""
    mux = _build_lifecycle_mux()
    mux.push(
        {
            "type": "event",
            "method": "values",
            "params": {"namespace": ["agent:abc"], "timestamp": TS, "data": {}},
        }
    )
    assert _drain_lifecycle(mux) == []


def test_scoped_transformer_filters_by_scope() -> None:
    """A transformer scoped to ('agent:abc',) emits only its direct children."""
    mux = _build_lifecycle_mux(scope=("agent:abc",))
    # Root-level task — out of scope.
    mux.push(_tasks_start(["other:1"], task_id="t1", name="other"))
    # Direct child of agent:abc — in scope.
    mux.push(_tasks_start(["agent:abc", "tool:def"], task_id="t2", name="tool"))
    # Grandchild of agent:abc — out of scope.
    mux.push(
        _tasks_start(["agent:abc", "tool:def", "deep:ghi"], task_id="t3", name="deep")
    )

    [payload] = _drain_lifecycle(mux)
    assert payload["namespace"] == ["agent:abc", "tool:def"]


def test_required_stream_modes_declared() -> None:
    assert LifecycleTransformer.required_stream_modes == ("tasks",)
