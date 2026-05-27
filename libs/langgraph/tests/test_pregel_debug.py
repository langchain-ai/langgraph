"""Tests for langgraph.pregel.debug helpers."""

from __future__ import annotations

from langgraph.pregel.debug import map_debug_tasks


class _FakeTask:
    """Minimal stand-in for PregelExecutableTask covering only what map_debug_tasks reads."""

    def __init__(
        self,
        *,
        id: str,
        name: str,
        input: object,
        triggers: list[str],
        config: dict | None,
    ) -> None:
        self.id = id
        self.name = name
        self.input = input
        self.triggers = triggers
        self.config = config


def test_map_debug_tasks_forwards_metadata_when_present() -> None:
    task = _FakeTask(
        id="t1",
        name="tools",
        input=[],
        triggers=["x"],
        config={
            "metadata": {"subagent_name": "weather_agent", "tool_call_id": "call_xyz"}
        },
    )
    payloads = list(map_debug_tasks([task]))
    assert len(payloads) == 1
    payload = payloads[0]
    assert payload["id"] == "t1"
    assert payload["name"] == "tools"
    assert payload["metadata"] == {
        "subagent_name": "weather_agent",
        "tool_call_id": "call_xyz",
    }


def test_map_debug_tasks_omits_metadata_when_empty() -> None:
    # Empty metadata dict in config: don't include metadata in the payload.
    task = _FakeTask(
        id="t1",
        name="tools",
        input=[],
        triggers=["x"],
        config={"metadata": {}},
    )
    payloads = list(map_debug_tasks([task]))
    assert "metadata" not in payloads[0]


def test_map_debug_tasks_omits_metadata_when_absent() -> None:
    # No metadata key in config: don't include metadata in the payload.
    task = _FakeTask(
        id="t1",
        name="tools",
        input=[],
        triggers=["x"],
        config={},
    )
    payloads = list(map_debug_tasks([task]))
    assert "metadata" not in payloads[0]


def test_map_debug_tasks_handles_none_config() -> None:
    # task.config can be None; should not crash.
    task = _FakeTask(
        id="t1",
        name="tools",
        input=[],
        triggers=["x"],
        config=None,
    )
    payloads = list(map_debug_tasks([task]))
    assert len(payloads) == 1
    assert "metadata" not in payloads[0]


def test_map_debug_tasks_filters_non_whitelisted_metadata_keys() -> None:
    """Internal langgraph metadata keys (thread_id, langgraph_*) must NOT
    appear in TaskPayload.metadata. Only the curated whitelist should pass.
    """
    task = _FakeTask(
        id="t1",
        name="tools",
        input=[],
        triggers=["x"],
        config={
            "metadata": {
                "subagent_name": "weather_agent",
                "tool_call_id": "call_xyz",
                # Internals that should NOT flow through:
                "thread_id": "thread-1",
                "langgraph_step": 1,
                "langgraph_node": "tools",
                "langgraph_path": ("__pregel_pull", "tools"),
            },
        },
    )
    payloads = list(map_debug_tasks([task]))
    assert len(payloads) == 1
    payload = payloads[0]
    assert payload["metadata"] == {
        "subagent_name": "weather_agent",
        "tool_call_id": "call_xyz",
    }
    assert "thread_id" not in payload["metadata"]
    assert "langgraph_step" not in payload["metadata"]
    assert "langgraph_node" not in payload["metadata"]


def test_map_debug_tasks_omits_metadata_when_only_internal_keys_present() -> None:
    """If config metadata contains only langgraph-internal keys (none of the
    whitelisted ones), TaskPayload.metadata should be omitted entirely.
    """
    task = _FakeTask(
        id="t1",
        name="tools",
        input=[],
        triggers=["x"],
        config={
            "metadata": {
                "thread_id": "thread-1",
                "langgraph_step": 1,
            },
        },
    )
    payloads = list(map_debug_tasks([task]))
    assert "metadata" not in payloads[0]
