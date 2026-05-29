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
        config={"metadata": {"lc_agent_name": "weather_agent"}},
    )
    payloads = list(map_debug_tasks([task]))
    assert len(payloads) == 1
    payload = payloads[0]
    assert payload["id"] == "t1"
    assert payload["name"] == "tools"
    assert payload["metadata"] == {"lc_agent_name": "weather_agent"}


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


def test_map_debug_tasks_forwards_all_metadata_keys() -> None:
    """All metadata keys flow through to TaskPayload.metadata, including
    framework keys. Mirrors stream_mode="messages" semantics, where the
    callback metadata dict is also forwarded whole. Consumers that don't
    want framework keys can filter them client-side.
    """
    md = {
        "lc_agent_name": "weather_agent",
        "thread_id": "thread-1",
        "langgraph_step": 1,
        "langgraph_node": "tools",
        "langgraph_path": ("__pregel_pull", "tools"),
    }
    task = _FakeTask(
        id="t1", name="tools", input=[], triggers=["x"], config={"metadata": md}
    )
    payloads = list(map_debug_tasks([task]))
    assert payloads[0]["metadata"] == md


def test_map_debug_tasks_metadata_is_copied_not_referenced() -> None:
    """Mutating the source config after emission must not affect the
    payload — TaskPayload.metadata is a defensive copy.
    """
    md = {"lc_agent_name": "a"}
    task = _FakeTask(
        id="t1", name="tools", input=[], triggers=["x"], config={"metadata": md}
    )
    payload = next(iter(map_debug_tasks([task])))
    md["lc_agent_name"] = "MUTATED"
    assert payload["metadata"]["lc_agent_name"] == "a"


def test_map_debug_tasks_folds_filtered_tags_into_metadata() -> None:
    """Config tags are folded into TaskPayload.metadata under `tags`, with
    langchain's internal `seq:step:*` tags filtered out — mirroring the
    messages stream handler so both channels surface the same tag set."""
    task = _FakeTask(
        id="t1",
        name="tools",
        input=[],
        triggers=["x"],
        config={
            "metadata": {"lc_agent_name": "weather_agent"},
            "tags": ["seq:step:1", "user-tag", "session-123"],
        },
    )
    payload = next(iter(map_debug_tasks([task])))
    assert payload["metadata"]["lc_agent_name"] == "weather_agent"
    assert payload["metadata"]["tags"] == ["user-tag", "session-123"]


def test_map_debug_tasks_omits_tags_when_only_seq_step() -> None:
    """If the only tags are internal `seq:step:*` markers, no `tags` key is
    added (matches the messages handler's `if filtered_tags:` guard)."""
    task = _FakeTask(
        id="t1",
        name="tools",
        input=[],
        triggers=["x"],
        config={"metadata": {"lc_agent_name": "a"}, "tags": ["seq:step:1"]},
    )
    payload = next(iter(map_debug_tasks([task])))
    assert "tags" not in payload["metadata"]


def test_map_debug_tasks_adds_tags_even_without_other_metadata() -> None:
    """Filtered tags surface even when config has no metadata dict."""
    task = _FakeTask(
        id="t1",
        name="tools",
        input=[],
        triggers=["x"],
        config={"tags": ["user-tag"]},
    )
    payload = next(iter(map_debug_tasks([task])))
    assert payload["metadata"] == {"tags": ["user-tag"]}
