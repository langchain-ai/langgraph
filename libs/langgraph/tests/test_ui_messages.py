from types import SimpleNamespace
from uuid import UUID

from langchain_core.runnables.config import var_child_runnable_config

from langgraph._internal._constants import CONF, CONFIG_KEY_RUNTIME, CONFIG_KEY_SEND
from langgraph.graph.ui import push_ui_message


def _test_config(
    streamed: list[dict], writes: list[tuple[str, dict]], run_id: str | None
):
    runtime = SimpleNamespace(
        stream_writer=lambda event: streamed.append(event), store=None
    )

    def send(pairs):
        writes.extend(pairs)

    return {
        CONF: {
            CONFIG_KEY_RUNTIME: runtime,
            CONFIG_KEY_SEND: send,
        },
        **({"run_id": run_id} if run_id else {}),
    }


def test_push_ui_message_generates_collision_resistant_ids_for_identical_payloads():
    streamed: list[dict] = []
    writes: list[tuple[str, dict]] = []
    token = var_child_runnable_config.set(_test_config(streamed, writes, run_id=None))
    try:
        first = push_ui_message("artifact", {"value": "same"})
        second = push_ui_message("artifact", {"value": "same"})
    finally:
        var_child_runnable_config.reset(token)

    assert first["id"] != second["id"]
    UUID(first["id"])
    UUID(second["id"])


def test_push_ui_message_preserves_run_id_in_metadata():
    streamed: list[dict] = []
    writes: list[tuple[str, dict]] = []
    token = var_child_runnable_config.set(
        _test_config(streamed, writes, run_id="run-123")
    )
    try:
        message = push_ui_message("artifact", {"value": "x"})
    finally:
        var_child_runnable_config.reset(token)

    assert message["metadata"]["run_id"] == "run-123"
