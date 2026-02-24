"""Type-narrowing tests for v2 streaming.

Runs mypy on inline code snippets to verify that the StreamPart discriminated
union narrows `data` to the correct type when checking `part["type"]`.
"""

from __future__ import annotations

import subprocess
import sys
import tempfile
import textwrap
from pathlib import Path

_MYPY_PREAMBLE = textwrap.dedent("""\
    from langgraph.types import (
        StreamPart,
        ValuesStreamPart,
        UpdatesStreamPart,
        MessagesStreamPart,
        CustomStreamPart,
        CheckpointStreamPart,
        TasksStreamPart,
        DebugStreamPart,
        CheckpointPayload,
        DebugPayload,
        TaskPayload,
        TaskResultPayload,
    )
    from typing import Any
    from typing_extensions import assert_type
    from langchain_core.messages import AnyMessage

    def check(part: StreamPart) -> None:
""")


def _run_mypy(code: str) -> subprocess.CompletedProcess[str]:
    """Run mypy on a code snippet written to a temp file."""
    full_code = _MYPY_PREAMBLE + textwrap.indent(code, "    ")
    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
        f.write(full_code)
        f.flush()
        tmp_path = f.name
    try:
        return subprocess.run(
            [sys.executable, "-m", "mypy", "--no-error-summary", tmp_path],
            capture_output=True,
            text=True,
        )
    finally:
        Path(tmp_path).unlink(missing_ok=True)


class TestStreamPartTypeNarrowing:
    """Verify that mypy narrows StreamPart.data correctly after type checks."""

    def test_values_narrows_data_to_dict(self) -> None:
        result = _run_mypy(
            textwrap.dedent("""\
            if part["type"] == "values":
                assert_type(part["data"], dict[str, Any])
        """)
        )
        assert result.returncode == 0, result.stdout

    def test_updates_narrows_data_to_dict(self) -> None:
        result = _run_mypy(
            textwrap.dedent("""\
            if part["type"] == "updates":
                assert_type(part["data"], dict[str, Any])
        """)
        )
        assert result.returncode == 0, result.stdout

    def test_messages_narrows_data_to_tuple(self) -> None:
        result = _run_mypy(
            textwrap.dedent("""\
            if part["type"] == "messages":
                assert_type(part["data"], tuple[AnyMessage, dict[str, Any]])
        """)
        )
        assert result.returncode == 0, result.stdout

    def test_custom_narrows_data_to_any(self) -> None:
        result = _run_mypy(
            textwrap.dedent("""\
            if part["type"] == "custom":
                assert_type(part["data"], Any)
        """)
        )
        assert result.returncode == 0, result.stdout

    def test_checkpoints_narrows_data_to_checkpoint_payload(self) -> None:
        result = _run_mypy(
            textwrap.dedent("""\
            if part["type"] == "checkpoints":
                assert_type(part["data"], CheckpointPayload)
        """)
        )
        assert result.returncode == 0, result.stdout

    def test_tasks_narrows_data_to_task_payload_union(self) -> None:
        result = _run_mypy(
            textwrap.dedent("""\
            if part["type"] == "tasks":
                assert_type(part["data"], TaskPayload | TaskResultPayload)
        """)
        )
        assert result.returncode == 0, result.stdout

    def test_debug_narrows_data_to_debug_payload(self) -> None:
        result = _run_mypy(
            textwrap.dedent("""\
            if part["type"] == "debug":
                assert_type(part["data"], DebugPayload)
        """)
        )
        assert result.returncode == 0, result.stdout

    def test_ns_is_always_tuple(self) -> None:
        result = _run_mypy(
            textwrap.dedent("""\
            assert_type(part["ns"], tuple[str, ...])
        """)
        )
        assert result.returncode == 0, result.stdout

    def test_wrong_type_assignment_fails(self) -> None:
        """Verify that assigning the wrong type after narrowing is caught."""
        result = _run_mypy(
            textwrap.dedent("""\
            if part["type"] == "values":
                x: tuple[str, ...] = part["data"]  # should fail: data is dict
        """)
        )
        assert result.returncode != 0, (
            "Expected mypy to reject assigning dict to tuple, but it passed"
        )


_INVOKE_PREAMBLE = textwrap.dedent("""\
    from langgraph.types import StreamPart
    from langgraph.graph import StateGraph
    from langgraph.constants import START, END
    from typing import Any
    from typing_extensions import TypedDict, assert_type

    class State(TypedDict):
        value: str

    builder = StateGraph(State)
    builder.add_edge(START, END)
    graph = builder.compile()
    inp: Any = {"value": "x"}
""")


def _run_mypy_invoke(code: str) -> subprocess.CompletedProcess[str]:
    """Run mypy on invoke-related code snippets."""
    full_code = _INVOKE_PREAMBLE + code
    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
        f.write(full_code)
        f.flush()
        tmp_path = f.name
    try:
        return subprocess.run(
            [sys.executable, "-m", "mypy", "--no-error-summary", tmp_path],
            capture_output=True,
            text=True,
        )
    finally:
        Path(tmp_path).unlink(missing_ok=True)


class TestInvokeOverloadTypes:
    """Verify that invoke/ainvoke overloads narrow return types based on version and stream_mode."""

    def test_invoke_v1_returns_dict_or_any(self) -> None:
        result = _run_mypy_invoke(
            textwrap.dedent("""\
            r = graph.invoke(inp)
            assert_type(r, dict[str, Any] | Any)
        """)
        )
        assert result.returncode == 0, result.stdout

    def test_invoke_v2_values_returns_dict(self) -> None:
        """v2 invoke with default stream_mode='values' returns dict[str, Any]."""
        result = _run_mypy_invoke(
            textwrap.dedent("""\
            r = graph.invoke(inp, version="v2")
            assert_type(r, dict[str, Any])
        """)
        )
        assert result.returncode == 0, result.stdout

    def test_invoke_v2_non_values_returns_list_streampart(self) -> None:
        """v2 invoke with non-values stream_mode returns list[StreamPart]."""
        result = _run_mypy_invoke(
            textwrap.dedent("""\
            r = graph.invoke(inp, stream_mode="updates", version="v2")
            assert_type(r, list[StreamPart])
        """)
        )
        assert result.returncode == 0, result.stdout

    def test_invoke_v2_values_assignable_to_dict(self) -> None:
        """v2 invoke with default values mode can be assigned to dict."""
        result = _run_mypy_invoke(
            textwrap.dedent("""\
            r: dict[str, Any] = graph.invoke(inp, version="v2")
        """)
        )
        assert result.returncode == 0, result.stdout
