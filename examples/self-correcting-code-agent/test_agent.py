"""Unit tests for the self-correcting code agent."""

import pytest
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import Command

from agent import (
    MAX_RETRIES,
    AgentState,
    _extract_code,
    _format_execution_result,
    build_graph,
    execute_code,
    fix_code,
)


# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------

class TestExtractCode:
    def test_plain_code_unchanged(self):
        code = "print('hello')"
        assert _extract_code(code) == code

    def test_strips_python_fence(self):
        fenced = "```python\nprint('hello')\n```"
        assert _extract_code(fenced) == "print('hello')"

    def test_strips_generic_fence(self):
        fenced = "```\nprint('hello')\n```"
        assert _extract_code(fenced) == "print('hello')"

    def test_strips_whitespace(self):
        code = "  \nprint('hello')\n  "
        assert _extract_code(code) == "print('hello')"


class TestFormatExecutionResult:
    def test_success(self):
        result = _format_execution_result("42\n", "")
        assert "succeeded" in result
        assert "42" in result

    def test_failure(self):
        result = _format_execution_result("", "NameError: name 'x' is not defined")
        assert "failed" in result
        assert "NameError" in result

    def test_success_no_output(self):
        result = _format_execution_result("", "")
        assert "(no output)" in result


# ---------------------------------------------------------------------------
# Node-level tests (no LLM calls needed)
# ---------------------------------------------------------------------------

class TestExecuteCode:
    def _state(self, code: str) -> AgentState:
        return {
            "messages": [],
            "code": code,
            "execution_output": "",
            "execution_error": "",
            "retry_count": 0,
            "task": "test",
            "approved": True,
        }

    def test_successful_execution_captures_stdout(self):
        result = execute_code(self._state("print('hello world')"))
        assert result["execution_output"].strip() == "hello world"
        assert result["execution_error"] == ""

    def test_failed_execution_captures_error(self):
        result = execute_code(self._state("raise ValueError('oops')"))
        assert result["execution_output"] == ""
        assert "ValueError" in result["execution_error"]
        assert "oops" in result["execution_error"]

    def test_syntax_error_is_captured(self):
        result = execute_code(self._state("def broken(:"))
        assert "SyntaxError" in result["execution_error"]

    def test_output_and_no_error_on_multiline(self):
        code = "for i in range(3):\n    print(i)"
        result = execute_code(self._state(code))
        assert "0" in result["execution_output"]
        assert "2" in result["execution_output"]
        assert result["execution_error"] == ""


# ---------------------------------------------------------------------------
# Graph-level integration tests (mocked LLM via monkeypatching)
# ---------------------------------------------------------------------------

class TestGraphRouting:
    """Test that graph edges route correctly without hitting a real LLM."""

    def _make_graph_with_mock_llm(self, monkeypatch, code_sequence: list[str]):
        """Patch the module-level `llm` so nodes return pre-set code strings."""
        call_iter = iter(code_sequence)

        class FakeResponse:
            def __init__(self, content):
                self.content = content

        class FakeLLM:
            def invoke(self, messages):
                return FakeResponse(next(call_iter))

        import agent as agent_module
        monkeypatch.setattr(agent_module, "llm", FakeLLM())
        return build_graph(checkpointer=MemorySaver())

    def test_successful_run_ends_after_one_execution(self, monkeypatch):
        graph = self._make_graph_with_mock_llm(monkeypatch, ["print('ok')"])
        config = {"configurable": {"thread_id": "t1"}}

        # Run until interrupt
        list(graph.stream(
            {"task": "test", "retry_count": 0, "approved": False},
            config,
            stream_mode="values",
        ))

        # Resume with approval
        events = list(graph.stream(Command(resume=True), config, stream_mode="values"))
        final = graph.get_state(config)

        assert final.values["execution_output"].strip() == "ok"
        assert final.values["execution_error"] == ""
        assert final.values["retry_count"] == 0

    def test_rejection_ends_graph_immediately(self, monkeypatch):
        graph = self._make_graph_with_mock_llm(monkeypatch, ["print('ok')"])
        config = {"configurable": {"thread_id": "t2"}}

        list(graph.stream(
            {"task": "test", "retry_count": 0, "approved": False},
            config,
            stream_mode="values",
        ))

        # Human rejects
        list(graph.stream(Command(resume=False), config, stream_mode="values"))
        final = graph.get_state(config)

        # Graph ended — no execution happened
        assert final.values["execution_output"] == ""
        assert final.values["execution_error"] == ""

    def test_max_retries_stops_loop(self, monkeypatch):
        bad_code = "raise RuntimeError('always fails')"
        # First call = write_code, then MAX_RETRIES fix attempts
        graph = self._make_graph_with_mock_llm(
            monkeypatch, [bad_code] + [bad_code] * MAX_RETRIES
        )
        config = {"configurable": {"thread_id": "t3"}}

        list(graph.stream(
            {"task": "test", "retry_count": 0, "approved": False},
            config,
            stream_mode="values",
        ))

        # Approve each cycle (graph re-interrupts before each fix attempt)
        for _ in range(MAX_RETRIES + 1):
            snapshot = graph.get_state(config)
            if snapshot.next:
                list(graph.stream(Command(resume=True), config, stream_mode="values"))

        final = graph.get_state(config)
        assert final.values["retry_count"] >= MAX_RETRIES
