"""End-to-end tests for `langgraph lint` via CliRunner + real graph modules."""

from __future__ import annotations

import builtins
import json
import pathlib

import pytest
from click.testing import CliRunner

from langgraph_cli.cli import cli

CLEAN_GRAPH = """\
from typing_extensions import TypedDict

from langgraph.graph import END, START, StateGraph


class State(TypedDict):
    value: str


def node(state):
    return {}


builder = StateGraph(State)
builder.add_node("a", node)
builder.add_edge(START, "a")
builder.add_edge("a", END)
graph = builder.compile()
"""

UNREACHABLE_NODE_GRAPH = """\
from typing_extensions import TypedDict

from langgraph.graph import END, START, StateGraph


class State(TypedDict):
    value: str


def node(state):
    return {}


builder = StateGraph(State)
builder.add_node("a", node)
builder.add_node("orphan", node)
builder.add_edge(START, "a")
builder.add_edge("a", END)
graph = builder.compile()
"""

UNBOUNDED_CYCLE_GRAPH = """\
from typing_extensions import TypedDict

from langgraph.graph import START, StateGraph


class State(TypedDict):
    value: str


def node(state):
    return {}


def always_loop(state):
    return "loop"


builder = StateGraph(State)
builder.add_node("stuck", node)
builder.add_edge(START, "stuck")
builder.add_conditional_edges("stuck", always_loop, {"loop": "stuck"})
graph = builder.compile()
"""

FACTORY_GRAPH = """\
def make_graph(config):
    raise NotImplementedError("should never be called by lint")
"""


def _write_project(
    tmp_path: pathlib.Path, graph_source: str, attr: str = "graph"
) -> pathlib.Path:
    (tmp_path / "agent.py").write_text(graph_source)
    config = {
        "dependencies": ["."],
        "graphs": {"agent": f"./agent.py:{attr}"},
    }
    config_path = tmp_path / "langgraph.json"
    config_path.write_text(json.dumps(config))
    return config_path


def test_lint_clean_graph_exits_zero(tmp_path: pathlib.Path) -> None:
    config_path = _write_project(tmp_path, CLEAN_GRAPH)
    result = CliRunner().invoke(cli, ["lint", "--config", str(config_path)])
    assert result.exit_code == 0, result.output
    assert "clean" in result.output
    assert "production debt score 0" in result.output


def test_lint_unreachable_node_warns_but_default_exit_zero(
    tmp_path: pathlib.Path,
) -> None:
    config_path = _write_project(tmp_path, UNREACHABLE_NODE_GRAPH)
    result = CliRunner().invoke(cli, ["lint", "--config", str(config_path)])
    assert result.exit_code == 0, result.output
    assert "unreachable-node" in result.output
    assert "orphan" in result.output


def test_lint_unreachable_node_fails_with_fail_on_warning(
    tmp_path: pathlib.Path,
) -> None:
    config_path = _write_project(tmp_path, UNREACHABLE_NODE_GRAPH)
    result = CliRunner().invoke(
        cli, ["lint", "--config", str(config_path), "--fail-on", "warning"]
    )
    assert result.exit_code == 1, result.output


def test_lint_unbounded_cycle_fails_by_default(tmp_path: pathlib.Path) -> None:
    config_path = _write_project(tmp_path, UNBOUNDED_CYCLE_GRAPH)
    result = CliRunner().invoke(cli, ["lint", "--config", str(config_path)])
    assert result.exit_code == 1, result.output
    assert "unbounded-cycle" in result.output


def test_lint_unbounded_cycle_fail_on_never_exits_zero(tmp_path: pathlib.Path) -> None:
    config_path = _write_project(tmp_path, UNBOUNDED_CYCLE_GRAPH)
    result = CliRunner().invoke(
        cli, ["lint", "--config", str(config_path), "--fail-on", "never"]
    )
    assert result.exit_code == 0, result.output
    assert "unbounded-cycle" in result.output


def test_lint_factory_graph_is_skipped_not_invoked(tmp_path: pathlib.Path) -> None:
    config_path = _write_project(tmp_path, FACTORY_GRAPH, attr="make_graph")
    result = CliRunner().invoke(cli, ["lint", "--config", str(config_path)])
    assert result.exit_code == 0, result.output
    assert "skipped" in result.output
    assert "NotImplementedError" not in result.output


def test_lint_missing_attribute_reports_error(tmp_path: pathlib.Path) -> None:
    config_path = _write_project(tmp_path, CLEAN_GRAPH, attr="does_not_exist")
    result = CliRunner().invoke(cli, ["lint", "--config", str(config_path)])
    assert result.exit_code == 1, result.output
    assert "failed to load" in result.output


def test_lint_reports_actionable_error_when_langgraph_not_installed(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config_path = _write_project(tmp_path, CLEAN_GRAPH)
    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "langgraph.graph.analysis" or name.startswith(
            "langgraph.graph.analysis"
        ):
            raise ImportError("No module named 'langgraph'")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    result = CliRunner().invoke(cli, ["lint", "--config", str(config_path)])
    assert result.exit_code == 2, result.output
    assert "pip install -U langgraph" in result.output


def test_lint_no_graphs_configured_rejected_by_config_validation(
    tmp_path: pathlib.Path,
) -> None:
    # langgraph_cli.config.validate_config already requires at least one
    # graph; `lint` reuses that same validation rather than duplicating it.
    config_path = tmp_path / "langgraph.json"
    config_path.write_text(json.dumps({"dependencies": ["."], "graphs": {}}))
    result = CliRunner().invoke(cli, ["lint", "--config", str(config_path)])
    assert result.exit_code == 2, result.output
    assert "No graphs found in config" in result.output
