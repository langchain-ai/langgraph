import json
import pathlib

import click
import pytest

from langgraph_cli.engine_runtime_mode import resolve_engine_runtime_mode


def _write_config(
    tmp_path: pathlib.Path,
    *,
    python_version: str | None = "3.11",
    node_version: str | None = None,
) -> pathlib.Path:
    cfg: dict = {"dependencies": ["."], "graphs": {"agent": "agent.py:graph"}}
    if python_version is not None:
        cfg["python_version"] = python_version
    if node_version is not None:
        cfg["node_version"] = node_version
    path = tmp_path / "langgraph.json"
    path.write_text(json.dumps(cfg))
    return path


class TestResolveEngineRuntimeMode:
    # -- cli_param == "distributed" -------------------------------------------

    def test_distributed_explicit_new_version(self, tmp_path: pathlib.Path) -> None:
        path = _write_config(tmp_path)
        assert (
            resolve_engine_runtime_mode(path, "0.7.68", "distributed") == "distributed"
        )

    def test_distributed_explicit_old_version_raises(
        self, tmp_path: pathlib.Path
    ) -> None:
        path = _write_config(tmp_path)
        with pytest.raises(click.ClickException, match="0.7.67"):
            resolve_engine_runtime_mode(path, "0.7.67", "distributed")

    def test_distributed_explicit_js_raises(self, tmp_path: pathlib.Path) -> None:
        path = _write_config(tmp_path, python_version=None, node_version="20")
        with pytest.raises(click.ClickException, match="JavaScript"):
            resolve_engine_runtime_mode(path, "0.8.0", "distributed")

    def test_distributed_explicit_js_and_old_version_raises(
        self, tmp_path: pathlib.Path
    ) -> None:
        path = _write_config(tmp_path, python_version=None, node_version="20")
        with pytest.raises(click.ClickException, match="JavaScript.*0.7.60"):
            resolve_engine_runtime_mode(path, "0.7.60", "distributed")

    # -- cli_param == "combined_queue_worker" ----------------------------------

    def test_combined_explicit(self, tmp_path: pathlib.Path) -> None:
        path = _write_config(tmp_path)
        assert (
            resolve_engine_runtime_mode(path, "0.8.0", "combined_queue_worker")
            == "combined_queue_worker"
        )

    def test_combined_explicit_old_version(self, tmp_path: pathlib.Path) -> None:
        path = _write_config(tmp_path)
        assert (
            resolve_engine_runtime_mode(path, "0.7.67", "combined_queue_worker")
            == "combined_queue_worker"
        )

    # -- cli_param is None (default) -------------------------------------------

    def test_default_is_distributed(self, tmp_path: pathlib.Path) -> None:
        path = _write_config(tmp_path)
        assert resolve_engine_runtime_mode(path, "0.8.0", None) == "distributed"

    def test_default_old_version(self, tmp_path: pathlib.Path) -> None:
        path = _write_config(tmp_path)
        assert resolve_engine_runtime_mode(path, "0.7.67", None) == "distributed"

    def test_default_js(self, tmp_path: pathlib.Path) -> None:
        path = _write_config(tmp_path, python_version=None, node_version="20")
        assert resolve_engine_runtime_mode(path, "0.8.0", None) == "distributed"

    # -- edge: version boundary ------------------------------------------------

    def test_version_boundary_0_7_67_blocks_distributed(
        self, tmp_path: pathlib.Path
    ) -> None:
        path = _write_config(tmp_path)
        with pytest.raises(click.ClickException):
            resolve_engine_runtime_mode(path, "0.7.67", "distributed")

    def test_version_boundary_0_7_68_allows_distributed(
        self, tmp_path: pathlib.Path
    ) -> None:
        path = _write_config(tmp_path)
        assert (
            resolve_engine_runtime_mode(path, "0.7.68", "distributed") == "distributed"
        )
