import pathlib
import sys

import pytest

from langgraph_cli import entrypoint


def test_main_uses_legacy_cli_when_go_flag_disabled(monkeypatch):
    captured = {}

    def fake_legacy(argv):
        captured["argv"] = list(argv)

    monkeypatch.delenv("LANGGRAPH_USE_GO_CLI", raising=False)
    monkeypatch.setattr(entrypoint, "_legacy_cli", fake_legacy)

    entrypoint.main(["build", "-t", "demo"])

    assert captured == {"argv": ["build", "-t", "demo"]}


def test_main_execs_go_cli_when_flag_enabled(monkeypatch, tmp_path):
    binary_path = tmp_path / "langgraph"
    binary_path.write_text("")

    captured = {}

    def fake_execvpe(file, args, env):
        captured["file"] = file
        captured["args"] = args
        captured["env"] = env.copy()
        raise SystemExit(0)

    monkeypatch.setenv("LANGGRAPH_USE_GO_CLI", "1")
    monkeypatch.setenv("LANGGRAPH_GO_CLI_PATH", str(binary_path))
    monkeypatch.delenv("LANGGRAPH_CALLING_PYTHON", raising=False)
    monkeypatch.setattr(entrypoint.os, "execvpe", fake_execvpe)

    with pytest.raises(SystemExit, match="0"):
        entrypoint.main(["dev", "--port", "8000"])

    assert captured["file"] == str(binary_path)
    assert captured["args"] == [str(binary_path), "dev", "--port", "8000"]
    assert captured["env"]["LANGGRAPH_CALLING_PYTHON"] == sys.executable


def test_main_preserves_existing_calling_python(monkeypatch, tmp_path):
    binary_path = tmp_path / "langgraph"
    binary_path.write_text("")

    captured = {}

    def fake_execvpe(file, args, env):
        captured["env"] = env.copy()
        raise SystemExit(0)

    monkeypatch.setenv("LANGGRAPH_USE_GO_CLI", "true")
    monkeypatch.setenv("LANGGRAPH_GO_CLI_PATH", str(binary_path))
    monkeypatch.setenv("LANGGRAPH_CALLING_PYTHON", "/custom/python")
    monkeypatch.setattr(entrypoint.os, "execvpe", fake_execvpe)

    with pytest.raises(SystemExit, match="0"):
        entrypoint.main(["dev"])

    assert captured["env"]["LANGGRAPH_CALLING_PYTHON"] == "/custom/python"


def test_main_errors_when_go_cli_requested_but_binary_missing(
    monkeypatch, capsys, tmp_path
):
    missing_path = tmp_path / "missing-langgraph"

    monkeypatch.setenv("LANGGRAPH_USE_GO_CLI", "1")
    monkeypatch.setenv("LANGGRAPH_GO_CLI_PATH", str(missing_path))

    with pytest.raises(SystemExit, match="1"):
        entrypoint.main(["build"])

    err = capsys.readouterr().err
    assert "LANGGRAPH_GO_CLI_PATH points to a missing file" in err


def test_resolve_go_cli_path_prefers_override(monkeypatch, tmp_path):
    override = tmp_path / "custom-langgraph"
    override.write_text("")
    bundled = tmp_path / "bin" / "langgraph"
    bundled.parent.mkdir()
    bundled.write_text("")

    monkeypatch.setenv("LANGGRAPH_GO_CLI_PATH", str(override))
    monkeypatch.setattr(entrypoint, "_bundled_go_cli_path", lambda: bundled)

    assert entrypoint._resolve_go_cli_path() == override.resolve()


def test_resolve_go_cli_path_uses_bundled_binary(monkeypatch, tmp_path):
    bundled = tmp_path / "bin" / "langgraph"
    bundled.parent.mkdir()
    bundled.write_text("")

    monkeypatch.delenv("LANGGRAPH_GO_CLI_PATH", raising=False)
    monkeypatch.setattr(entrypoint, "_bundled_go_cli_path", lambda: bundled)

    assert entrypoint._resolve_go_cli_path() == bundled


def test_resolve_go_cli_path_returns_none_when_nothing_available(monkeypatch):
    monkeypatch.delenv("LANGGRAPH_GO_CLI_PATH", raising=False)
    monkeypatch.setattr(
        entrypoint,
        "_bundled_go_cli_path",
        lambda: pathlib.Path("/definitely/not/present/langgraph"),
    )

    assert entrypoint._resolve_go_cli_path() is None
