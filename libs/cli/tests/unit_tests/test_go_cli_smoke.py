import os
import shutil
import subprocess

import pytest


@pytest.fixture
def go_cli_env():
    if os.environ.get("LANGGRAPH_USE_GO_CLI") != "1":
        pytest.skip("Go CLI smoke tests only run when LANGGRAPH_USE_GO_CLI=1")

    langgraph = shutil.which("langgraph")
    assert langgraph is not None, "langgraph executable is not installed"

    env = os.environ.copy()
    env["LANGGRAPH_USE_GO_CLI"] = "1"
    return langgraph, env


def test_go_cli_help(go_cli_env):
    langgraph, env = go_cli_env

    result = subprocess.run(
        [langgraph, "--help"],
        capture_output=True,
        text=True,
        env=env,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert "Usage: langgraph" in result.stdout


def test_go_cli_validate(go_cli_env, tmp_path):
    langgraph, env = go_cli_env
    config_path = tmp_path / "langgraph.json"
    config_path.write_text(
        '{"dependencies": ["langchain"], "graphs": {"agent": "./agent.py:graph"}}'
    )

    result = subprocess.run(
        [langgraph, "validate", "-c", str(config_path)],
        capture_output=True,
        text=True,
        env=env,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert "is valid" in result.stdout


def test_go_cli_dev_help(go_cli_env):
    langgraph, env = go_cli_env

    result = subprocess.run(
        [langgraph, "dev", "--help"],
        capture_output=True,
        text=True,
        env=env,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert "Run LangGraph API server in development mode" in result.stdout
