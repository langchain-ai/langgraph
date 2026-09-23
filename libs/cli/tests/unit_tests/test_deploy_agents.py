import json
from unittest.mock import Mock

import httpx
import pytest
from click.testing import CliRunner

import langgraph_cli.deploy as deploy
from langgraph_cli.cli import cli
from langgraph_cli.host_backend import HostBackendClient


@pytest.fixture
def deployment_api(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("LANGSMITH_DEPLOYMENT_NAME", raising=False)
    monkeypatch.setattr(deploy, "_emitter", None)
    monkeypatch.setattr(deploy, "_no_input", False)
    (tmp_path / "langgraph.json").write_text(
        json.dumps({"dependencies": ["."], "graphs": {"agent": "./agent.py:graph"}})
    )
    (tmp_path / ".env").write_text("LANGSMITH_DEPLOYMENT_NAME=legacy\n")
    requests = []
    state = {"enabled": True, "resources": []}

    def handler(request):
        requests.append(request)
        assert request.url.path == "/v2/deployments"
        if request.method == "GET":
            if not state["enabled"] and (
                "agent_id" in request.url.params
                or "agent_environment" in request.url.params
            ):
                return httpx.Response(
                    400, text="Agent filters are not available for this tenant."
                )
            return httpx.Response(200, json={"resources": state["resources"]})
        assert request.method == "POST"
        return httpx.Response(200, json={"id": "runtime-id", "name": "server-name"})

    client = HostBackendClient("https://api.example.com", "test-key")
    client._client.close()
    client._client = httpx.Client(
        base_url="https://api.example.com",
        transport=httpx.MockTransport(handler),
        headers={"X-Api-Key": "test-key"},
    )
    monkeypatch.setattr(deploy, "_create_host_backend_client", lambda *a, **kw: client)
    monkeypatch.setattr(deploy, "find_tracked_packages", lambda *a: [])
    remote_build = Mock(return_value=deploy.BuildResult())
    monkeypatch.setattr(deploy, "_run_remote_build", remote_build)
    monkeypatch.setattr(deploy, "_resolve_build_mode", lambda flag, **kw: (flag, None))
    yield state, requests, remote_build
    client._client.close()


AGENT_ARGS = [
    "deploy",
    "--agent-id",
    "customer-support",
    "--agent-environment",
    "staging",
    "--remote",
    "--no-wait",
    "--no-input",
]


def test_agent_create(deployment_api, tmp_path, monkeypatch):
    monkeypatch.setenv("LANGSMITH_DEPLOYMENT_NAME", "legacy")
    _, requests, build = deployment_api
    result = CliRunner().invoke(cli, AGENT_ARGS)
    assert result.exit_code == 0, result.output
    assert dict(requests[0].url.params) == {
        "agent_id": "customer-support",
        "agent_environment": "staging",
        "limit": "100",
    }
    payload = json.loads(requests[1].content)
    assert payload["agent"] == {
        "agent_id": "customer-support",
        "environment": "staging",
    }
    assert "name" not in payload
    assert build.call_args.kwargs["deployment_id"] == "runtime-id"
    assert "server-name" in result.output
    assert (tmp_path / ".env").read_text() == "LANGSMITH_DEPLOYMENT_NAME=legacy\n"


def test_agent_update(deployment_api):
    state, requests, build = deployment_api
    state["resources"] = [{"id": "existing-id", "is_preview": False}]
    result = CliRunner().invoke(cli, AGENT_ARGS)
    assert result.exit_code == 0, result.output
    assert len(requests) == 1
    assert build.call_args.kwargs["deployment_id"] == "existing-id"


def test_agent_rejects_explicit_name(deployment_api, monkeypatch):
    monkeypatch.setenv("LANGSMITH_DEPLOYMENT_NAME", "legacy")
    _, requests, _ = deployment_api
    result = CliRunner().invoke(cli, [*AGENT_ARGS, "--name", "legacy"])
    assert result.exit_code == 2
    assert "cannot be combined" in result.output
    assert not requests


def test_agent_lookup_refuses_a_control_plane_that_ignores_the_filter(deployment_api):
    state, requests, _ = deployment_api
    state["resources"] = [
        {"id": "someone-elses", "is_preview": False},
        {"id": "another", "is_preview": False},
    ]

    result = CliRunner().invoke(cli, AGENT_ARGS)

    assert result.exit_code != 0
    assert "does not filter deployments by agent" in result.output
    assert len(requests) == 1
