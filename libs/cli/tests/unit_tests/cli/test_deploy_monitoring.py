import json

import httpx
import pytest
from click.testing import CliRunner

import langgraph_cli.deploy as deploy_module
from langgraph_cli.cli import cli
from langgraph_cli.host_backend import HostBackendClient

DEPLOYMENT_ID = "019a0000-0000-7000-8000-000000000001"


def invoke(monkeypatch, args, handler, input=None):
    client = HostBackendClient(
        "https://api.example.com",
        "key",
        tenant_id="workspace-id",
        transport=httpx.MockTransport(handler),
    )
    monkeypatch.setattr(deploy_module, "HostBackendClient", lambda *a, **kw: client)
    return CliRunner().invoke(
        cli,
        ["deploy", *args, "--api-key", "key"],
        input=input,
    )


@pytest.mark.parametrize("filtered", [False, True])
def test_metrics(monkeypatch, filtered):
    charts = {"id": "metrics", "charts": [], "sub_sections": []}
    params = (
        {
            "start_time": "2026-10-28T00:00:00Z",
            "end_time": "2026-10-28T01:00:00Z",
            "metrics": "cpu_usage,memory_usage",
        }
        if filtered
        else {}
    )

    def handler(request):
        assert request.method == "GET"
        assert request.url.path == f"/v2/deployments/{DEPLOYMENT_ID}/metrics"
        assert dict(request.url.params) == params
        assert request.headers["x-tenant-id"] == "workspace-id"
        assert request.headers["x-api-key"] == "key"
        return httpx.Response(200, json=charts)

    args = ["metrics", DEPLOYMENT_ID]
    for key, value in params.items():
        args.extend([f"--{key.replace('_', '-')}", value])
    result = invoke(monkeypatch, args, handler)
    assert result.exit_code == 0, result.output
    assert json.loads(result.output) == charts


@pytest.mark.parametrize(
    "tiers",
    [
        {"compute_tier": "SERVERLESS_M"},
        {"database_tier": "DEDICATED_L"},
        {"compute_tier": "DEDICATED_M", "database_tier": "DEDICATED_L"},
    ],
)
def test_scale(monkeypatch, tiers):
    def handler(request):
        assert request.method == "PATCH"
        assert request.url.path == f"/v2/deployments/{DEPLOYMENT_ID}/resource-tiers"
        assert request.headers["x-tenant-id"] == "workspace-id"
        assert json.loads(request.content) == tiers
        return httpx.Response(200, json={"secrets": [{"value": "secret-value"}]})

    args = ["scale", DEPLOYMENT_ID, "--yes"]
    for key, value in tiers.items():
        args.extend([f"--{key.replace('_', '-')}", value])
    result = invoke(monkeypatch, args, handler)
    assert result.exit_code == 0, result.output
    assert "Scaling requested" in result.output
    assert "secret-value" not in result.output


@pytest.mark.parametrize("answer,expected_code", [("y\n", 0), ("n\n", 1)])
def test_scale_confirmation(monkeypatch, answer, expected_code):
    requests = []

    def handler(request):
        requests.append(request)
        return httpx.Response(200, json={})

    result = invoke(
        monkeypatch,
        ["scale", DEPLOYMENT_ID, "--compute-tier", "DEDICATED_S"],
        handler,
        input=answer,
    )
    assert result.exit_code == expected_code, result.output
    assert len(requests) == (1 if expected_code == 0 else 0)
    assert "may change billing or interrupt service" in result.output


@pytest.mark.parametrize(
    "args",
    [
        ["scale", DEPLOYMENT_ID, "--yes"],
        ["scale", DEPLOYMENT_ID, "--database-tier", "SERVERLESS_M", "--yes"],
        ["metrics", DEPLOYMENT_ID, "--step-seconds", "60"],
        ["metrics", "invalid-id"],
    ],
)
def test_invalid_options_do_not_send_requests(monkeypatch, args):
    def handler(request):
        pytest.fail("Invalid options must not send a request")

    result = invoke(monkeypatch, args, handler)
    assert result.exit_code == 2, result.output


@pytest.mark.parametrize("command", ["metrics", "scale"])
@pytest.mark.parametrize("status", [403, 422, 429])
def test_backend_errors(monkeypatch, command, status):
    def handler(request):
        return httpx.Response(status, json={"detail": "Operation rejected"})

    args = [command, DEPLOYMENT_ID]
    if command == "scale":
        args.extend(["--compute-tier", "DEDICATED_M", "--yes"])
    result = invoke(monkeypatch, args, handler)
    assert result.exit_code == 1
    assert "Operation rejected" in result.output
