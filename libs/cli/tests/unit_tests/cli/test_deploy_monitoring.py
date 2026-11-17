import json
from datetime import datetime, timezone

import httpx
import pytest
from click.testing import CliRunner

import langgraph_cli.deploy as deploy_module
from langgraph_cli.cli import cli
from langgraph_cli.deployment_metrics import compact_deployment_metrics
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


@pytest.mark.parametrize(
    "bounds", [[], ["start_time"], ["end_time"], ["start_time", "end_time"]]
)
@pytest.mark.parametrize("mode", [None, "--summary", "--raw"])
def test_metrics(monkeypatch, bounds, mode, deployment_metrics_response):
    charts = deployment_metrics_response

    class FrozenDatetime(datetime):
        @classmethod
        def now(cls, tz=None):
            return cls(2026, 10, 28, 23, 2, 33, tzinfo=timezone.utc)

    monkeypatch.setattr(deploy_module, "datetime", FrozenDatetime)
    explicit = {
        "start_time": "2026-10-27T22:32:33Z",
        "end_time": "2026-10-28T21:32:33Z",
    }
    options = {key: explicit[key] for key in bounds}
    if bounds:
        options["metrics"] = "cpu_usage,memory_usage"
    params = {
        "start_time": (
            "2026-10-28T21:17:33+00:00"
            if "end_time" in bounds
            else "2026-10-28T22:47:33+00:00"
        ),
        "end_time": "2026-10-28T23:02:33+00:00",
        **options,
    }

    def handler(request):
        assert request.method == "GET"
        assert request.url.path == f"/v2/deployments/{DEPLOYMENT_ID}/metrics"
        assert dict(request.url.params) == params
        assert request.headers["x-tenant-id"] == "workspace-id"
        assert request.headers["x-api-key"] == "key"
        return httpx.Response(200, json=charts)

    args = ["metrics", DEPLOYMENT_ID]
    for key, value in options.items():
        args.extend([f"--{key.replace('_', '-')}", value])
    if mode:
        args.append(mode)
    result = invoke(monkeypatch, args, handler)
    assert result.exit_code == 0, result.output
    expected = (
        charts
        if mode == "--raw"
        else compact_deployment_metrics(
            charts,
            DEPLOYMENT_ID,
            start_time=params.get("start_time"),
            end_time=params.get("end_time"),
            summary=mode == "--summary",
        )
    )
    assert json.loads(result.output) == expected
    if mode != "--raw":
        assert expected["start_time"] == params["start_time"]
        assert expected["end_time"] == params["end_time"]
    assert result.output.strip() == json.dumps(expected, separators=(",", ":"))


@pytest.mark.parametrize(
    "args",
    [
        ["metrics", DEPLOYMENT_ID, "--step-seconds", "60"],
        ["metrics", "invalid-id"],
        ["metrics", DEPLOYMENT_ID, "--summary", "--raw"],
    ],
)
def test_invalid_options_do_not_send_requests(monkeypatch, args):
    def handler(request):
        pytest.fail("Invalid options must not send a request")

    result = invoke(monkeypatch, args, handler)
    assert result.exit_code == 2, result.output


def test_end_only_with_timezone_offset(monkeypatch):
    def handler(request):
        assert request.url.params["end_time"] == "2026-10-28T01:05:00+02:00"
        assert request.url.params["start_time"] == "2026-10-27T22:50:00+00:00"
        return httpx.Response(200, json={"charts": [], "sub_sections": []})

    result = invoke(
        monkeypatch,
        ["metrics", DEPLOYMENT_ID, "--end-time", "2026-10-28T01:05:00+02:00"],
        handler,
    )
    assert result.exit_code == 0, result.output
    output = json.loads(result.output)
    assert output["start_time"] == "2026-10-27T22:50:00+00:00"
    assert output["end_time"] == "2026-10-28T01:05:00+02:00"


def test_invalid_end_only_does_not_send_request(monkeypatch):
    def handler(request):
        pytest.fail("Invalid end time must not send a request")

    result = invoke(
        monkeypatch, ["metrics", DEPLOYMENT_ID, "--end-time", "not-a-date"], handler
    )
    assert result.exit_code == 2, result.output
    assert "--end-time" in result.output
    assert "ISO8601" in result.output


def test_metrics_help_lists_supported_filters():
    result = CliRunner().invoke(cli, ["deploy", "metrics", "--help"])
    assert result.exit_code == 0, result.output
    for metric in (
        "memory_usage",
        "cpu_usage",
        "disk_usage",
        "restart_count",
        "replica_count",
        "worker_count",
        "lg_run_count",
        "responses_per_second",
        "error_responses_per_second",
        "p95_latency",
        "run_queue_wait_time",
    ):
        assert metric in result.output


@pytest.mark.parametrize("status", [403, 422, 429])
def test_backend_errors(monkeypatch, status):
    def handler(request):
        return httpx.Response(status, json={"detail": "Operation rejected"})

    args = ["metrics", DEPLOYMENT_ID]
    result = invoke(monkeypatch, args, handler)
    assert result.exit_code == 1
    assert "Operation rejected" in result.output
