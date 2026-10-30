import json
from datetime import datetime, timedelta, timezone

import click
import httpx
import pytest

from langgraph_cli import deploy as deploy_mod
from langgraph_cli import langsmith_auth
from langgraph_cli.deploy import _create_host_backend_client
from langgraph_cli.host_backend import HostBackendClient

API_URL = "https://api.smith.langchain.com"


def _iso(delta: timedelta) -> str:
    return (datetime.now(timezone.utc) + delta).strftime("%Y-%m-%dT%H:%M:%SZ")


@pytest.fixture
def config_file(tmp_path, monkeypatch):
    path = tmp_path / "config.json"
    monkeypatch.setenv("LANGSMITH_CONFIG_FILE", str(path))
    monkeypatch.delenv("LANGSMITH_PROFILE", raising=False)
    for name in ("LANGGRAPH_HOST_API_KEY", "LANGSMITH_API_KEY", "LANGCHAIN_API_KEY"):
        monkeypatch.delenv(name, raising=False)
    return path


def _write_profile(path, **oauth):
    path.write_text(
        json.dumps(
            {
                "current_profile": "default",
                "profiles": {
                    "default": {
                        "api_url": API_URL,
                        "workspace_id": "ws-1",
                        "oauth": oauth,
                    }
                },
            }
        )
    )


def _no_discovery(monkeypatch):
    def get(*_a, **_k):
        raise httpx.ConnectError("no discovery")

    monkeypatch.setattr(httpx, "get", get)


def test_valid_token_is_returned_without_network(config_file, monkeypatch):
    _write_profile(
        config_file, access_token="tok", expires_at=_iso(timedelta(minutes=5))
    )

    def boom(*_a, **_k):
        raise AssertionError("no network expected")

    monkeypatch.setattr(httpx, "get", boom)
    monkeypatch.setattr(httpx, "post", boom)

    assert langsmith_auth.get_access_token(API_URL) == "tok"


def test_expired_token_is_refreshed_and_saved(config_file, monkeypatch):
    _write_profile(
        config_file,
        access_token="old",
        refresh_token="refresh-1",
        expires_at=_iso(timedelta(seconds=-5)),
    )
    _no_discovery(monkeypatch)
    seen = {}

    def post(url, data, timeout):
        seen.update(url=url, data=data)
        return httpx.Response(
            200,
            json={
                "access_token": "new",
                "refresh_token": "refresh-2",
                "expires_in": 300,
            },
            request=httpx.Request("POST", url),
        )

    monkeypatch.setattr(httpx, "post", post)

    assert langsmith_auth.get_access_token(API_URL) == "new"
    assert seen["url"] == f"{API_URL}/oauth/token"
    assert seen["data"]["refresh_token"] == "refresh-1"
    saved = json.loads(config_file.read_text())["profiles"]["default"]["oauth"]
    assert saved["access_token"] == "new"
    assert saved["refresh_token"] == "refresh-2"


def test_expired_token_without_refresh_token_asks_to_log_in(config_file):
    _write_profile(
        config_file, access_token="old", expires_at=_iso(timedelta(seconds=-5))
    )
    with pytest.raises(click.ClickException, match="langgraph login"):
        langsmith_auth.get_access_token(API_URL)


def test_login_stores_tokens_after_pending(config_file, monkeypatch):
    _no_discovery(monkeypatch)
    monkeypatch.setattr(langsmith_auth.time, "sleep", lambda _s: None)
    monkeypatch.setattr(langsmith_auth.webbrowser, "open", lambda _u: True)
    token_responses = iter(
        [
            {"error": "authorization_pending"},
            {"access_token": "tok", "refresh_token": "r", "expires_in": 300},
        ]
    )

    def post(url, data, timeout):
        request = httpx.Request("POST", url)
        if url.endswith("/oauth/device/code"):
            return httpx.Response(
                200,
                json={
                    "device_code": "dc",
                    "user_code": "ABCD",
                    "verification_uri": "https://example.com/activate",
                    "expires_in": 600,
                    "interval": 5,
                },
                request=request,
            )
        body = next(token_responses)
        return httpx.Response(
            400 if "error" in body else 200, json=body, request=request
        )

    monkeypatch.setattr(httpx, "post", post)

    langsmith_auth.login(API_URL)

    profile = json.loads(config_file.read_text())["profiles"]["default"]
    assert profile["api_url"] == API_URL
    assert profile["oauth"]["access_token"] == "tok"


def test_client_sends_bearer_token_per_request():
    seen = []

    def handler(req: httpx.Request) -> httpx.Response:
        seen.append(dict(req.headers))
        return httpx.Response(200, json={"resources": []})

    tokens = iter(["t1", "t2"])
    client = HostBackendClient(
        "https://api.example.com",
        None,
        access_token=lambda: next(tokens),
        transport=httpx.MockTransport(handler),
    )
    client.list_deployments()
    client.list_deployments()

    assert [h["authorization"] for h in seen] == ["Bearer t1", "Bearer t2"]
    assert "x-api-key" not in seen[0]


def test_factory_uses_logged_in_profile(config_file, monkeypatch):
    _write_profile(
        config_file, access_token="tok", expires_at=_iso(timedelta(minutes=5))
    )
    monkeypatch.setattr(deploy_mod, "_no_input", True)
    monkeypatch.delenv("LANGSMITH_TENANT_ID", raising=False)
    monkeypatch.delenv("LANGSMITH_ENDPOINT", raising=False)

    client = _create_host_backend_client(
        host_url="https://api.example.com", api_key=None, env_vars={}
    )

    assert client._access_token is not None
    assert client._access_token() == "tok"
    assert client._client.headers["x-tenant-id"] == "ws-1"
    assert "x-api-key" not in client._client.headers


def test_factory_without_credentials_and_no_input_fails(config_file, monkeypatch):
    monkeypatch.setattr(deploy_mod, "_no_input", True)
    with pytest.raises(click.ClickException, match="langgraph login"):
        _create_host_backend_client(
            host_url="https://api.example.com", api_key=None, env_vars={}
        )
