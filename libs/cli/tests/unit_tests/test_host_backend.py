import httpx
import pytest

from langgraph_cli.host_backend import HostBackendClient, HostBackendError


@pytest.fixture
def mock_transport():
    return httpx.MockTransport(lambda req: httpx.Response(200, json={"ok": True}))


@pytest.fixture
def client(mock_transport):
    c = HostBackendClient("https://api.example.com", "test-key")
    c._client = httpx.Client(
        base_url="https://api.example.com",
        transport=mock_transport,
        headers={"X-Api-Key": "test-key", "Accept": "application/json"},
        timeout=30,
    )
    return c


def test_constructor_strips_trailing_slash():
    c = HostBackendClient("https://api.example.com/", "key")
    assert str(c._client.base_url) == "https://api.example.com"


def test_constructor_empty_url_raises():
    with pytest.raises(Exception, match="Host backend URL is required"):
        HostBackendClient("", "key")


def test_request_sends_headers():
    def handler(req: httpx.Request) -> httpx.Response:
        assert req.headers["x-api-key"] == "test-key"
        assert req.headers["accept"] == "application/json"
        return httpx.Response(200, json={"ok": True})

    c = HostBackendClient("https://api.example.com", "test-key")
    c._client = httpx.Client(
        base_url="https://api.example.com",
        transport=httpx.MockTransport(handler),
        headers={"X-Api-Key": "test-key", "Accept": "application/json"},
        timeout=30,
    )
    result = c._request("GET", "/test")
    assert result == {"ok": True}


def test_request_sends_json_payload():
    def handler(req: httpx.Request) -> httpx.Response:
        assert req.headers["content-type"] == "application/json"
        assert req.content == b'{"key":"value"}'
        return httpx.Response(200, json={"created": True})

    c = HostBackendClient("https://api.example.com", "test-key")
    c._client = httpx.Client(
        base_url="https://api.example.com",
        transport=httpx.MockTransport(handler),
        headers={"X-Api-Key": "test-key", "Accept": "application/json"},
        timeout=30,
    )
    result = c._request("POST", "/test", {"key": "value"})
    assert result == {"created": True}


def test_request_empty_body_returns_none():
    transport = httpx.MockTransport(lambda req: httpx.Response(200, content=b""))
    c = HostBackendClient("https://api.example.com", "test-key")
    c._client = httpx.Client(
        base_url="https://api.example.com",
        transport=transport,
        headers={"X-Api-Key": "test-key", "Accept": "application/json"},
        timeout=30,
    )
    assert c._request("DELETE", "/test") is None


def test_request_http_error_raises():
    transport = httpx.MockTransport(lambda req: httpx.Response(404, text="not found"))
    c = HostBackendClient("https://api.example.com", "test-key")
    c._client = httpx.Client(
        base_url="https://api.example.com",
        transport=transport,
        headers={"X-Api-Key": "test-key", "Accept": "application/json"},
        timeout=30,
    )
    with pytest.raises(HostBackendError, match="404"):
        c._request("GET", "/missing")


def test_request_invalid_json_raises():
    transport = httpx.MockTransport(
        lambda req: httpx.Response(200, content=b"not json")
    )
    c = HostBackendClient("https://api.example.com", "test-key")
    c._client = httpx.Client(
        base_url="https://api.example.com",
        transport=transport,
        headers={"X-Api-Key": "test-key", "Accept": "application/json"},
        timeout=30,
    )
    with pytest.raises(HostBackendError, match="Failed to decode"):
        c._request("GET", "/bad-json")


def test_request_transport_error_raises():
    def handler(req: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused")

    c = HostBackendClient("https://api.example.com", "test-key")
    c._client = httpx.Client(
        base_url="https://api.example.com",
        transport=httpx.MockTransport(handler),
        headers={"X-Api-Key": "test-key", "Accept": "application/json"},
        timeout=30,
    )
    with pytest.raises(HostBackendError, match="connection refused"):
        c._request("GET", "/test")


def test_create_deployment(client):
    result = client.create_deployment({"name": "my-deploy"})
    assert result == {"ok": True}


def test_get_deployment(client):
    result = client.get_deployment("dep-123")
    assert result == {"ok": True}


def test_list_deployments(client):
    result = client.list_deployments("my-app")
    assert result == {"ok": True}


def test_list_deployments_sends_query_params():
    def handler(req: httpx.Request) -> httpx.Response:
        assert req.url.path == "/v2/deployments"
        assert req.url.params["name_contains"] == "my app"
        return httpx.Response(200, json={"ok": True})

    c = HostBackendClient("https://api.example.com", "test-key")
    c._client = httpx.Client(
        base_url="https://api.example.com",
        transport=httpx.MockTransport(handler),
        headers={"X-Api-Key": "test-key", "Accept": "application/json"},
        timeout=30,
    )
    result = c.list_deployments("my app")
    assert result == {"ok": True}


def test_delete_deployment(client):
    result = client.delete_deployment("dep-123")
    assert result == {"ok": True}


def test_request_push_token(client):
    result = client.request_push_token("dep-123")
    assert result == {"ok": True}


def test_update_deployment(client):
    result = client.update_deployment(
        "dep-123", "image:latest", secrets=[{"name": "KEY", "value": "val"}]
    )
    assert result == {"ok": True}


def test_update_deployment_no_secrets(client):
    result = client.update_deployment("dep-123", "image:latest")
    assert result == {"ok": True}


def test_list_revisions(client):
    result = client.list_revisions("dep-123", limit=5)
    assert result == {"ok": True}


def test_get_revision(client):
    result = client.get_revision("dep-123", "rev-456")
    assert result == {"ok": True}


def test_get_build_logs(client):
    result = client.get_build_logs("proj-1", "rev-1", {"limit": 10})
    assert result == {"ok": True}


def test_get_deploy_logs_all_revisions():
    def handler(req: httpx.Request) -> httpx.Response:
        assert "/v1/projects/proj-1/deploy_logs" in str(req.url)
        assert "/revisions/" not in str(req.url)
        return httpx.Response(200, json={"logs": [{"message": "running"}]})

    c = HostBackendClient("https://api.example.com", "key")
    c._client = httpx.Client(
        base_url="https://api.example.com",
        transport=httpx.MockTransport(handler),
        headers={"X-Api-Key": "key", "Accept": "application/json"},
        timeout=30,
    )
    result = c.get_deploy_logs("proj-1", {"limit": 10})
    assert result == {"logs": [{"message": "running"}]}


def test_get_deploy_logs_specific_revision():
    def handler(req: httpx.Request) -> httpx.Response:
        assert "/v1/projects/proj-1/revisions/rev-2/deploy_logs" in str(req.url)
        return httpx.Response(200, json={"logs": []})

    c = HostBackendClient("https://api.example.com", "key")
    c._client = httpx.Client(
        base_url="https://api.example.com",
        transport=httpx.MockTransport(handler),
        headers={"X-Api-Key": "key", "Accept": "application/json"},
        timeout=30,
    )
    result = c.get_deploy_logs("proj-1", {"limit": 10}, revision_id="rev-2")
    assert result == {"logs": []}
