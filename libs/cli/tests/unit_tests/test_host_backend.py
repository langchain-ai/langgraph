import json

import httpx
import pytest

from langgraph_cli.host_backend import (
    ControlPlaneEndpoints,
    HostBackendClient,
    HostBackendError,
)


def test_constructor_strips_trailing_slash():
    c = HostBackendClient("https://api.example.com/", "key")
    assert c.base_url == "https://api.example.com"


def test_constructor_empty_url_raises():
    with pytest.raises(Exception, match="Host backend URL is required"):
        HostBackendClient("", "key")


def test_request_sends_headers():
    def handler(req: httpx.Request) -> httpx.Response:
        assert req.headers["x-api-key"] == "test-key"
        assert req.headers["accept"] == "application/json"
        return httpx.Response(200, json={"ok": True})

    c = HostBackendClient(
        "https://api.example.com", "test-key", transport=httpx.MockTransport(handler)
    )
    result = c._request("GET", "/test")
    assert result == {"ok": True}


def test_request_sends_json_payload():
    def handler(req: httpx.Request) -> httpx.Response:
        assert req.headers["content-type"] == "application/json"
        assert req.content == b'{"key":"value"}'
        return httpx.Response(200, json={"created": True})

    c = HostBackendClient(
        "https://api.example.com", "test-key", transport=httpx.MockTransport(handler)
    )
    result = c._request("POST", "/test", {"key": "value"})
    assert result == {"created": True}


def test_request_empty_body_returns_none():
    transport = httpx.MockTransport(lambda req: httpx.Response(200, content=b""))
    c = HostBackendClient("https://api.example.com", "test-key", transport=transport)
    assert c._request("DELETE", "/test") is None


def test_request_http_error_raises():
    transport = httpx.MockTransport(lambda req: httpx.Response(404, text="not found"))
    c = HostBackendClient("https://api.example.com", "test-key", transport=transport)
    with pytest.raises(HostBackendError, match="404"):
        c._request("GET", "/missing")


def test_request_invalid_json_raises():
    transport = httpx.MockTransport(
        lambda req: httpx.Response(200, content=b"not json")
    )
    c = HostBackendClient("https://api.example.com", "test-key", transport=transport)
    with pytest.raises(HostBackendError, match="Failed to decode"):
        c._request("GET", "/bad-json")


def test_request_transport_error_raises():
    def handler(req: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused")

    c = HostBackendClient(
        "https://api.example.com", "test-key", transport=httpx.MockTransport(handler)
    )
    with pytest.raises(HostBackendError, match="connection refused"):
        c._request("GET", "/test")


def _capturing_client(captured: dict) -> HostBackendClient:
    def handler(req: httpx.Request) -> httpx.Response:
        captured["body"] = req.read()
        return httpx.Response(200, json={"ok": True})

    c = HostBackendClient(
        "https://api.example.com", "key", transport=httpx.MockTransport(handler)
    )
    return c


def test_update_deployment_forwards_tracked_packages():
    captured: dict = {}
    c = _capturing_client(captured)
    c.update_deployment(
        "dep-123",
        "image:latest",
        revision_source="internal_docker",
        tracked_packages=["google-adk:1.0.0"],
    )
    body = json.loads(captured["body"])
    assert body["tracked_packages"] == ["google-adk:1.0.0"]
    assert "tracked_packages" not in body["source_revision_config"]


def test_update_deployment_omits_tracked_packages_when_absent():
    captured: dict = {}
    c = _capturing_client(captured)
    c.update_deployment("dep-123", "image:latest", revision_source="internal_docker")
    body = json.loads(captured["body"])
    assert "tracked_packages" not in body


def test_update_deployment_internal_source_forwards_tracked_packages():
    captured: dict = {}
    c = _capturing_client(captured)
    c.update_deployment_internal_source(
        "dep-123",
        source_tarball_path="path/to/tarball",
        config_path="langgraph.json",
        tracked_packages=["google-adk:>=0.5"],
    )
    body = json.loads(captured["body"])
    assert body["tracked_packages"] == ["google-adk:>=0.5"]
    assert body["source_revision_config"]["source_tarball_path"] == "path/to/tarball"
    assert "tracked_packages" not in body["source_revision_config"]


def test_update_deployment_internal_source_omits_tracked_packages_when_absent():
    captured: dict = {}
    c = _capturing_client(captured)
    c.update_deployment_internal_source(
        "dep-123",
        source_tarball_path="path/to/tarball",
        config_path="langgraph.json",
    )
    body = json.loads(captured["body"])
    assert "tracked_packages" not in body


def test_get_deploy_logs_all_revisions():
    def handler(req: httpx.Request) -> httpx.Response:
        assert "/v1/projects/proj-1/deploy_logs" in str(req.url)
        assert "/revisions/" not in str(req.url)
        return httpx.Response(200, json={"logs": [{"message": "running"}]})

    c = HostBackendClient(
        "https://api.example.com", "key", transport=httpx.MockTransport(handler)
    )
    result = c.get_deploy_logs("proj-1", {"limit": 10})
    assert result == {"logs": [{"message": "running"}]}


def test_get_deploy_logs_specific_revision():
    def handler(req: httpx.Request) -> httpx.Response:
        assert "/v1/projects/proj-1/revisions/rev-2/deploy_logs" in str(req.url)
        return httpx.Response(200, json={"logs": []})

    c = HostBackendClient(
        "https://api.example.com", "key", transport=httpx.MockTransport(handler)
    )
    result = c.get_deploy_logs("proj-1", {"limit": 10}, revision_id="rev-2")
    assert result == {"logs": []}


def _routing_client(seen: dict) -> HostBackendClient:
    def handler(req: httpx.Request) -> httpx.Response:
        seen["method"] = req.method
        seen["url"] = str(req.url)
        return httpx.Response(200, json={"ok": True})

    c = HostBackendClient(
        "https://api.example.com/prefix", "key", transport=httpx.MockTransport(handler)
    )
    return c


@pytest.mark.parametrize(
    ("call", "expected_body"),
    [
        pytest.param(
            lambda c: c.create_deployment(
                name="my-deploy",
                source="internal_docker",
                source_config={"deployment_type": "dev"},
                source_revision_config={},
            ),
            {
                "name": "my-deploy",
                "source": "internal_docker",
                "source_config": {"deployment_type": "dev"},
                "source_revision_config": {},
            },
            id="internal_docker_create_omits_secrets_key_when_not_given",
        ),
        pytest.param(
            lambda c: c.create_deployment(
                name="my-deploy",
                source="internal_docker",
                source_config={"deployment_type": "prod"},
                source_revision_config={},
                secrets=[{"name": "KEY", "value": "val"}],
            ),
            {
                "name": "my-deploy",
                "source": "internal_docker",
                "source_config": {"deployment_type": "prod"},
                "source_revision_config": {},
                "secrets": [{"name": "KEY", "value": "val"}],
            },
            id="internal_docker_create_forwards_secrets",
        ),
        pytest.param(
            lambda c: c.update_deployment(
                "dep-123",
                "registry.example.com/app@sha256:abc",
                revision_source="internal_docker",
                secrets=[{"name": "KEY", "value": "val"}],
            ),
            {
                "revision_source": "internal_docker",
                "source_revision_config": {
                    "image_uri": "registry.example.com/app@sha256:abc"
                },
                "secrets": [{"name": "KEY", "value": "val"}],
            },
            id="internal_docker_revision_names_its_source",
        ),
        pytest.param(
            lambda c: c.update_deployment_internal_source(
                "dep-123",
                source_tarball_path="tarballs/src.tgz",
                config_path="langgraph.json",
                secrets=[],
                install_command="yarn install",
                build_command="yarn build",
            ),
            {
                "revision_source": "internal_source",
                "source_revision_config": {
                    "source_tarball_path": "tarballs/src.tgz",
                    "langgraph_config_path": "langgraph.json",
                },
                "source_config": {
                    "install_command": "yarn install",
                    "build_command": "yarn build",
                },
                "secrets": [],
            },
            id="internal_source_revision_sends_js_build_commands",
        ),
        pytest.param(
            lambda c: c.update_deployment_internal_source(
                "dep-123",
                source_tarball_path="tarballs/src.tgz",
                config_path="langgraph.json",
            ),
            {
                "revision_source": "internal_source",
                "source_revision_config": {
                    "source_tarball_path": "tarballs/src.tgz",
                    "langgraph_config_path": "langgraph.json",
                },
            },
            id="internal_source_revision_omits_source_config_without_commands",
        ),
        pytest.param(
            lambda c: c.create_deployment(
                name="agent",
                source="external_docker",
                source_config={"resource_spec": {}},
                source_revision_config={
                    "image_uri": "registry.example.com/agent@sha256:1"
                },
                secrets=[],
            ),
            {
                "name": "agent",
                "source": "external_docker",
                "source_config": {"resource_spec": {}},
                "source_revision_config": {
                    "image_uri": "registry.example.com/agent@sha256:1"
                },
                "secrets": [],
            },
            id="create_sends_the_source_configs_as_given",
        ),
        pytest.param(
            lambda c: c.update_deployment(
                "dep-1", "registry.example.com/agent@sha256:2", revision_source=None
            ),
            {
                "source_revision_config": {
                    "image_uri": "registry.example.com/agent@sha256:2"
                }
            },
            id="revision_without_source_override_omits_revision_source",
        ),
        pytest.param(
            lambda c: c.update_deployment(
                "dep-1",
                "registry.example.com/agent@sha256:2",
                revision_source="internal_docker",
                tracked_packages=["langgraph:1.0.0"],
            ),
            {
                "revision_source": "internal_docker",
                "source_revision_config": {
                    "image_uri": "registry.example.com/agent@sha256:2"
                },
                "tracked_packages": ["langgraph:1.0.0"],
            },
            id="revision_with_source_override_names_it",
        ),
    ],
)
def test_request_body_matches_control_plane_contract(call, expected_body):
    captured: dict = {}
    call(_capturing_client(captured))
    assert json.loads(captured["body"]) == expected_body


@pytest.mark.parametrize(
    ("call", "method", "route"),
    [
        pytest.param(
            lambda c: c.create_deployment(
                name="n",
                source="internal_docker",
                source_config={"deployment_type": "dev"},
                source_revision_config={},
            ),
            "POST",
            "/v2/deployments",
            id="create_deployment",
        ),
        pytest.param(
            lambda c: c.get_deployment("dep-1"),
            "GET",
            "/v2/deployments/dep-1",
            id="get_deployment",
        ),
        pytest.param(
            lambda c: c.delete_deployment("dep-1"),
            "DELETE",
            "/v2/deployments/dep-1",
            id="delete_deployment",
        ),
        pytest.param(
            lambda c: c.update_deployment("dep-1", "img", revision_source=None),
            "PATCH",
            "/v2/deployments/dep-1",
            id="patch_deployment",
        ),
        pytest.param(
            lambda c: c.request_push_token("dep-1"),
            "POST",
            "/v2/deployments/dep-1/push-token",
            id="push_token",
        ),
        pytest.param(
            lambda c: c.request_upload_url("dep-1"),
            "POST",
            "/v2/deployments/dep-1/upload-url",
            id="upload_url",
        ),
        pytest.param(
            lambda c: c.list_revisions("dep-1", limit=5),
            "GET",
            "/v2/deployments/dep-1/revisions?limit=5",
            id="list_revisions_puts_limit_in_query",
        ),
        pytest.param(
            lambda c: c.get_revision("dep-1", "rev-2"),
            "GET",
            "/v2/deployments/dep-1/revisions/rev-2",
            id="get_revision",
        ),
        pytest.param(
            lambda c: c.get_build_logs("dep-1", "rev-2", {"limit": 10}),
            "POST",
            "/v1/projects/dep-1/revisions/rev-2/build_logs",
            id="build_logs",
        ),
    ],
)
def test_request_targets_control_plane_route_under_base_url(call, method, route):
    seen: dict = {}
    call(_routing_client(seen))
    assert (seen["method"], seen["url"]) == (
        method,
        f"https://api.example.com/prefix{route}",
    )


def test_injected_transport_receives_requests_under_the_prefixed_base_url():
    seen: dict = {}

    def handler(req: httpx.Request) -> httpx.Response:
        seen["url"] = str(req.url)
        seen["api_key"] = req.headers["x-api-key"]
        return httpx.Response(200, json={"ok": True})

    c = HostBackendClient(
        "https://smith.example.com/api-host",
        "key",
        transport=httpx.MockTransport(handler),
    )

    assert c.list_revisions("dep-1", limit=2) == []
    assert seen == {
        "url": "https://smith.example.com/api-host/v2/deployments/dep-1/revisions?limit=2",
        "api_key": "key",
    }


CLOUD = ("https://api.host.langchain.com", "https://smith.langchain.com")


@pytest.mark.parametrize(
    ("host_url", "langsmith_endpoint", "expected"),
    [
        pytest.param(None, None, CLOUD, id="nothing_configured_targets_cloud"),
        pytest.param(
            None, "https://api.smith.langchain.com", CLOUD, id="cloud_langsmith_api"
        ),
        pytest.param(
            None,
            "https://api.smith.langchain.com/api/v1",
            CLOUD,
            id="cloud_langsmith_api_with_versioned_path",
        ),
        pytest.param(
            None, "https://api.langchain.com", CLOUD, id="cloud_langchain_api_alias"
        ),
        pytest.param(
            None,
            "https://xapi.smith.langchain.com",
            CLOUD,
            id="lookalike_cloud_host_is_not_rewritten_into_a_control_plane",
        ),
        pytest.param(
            None,
            "https://eu.api.smith.langchain.com",
            ("https://eu.api.host.langchain.com", "https://eu.smith.langchain.com"),
            id="eu_cloud_maps_to_eu_control_plane",
        ),
        pytest.param(
            None,
            "https://dev.api.smith.langchain.com",
            ("https://dev.api.host.langchain.com", "https://dev.smith.langchain.com"),
            id="dev_cloud_maps_to_dev_control_plane",
        ),
        pytest.param(
            None,
            "https://aks.smith.langchain.dev/api",
            (
                "https://aks.smith.langchain.dev/api-host",
                "https://aks.smith.langchain.dev",
            ),
            id="self_hosted_api_path_becomes_api_host",
        ),
        pytest.param(
            None,
            "https://smith.example.com/api/v1",
            ("https://smith.example.com/api-host", "https://smith.example.com"),
            id="self_hosted_versioned_api_path_becomes_api_host",
        ),
        pytest.param(
            None,
            "https://smith.example.com",
            ("https://smith.example.com/api-host", "https://smith.example.com"),
            id="self_hosted_origin_gets_api_host_appended",
        ),
        pytest.param(
            None,
            "https://corp.example.com/langsmith/api/v1",
            (
                "https://corp.example.com/langsmith/api-host",
                "https://corp.example.com/langsmith",
            ),
            id="self_hosted_path_prefix_is_kept",
        ),
        pytest.param(
            "https://custom.host.example",
            "https://aks.smith.langchain.dev/api",
            ("https://custom.host.example", "https://smith.langchain.com"),
            id="explicit_host_url_beats_langsmith_endpoint",
        ),
        pytest.param(
            "https://api.host.langchain.com",
            "https://aks.smith.langchain.dev/api",
            CLOUD,
            id="explicit_cloud_host_url_beats_self_hosted_endpoint",
        ),
        pytest.param(
            "https://smith.example.com/api-host/",
            None,
            ("https://smith.example.com/api-host", "https://smith.example.com"),
            id="explicit_api_host_url_derives_dashboard_root",
        ),
        pytest.param(
            "https://corp.example.com/langsmith/api-host",
            None,
            (
                "https://corp.example.com/langsmith/api-host",
                "https://corp.example.com/langsmith",
            ),
            id="explicit_api_host_url_keeps_path_prefix_in_dashboard",
        ),
        pytest.param(
            "http://localhost:8080",
            None,
            ("http://localhost:8080", "http://localhost:8080"),
            id="localhost_dashboard_is_the_same_origin",
        ),
        pytest.param(
            "http://localhost:8080/api-host",
            None,
            ("http://localhost:8080/api-host", "http://localhost:8080"),
            id="localhost_api_host_dashboard_is_the_origin",
        ),
        pytest.param(
            "https://eu.api.host.langchain.com",
            None,
            ("https://eu.api.host.langchain.com", "https://eu.smith.langchain.com"),
            id="regional_control_plane_maps_to_regional_dashboard",
        ),
    ],
)
def test_control_plane_endpoints_resolve(host_url, langsmith_endpoint, expected):
    endpoints = ControlPlaneEndpoints.resolve(host_url, langsmith_endpoint)

    assert (endpoints.control_plane_url, endpoints.dashboard_url) == expected


@pytest.mark.parametrize(
    ("payload", "expected"),
    [
        pytest.param(
            {"resources": [{"id": "a"}, {"id": "b"}]},
            [{"id": "a"}, {"id": "b"}],
            id="list_returns_the_resources",
        ),
        pytest.param({"resources": []}, [], id="empty_list"),
        pytest.param({}, [], id="missing_key"),
        pytest.param({"resources": None}, [], id="null_resources"),
        pytest.param(
            {"resources": ["nope", {"id": "a"}]}, [{"id": "a"}], id="skips_non_objects"
        ),
        pytest.param([], [], id="unexpected_envelope"),
    ],
)
def test_list_endpoints_return_resource_objects(payload, expected):
    def handler(req: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=payload)

    c = HostBackendClient(
        "https://api.example.com", "key", transport=httpx.MockTransport(handler)
    )

    assert c.list_deployments() == expected


def test_list_listeners_asks_for_a_full_page():
    seen: dict = {}

    def handler(req: httpx.Request) -> httpx.Response:
        seen["url"] = str(req.url)
        return httpx.Response(200, json={"resources": [{"id": "listener-1"}]})

    c = HostBackendClient(
        "https://api.example.com", "key", transport=httpx.MockTransport(handler)
    )

    assert c.list_listeners() == [{"id": "listener-1"}]
    assert seen["url"] == "https://api.example.com/v2/listeners?limit=100"


@pytest.mark.parametrize(
    ("control_plane_url", "expected"),
    [
        pytest.param("https://api.host.langchain.com", True, id="cloud"),
        pytest.param("https://eu.api.host.langchain.com", True, id="cloud_region"),
        pytest.param("https://dev.api.host.langchain.com", True, id="cloud_dev"),
        pytest.param("https://smith.example.com/api-host", False, id="self_hosted"),
        pytest.param(
            "https://corp.example.com/langsmith/api-host",
            False,
            id="self_hosted_prefix",
        ),
        pytest.param("http://localhost:8080/api-host", False, id="local"),
        pytest.param(
            "https://evil-api.host.langchain.com", False, id="lookalike_needs_a_dot"
        ),
    ],
)
def test_is_cloud_recognises_the_managed_control_plane(control_plane_url, expected):
    endpoints = ControlPlaneEndpoints.from_control_plane_url(control_plane_url)

    assert endpoints.is_cloud is expected


@pytest.mark.parametrize(
    ("call", "expected_params"),
    [
        pytest.param(
            lambda c: c.list_deployments(name="agent"),
            {"name": "agent"},
            id="exact_name_filters_server_side",
        ),
        pytest.param(
            lambda c: c.list_deployments(name_contains="age"),
            {"name_contains": "age"},
            id="substring_search_keeps_its_own_parameter",
        ),
        pytest.param(
            lambda c: c.list_deployments(),
            {},
            id="no_filter_sends_no_parameters",
        ),
        pytest.param(
            lambda c: c.list_deployments(
                name="agent", name_contains="agent", limit=100
            ),
            {"name": "agent", "name_contains": "agent", "limit": "100"},
            id="both_filters_travel_together_for_older_servers",
        ),
    ],
)
def test_list_deployments_sends_one_name_filter(call, expected_params):
    seen: dict = {}

    def handler(req: httpx.Request) -> httpx.Response:
        seen.update(dict(req.url.params))
        return httpx.Response(200, json={"resources": []})

    call(
        HostBackendClient(
            "https://api.example.com", "key", transport=httpx.MockTransport(handler)
        )
    )

    assert seen == expected_params


@pytest.mark.parametrize(
    ("body", "expected"),
    [
        pytest.param(
            {"detail": "Source configuration error: bad listener"},
            "Source configuration error: bad listener",
            id="fastapi_detail_is_unwrapped",
        ),
        pytest.param(
            {"detail": {"loc": ["body"], "msg": "nope"}},
            None,
            id="a_structured_detail_is_left_alone",
        ),
        pytest.param({"other": "shape"}, None, id="an_unknown_shape_is_left_alone"),
    ],
)
def test_error_detail_is_readable(body, expected):
    c = HostBackendClient(
        "https://api.example.com",
        "key",
        transport=httpx.MockTransport(lambda req: httpx.Response(400, json=body)),
    )

    with pytest.raises(HostBackendError) as error:
        c.get_deployment("dep-1")

    assert error.value.detail == expected
    if expected is not None:
        assert error.value.message.endswith(expected)
