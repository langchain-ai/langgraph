import base64
import json
import os

import click
import httpx
import pytest

from langgraph_cli.cli import (
    _call_host_backend_with_optional_tenant,
    _docker_config_for_token,
    _normalize_image_name,
    _normalize_image_tag,
    _parse_env_from_config,
)
from langgraph_cli.host_backend import HostBackendClient, HostBackendError


class TestDockerConfigForToken:
    def test_creates_config_json(self):
        with _docker_config_for_token("us-docker.pkg.dev", "my-token") as cfg:
            config_path = os.path.join(cfg, "config.json")
            assert os.path.isfile(config_path)
            with open(config_path) as f:
                data = json.load(f)
            expected_auth = base64.b64encode(b"oauth2accesstoken:my-token").decode()
            assert data == {"auths": {"us-docker.pkg.dev": {"auth": expected_auth}}}

    def test_tempdir_cleaned_up(self):
        with _docker_config_for_token("registry.example.com", "tok") as cfg:
            assert os.path.isdir(cfg)
        assert not os.path.exists(cfg)

    def test_different_registries(self):
        with _docker_config_for_token("gcr.io", "token123") as cfg:
            with open(os.path.join(cfg, "config.json")) as f:
                data = json.load(f)
            assert "gcr.io" in data["auths"]


class TestNormalizeImageName:
    def test_simple_name(self):
        assert _normalize_image_name("myapp") == "myapp"

    def test_uppercase_lowered(self):
        assert _normalize_image_name("MyApp") == "myapp"

    def test_special_chars_replaced(self):
        assert _normalize_image_name("my app!@#v2") == "my-app-v2"

    def test_dots_and_hyphens_kept(self):
        assert _normalize_image_name("my-app.v2") == "my-app.v2"

    def test_leading_trailing_stripped(self):
        assert _normalize_image_name("--my-app..") == "my-app"

    def test_empty_string_returns_app(self):
        assert _normalize_image_name("") == "app"

    def test_none_returns_app(self):
        assert _normalize_image_name(None) == "app"

    def test_all_invalid_chars_returns_app(self):
        assert _normalize_image_name("!!!") == "app"


class TestNormalizeImageTag:
    def test_valid_tag(self):
        assert _normalize_image_tag("v1.2.3") == "v1.2.3"

    def test_empty_defaults_to_latest(self):
        assert _normalize_image_tag("") == "latest"

    def test_alphanumeric_and_special(self):
        assert _normalize_image_tag("my_tag-1.0") == "my_tag-1.0"

    def test_invalid_chars_raises(self):
        with pytest.raises(click.UsageError, match="Image tag may only contain"):
            _normalize_image_tag("v1.0:bad")

    def test_spaces_raises(self):
        with pytest.raises(click.UsageError, match="Image tag may only contain"):
            _normalize_image_tag("has space")


class TestParseEnvFromConfig:
    def test_env_dict(self, tmp_path):
        config_path = tmp_path / "langgraph.json"
        config_path.touch()
        result = _parse_env_from_config({"env": {"FOO": "bar", "NUM": 42}}, config_path)
        assert result == {"FOO": "bar", "NUM": "42"}

    def test_env_string_dotenv_file(self, tmp_path):
        env_file = tmp_path / "my.env"
        env_file.write_text("KEY1=val1\nKEY2=val2\n")
        config_path = tmp_path / "langgraph.json"
        config_path.touch()
        result = _parse_env_from_config({"env": "my.env"}, config_path)
        assert result == {"KEY1": "val1", "KEY2": "val2"}

    def test_env_missing_falls_back_to_dotenv(self, tmp_path, monkeypatch):
        env_file = tmp_path / ".env"
        env_file.write_text("DEFAULT_KEY=default_val\n")
        monkeypatch.chdir(tmp_path)
        config_path = tmp_path / "langgraph.json"
        config_path.touch()
        result = _parse_env_from_config({}, config_path)
        assert result == {"DEFAULT_KEY": "default_val"}

    def test_env_empty_dict_falls_back_to_dotenv(self, tmp_path, monkeypatch):
        """validate_config defaults env to {}, should still fall back to .env."""
        env_file = tmp_path / ".env"
        env_file.write_text("MY_KEY=my_val\n")
        monkeypatch.chdir(tmp_path)
        config_path = tmp_path / "langgraph.json"
        config_path.touch()
        result = _parse_env_from_config({"env": {}}, config_path)
        assert result == {"MY_KEY": "my_val"}

    def test_env_missing_no_dotenv_returns_empty(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        config_path = tmp_path / "langgraph.json"
        config_path.touch()
        result = _parse_env_from_config({}, config_path)
        assert result == {}

    def test_env_dotenv_filters_none_values(self, tmp_path):
        # Lines like "KEY=" produce empty string, lines like "KEY" produce None
        env_file = tmp_path / "test.env"
        env_file.write_text("GOOD=value\nEMPTY=\n")
        config_path = tmp_path / "langgraph.json"
        config_path.touch()
        result = _parse_env_from_config({"env": "test.env"}, config_path)
        assert "GOOD" in result
        assert result["GOOD"] == "value"
        # EMPTY= gives empty string, not None, so it should be present
        assert result["EMPTY"] == ""


class TestCallHostBackendWithOptionalTenant:
    def _make_client(self, handler):
        c = HostBackendClient("https://api.example.com", "test-key")
        c._client = httpx.Client(
            base_url="https://api.example.com",
            transport=httpx.MockTransport(handler),
            headers={"X-Api-Key": "test-key", "Accept": "application/json"},
            timeout=30,
        )
        return c

    def _make_eu_client(self, handler):
        c = HostBackendClient("https://eu.api.host.langchain.com", "test-key")
        c._client = httpx.Client(
            base_url="https://eu.api.host.langchain.com",
            transport=httpx.MockTransport(handler),
            headers={"X-Api-Key": "test-key", "Accept": "application/json"},
            timeout=30,
        )
        return c

    def test_success_passes_through(self):
        client = self._make_client(lambda req: httpx.Response(200, json={"ok": True}))
        result = _call_host_backend_with_optional_tenant(
            client, lambda c: c.list_deployments()
        )
        assert result == {"ok": True}

    def test_403_not_enabled_gives_actionable_error(self):
        detail = (
            '{"detail":"LangSmith Deployment is not enabled for this organization"}'
        )
        client = self._make_client(lambda req: httpx.Response(403, text=detail))
        with pytest.raises(HostBackendError, match="not enabled") as exc_info:
            _call_host_backend_with_optional_tenant(
                client, lambda c: c.list_deployments()
            )
        assert exc_info.value.status_code == 403
        assert "smith.langchain.com" in exc_info.value.message

    def test_403_not_enabled_eu_url(self):
        detail = (
            '{"detail":"LangSmith Deployment is not enabled for this organization"}'
        )
        client = self._make_eu_client(lambda req: httpx.Response(403, text=detail))
        with pytest.raises(HostBackendError, match="not enabled") as exc_info:
            _call_host_backend_with_optional_tenant(
                client, lambda c: c.list_deployments()
            )
        assert "eu.smith.langchain.com" in exc_info.value.message

    def test_workspace_retry_then_not_enabled_gives_actionable_error(self, monkeypatch):
        requires_workspace = '{"detail":"requires workspace specification"}'
        not_enabled = (
            '{"detail":"LangSmith Deployment is not enabled for this organization"}'
        )
        seen_tenant_ids = []

        def handler(req):
            seen_tenant_ids.append(req.headers.get("X-Tenant-ID"))
            if len(seen_tenant_ids) == 1:
                return httpx.Response(403, text=requires_workspace)
            if len(seen_tenant_ids) == 2:
                return httpx.Response(403, text=not_enabled)
            raise AssertionError("unexpected extra request")

        monkeypatch.setattr(click, "prompt", lambda _text: "workspace-123")
        client = self._make_client(handler)

        with pytest.raises(HostBackendError, match="not enabled") as exc_info:
            _call_host_backend_with_optional_tenant(
                client, lambda c: c.list_deployments()
            )

        assert exc_info.value.status_code == 403
        assert "smith.langchain.com" in exc_info.value.message
        assert seen_tenant_ids == [None, "workspace-123"]
        assert client._client.headers["X-Tenant-ID"] == "workspace-123"

    def test_other_403_re_raises_original(self):
        client = self._make_client(
            lambda req: httpx.Response(403, text='{"detail":"some other error"}')
        )
        with pytest.raises(HostBackendError, match="some other error"):
            _call_host_backend_with_optional_tenant(
                client, lambda c: c.list_deployments()
            )
