import base64
import io
import json
import os
import sys

import click
import httpx
import pytest

from langgraph_cli.deploy import (
    _call_host_backend_with_optional_tenant,
    _create_host_backend_client,
    _docker_config_for_token,
    _Emitter,
    _env_without_deployment_name,
    _parse_env_from_config,
    _resolve_env_path,
    _smith_dashboard_base_url,
    normalize_image_name,
    normalize_image_tag,
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
        assert normalize_image_name("myapp") == "myapp"

    def test_uppercase_lowered(self):
        assert normalize_image_name("MyApp") == "myapp"

    def test_special_chars_replaced(self):
        assert normalize_image_name("my app!@#v2") == "my-app-v2"

    def test_dots_and_hyphens_kept(self):
        assert normalize_image_name("my-app.v2") == "my-app.v2"

    def test_leading_trailing_stripped(self):
        assert normalize_image_name("--my-app..") == "my-app"

    def test_empty_string_returns_app(self):
        assert normalize_image_name("") == "app"

    def test_none_returns_app(self):
        assert normalize_image_name(None) == "app"

    def test_all_invalid_chars_returns_app(self):
        assert normalize_image_name("!!!") == "app"


class TestNormalizeImageTag:
    def test_valid_tag(self):
        assert normalize_image_tag("v1.2.3") == "v1.2.3"

    def test_empty_defaults_to_latest(self):
        assert normalize_image_tag("") == "latest"

    def test_alphanumeric_and_special(self):
        assert normalize_image_tag("my_tag-1.0") == "my_tag-1.0"

    def test_invalid_chars_raises(self):
        with pytest.raises(click.UsageError, match="Image tag may only contain"):
            normalize_image_tag("v1.0:bad")

    def test_spaces_raises(self):
        with pytest.raises(click.UsageError, match="Image tag may only contain"):
            normalize_image_tag("has space")


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


class TestResolveEnvPath:
    def test_inline_env_dict_returns_none(self, tmp_path):
        config_path = tmp_path / "langgraph.json"
        config_path.touch()
        assert _resolve_env_path({"env": {"FOO": "bar"}}, config_path) is None

    def test_relative_env_path_resolves(self, tmp_path):
        env_file = tmp_path / "custom.env"
        env_file.write_text("FOO=bar\n")
        config_path = tmp_path / "langgraph.json"
        config_path.touch()

        resolved = _resolve_env_path({"env": "custom.env"}, config_path)
        assert resolved == env_file.resolve()

    def test_missing_env_file_returns_none(self, tmp_path):
        config_path = tmp_path / "langgraph.json"
        config_path.touch()
        assert _resolve_env_path({"env": "missing.env"}, config_path) is None

    def test_default_env_is_cwd_dotenv(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        config_path = tmp_path / "langgraph.json"
        config_path.touch()
        assert _resolve_env_path({}, config_path) == tmp_path / ".env"


class TestEnvWithoutDeploymentName:
    def test_removes_deployment_name_only(self):
        env = {
            "LANGSMITH_DEPLOYMENT_NAME": "my-deploy",
            "KEEP_ME": "value",
        }
        cleaned = _env_without_deployment_name(env)

        assert "LANGSMITH_DEPLOYMENT_NAME" not in cleaned
        assert cleaned["KEEP_ME"] == "value"
        # Original dict should be unchanged.
        assert env["LANGSMITH_DEPLOYMENT_NAME"] == "my-deploy"

    def test_noop_when_deployment_name_absent(self):
        env = {"FOO": "bar"}
        assert _env_without_deployment_name(env) == {"FOO": "bar"}


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

    def test_workspace_prompt_blocked_by_no_input(self, monkeypatch):
        """With _no_input=True, 403 requiring workspace should raise ClickException."""
        import langgraph_cli.deploy as deploy_mod

        monkeypatch.setattr(deploy_mod, "_no_input", True)

        requires_workspace = '{"detail":"requires workspace specification"}'
        client = self._make_client(
            lambda req: httpx.Response(403, text=requires_workspace)
        )
        with pytest.raises(click.ClickException, match="workspace"):
            _call_host_backend_with_optional_tenant(
                client, lambda c: c.list_deployments()
            )


# ---------------------------------------------------------------------------
# _Emitter JSON mode
# ---------------------------------------------------------------------------


class TestEmitterJsonMode:
    """Verify that _Emitter in json_mode writes valid JSON-lines to stdout."""

    def _capture(self, fn):
        """Run fn with stdout captured and return parsed JSON objects."""
        buf = io.StringIO()
        old = sys.stdout
        sys.stdout = buf
        try:
            fn()
        finally:
            sys.stdout = old
        lines = [l for l in buf.getvalue().splitlines() if l.strip()]
        return [json.loads(l) for l in lines]

    def test_step_event(self):
        em = _Emitter(json_mode=True)
        events = self._capture(lambda: em.step(1, "Building image"))
        assert len(events) == 1
        assert events[0]["event"] == "step"
        assert events[0]["step"] == 1
        assert events[0]["message"] == "Building image"

    def test_info_event(self):
        em = _Emitter(json_mode=True)
        events = self._capture(lambda: em.info("All good"))
        assert events[0]["event"] == "info"
        assert events[0]["message"] == "All good"

    def test_warn_event(self):
        em = _Emitter(json_mode=True)
        events = self._capture(lambda: em.warn("Careful"))
        assert events[0]["event"] == "warn"

    def test_error_event(self):
        em = _Emitter(json_mode=True)
        events = self._capture(lambda: em.error("Boom"))
        assert events[0]["event"] == "error"
        assert events[0]["message"] == "Boom"

    def test_status_change_event(self):
        em = _Emitter(json_mode=True)
        events = self._capture(lambda: em.status_change("building", 12.345))
        assert events[0]["event"] == "status_change"
        assert events[0]["status"] == "building"
        assert events[0]["elapsed_seconds"] == 12.3
        assert events[0]["message"] == "building... (12s)"

    def test_status_change_event_with_minutes(self):
        em = _Emitter(json_mode=True)
        events = self._capture(lambda: em.status_change("deploying", 95.0))
        assert events[0]["message"] == "deploying... (1m 35s)"

    def test_log_event(self):
        em = _Emitter(json_mode=True)
        events = self._capture(lambda: em.log("some output"))
        assert events[0] == {"event": "log", "message": "some output"}

    def test_status_url_event(self):
        em = _Emitter(json_mode=True)
        events = self._capture(
            lambda: em.status_url("https://smith.langchain.com/deploy/123")
        )
        assert events[0]["event"] == "status_url"
        assert events[0]["url"] == "https://smith.langchain.com/deploy/123"

    def test_result_event_full(self):
        em = _Emitter(json_mode=True)
        events = self._capture(
            lambda: em.result(
                "succeeded",
                deployment_id="dep-1",
                url="https://app.example.com",
                status_url="https://smith.langchain.com/deploy/dep-1",
            )
        )
        assert events[0]["event"] == "result"
        assert events[0]["status"] == "succeeded"
        assert events[0]["deployment_id"] == "dep-1"
        assert events[0]["message"] == "Deployment successful!"
        assert events[0]["url"] == "https://app.example.com"
        assert events[0]["status_url"] == "https://smith.langchain.com/deploy/dep-1"

    def test_result_event_minimal(self):
        em = _Emitter(json_mode=True)
        events = self._capture(lambda: em.result("failed", deployment_id="dep-2"))
        assert events[0]["event"] == "result"
        assert events[0]["status"] == "failed"
        assert events[0]["message"] == "Deployment failed"
        assert "url" not in events[0]
        assert "status_url" not in events[0]

    def test_heartbeat_event(self):
        em = _Emitter(json_mode=True)
        events = self._capture(lambda: em.heartbeat("building", 30.789))
        assert events[0]["event"] == "heartbeat"
        assert events[0]["elapsed_seconds"] == 30.8
        assert events[0]["message"] == "building... (30s)"

    def test_heartbeat_silent_in_text_mode(self, capsys):
        em = _Emitter(json_mode=False)
        em.heartbeat("building", 10.0)
        captured = capsys.readouterr()
        assert captured.out == ""

    def test_upload_progress_event(self):
        em = _Emitter(json_mode=True)
        events = self._capture(lambda: em.upload_progress(5.678, 42))
        assert events[0]["event"] == "upload_progress"
        assert events[0]["size_mb"] == 5.7
        assert events[0]["pct"] == 42


# ---------------------------------------------------------------------------
# _Emitter text mode (non-json)
# ---------------------------------------------------------------------------


class TestEmitterTextMode:
    """Verify that _Emitter in text mode uses click.echo/click.secho."""

    def test_step_writes_text(self, capsys):
        em = _Emitter(json_mode=False)
        em.step(1, "Hello")
        captured = capsys.readouterr()
        assert "1. Hello" in captured.out

    def test_log_writes_text(self, capsys):
        em = _Emitter(json_mode=False)
        em.log("my line")
        captured = capsys.readouterr()
        assert "my line" in captured.out

    def test_result_succeeded_text(self, capsys):
        em = _Emitter(json_mode=False)
        em.result("succeeded", deployment_id="d1", url="https://app.test")
        captured = capsys.readouterr()
        assert "successful" in captured.out.lower()
        assert "https://app.test" in captured.out


# ---------------------------------------------------------------------------
# --no-input guard on _create_host_backend_client
# ---------------------------------------------------------------------------


class TestCreateHostBackendClientNoInput:
    def test_raises_when_no_api_key_and_no_input(self, monkeypatch, tmp_path):
        import langgraph_cli.deploy as deploy_mod

        monkeypatch.setattr(deploy_mod, "_no_input", True)
        monkeypatch.delenv("LANGSMITH_API_KEY", raising=False)
        monkeypatch.delenv("LANGCHAIN_API_KEY", raising=False)
        monkeypatch.delenv("LANGGRAPH_HOST_API_KEY", raising=False)

        with pytest.raises(click.ClickException, match="API key"):
            _create_host_backend_client(
                host_url="https://api.example.com",
                api_key=None,
                env_vars={},
            )

    def test_succeeds_with_api_key_in_env(self, monkeypatch, tmp_path):
        import langgraph_cli.deploy as deploy_mod

        monkeypatch.setattr(deploy_mod, "_no_input", True)
        monkeypatch.setenv("LANGSMITH_API_KEY", "lsv2_test")

        client = _create_host_backend_client(
            host_url="https://api.example.com",
            api_key=None,
            env_vars={},
        )
        assert client is not None


class TestSmithDashboardBaseUrl:
    def test_none_returns_default(self):
        assert _smith_dashboard_base_url(None) == "https://smith.langchain.com"

    def test_empty_returns_default(self):
        assert _smith_dashboard_base_url("") == "https://smith.langchain.com"

    def test_prod_host_url(self):
        assert (
            _smith_dashboard_base_url("https://api.host.langchain.com")
            == "https://smith.langchain.com"
        )

    def test_dev_host_url(self):
        assert (
            _smith_dashboard_base_url("https://dev.api.host.langchain.com")
            == "https://dev.smith.langchain.com"
        )

    def test_eu_host_url(self):
        assert (
            _smith_dashboard_base_url("https://eu.api.host.langchain.com")
            == "https://eu.smith.langchain.com"
        )

    def test_staging_host_url(self):
        assert (
            _smith_dashboard_base_url("https://staging.api.host.langchain.com")
            == "https://staging.smith.langchain.com"
        )

    def test_localhost(self):
        assert (
            _smith_dashboard_base_url("http://localhost:8080")
            == "http://localhost:8080"
        )

    def test_localhost_trailing_slash(self):
        assert (
            _smith_dashboard_base_url("http://localhost:8080/")
            == "http://localhost:8080"
        )

    def test_127_0_0_1(self):
        assert (
            _smith_dashboard_base_url("http://127.0.0.1:3000")
            == "http://127.0.0.1:3000"
        )

    def test_unknown_domain_returns_default(self):
        assert (
            _smith_dashboard_base_url("https://custom.example.com")
            == "https://smith.langchain.com"
        )
