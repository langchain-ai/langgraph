import asyncio
import base64
import io
import json
import os
import sys
from unittest.mock import MagicMock

import click
import click.exceptions
import httpx
import pytest

import langgraph_cli.deploy as deploy_mod
from langgraph_cli.deploy import (
    ById,
    ByName,
    CustomerRegistrySource,
    DockerBuildCommand,
    ExistingDeployment,
    Listener,
    ManagedRegistrySource,
    OnListener,
    RemoteBuildSource,
    RequestedPlacement,
    Unplaced,
    _call_host_backend_with_optional_tenant,
    _create_host_backend_client,
    _docker_config_for_token,
    _Emitter,
    _env_without_deployment_name,
    _parse_env_from_config,
    _resolve_env_path,
    _resolve_pushed_image_digest,
    _select_source,
    _validate_prebuilt_image,
    find_deployment_by_name,
    normalize_image_tag,
    normalize_name,
)
from langgraph_cli.host_backend import HostBackendClient, HostBackendError
from langgraph_cli.image_reference import ImageReference


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


class TestNormalizeName:
    def test_simple_name(self):
        assert normalize_name("myapp") == "myapp"

    def test_uppercase_lowered(self):
        assert normalize_name("MyApp") == "myapp"

    def test_special_chars_replaced(self):
        assert normalize_name("my app!@#v2") == "my-app-v2"

    def test_dots_replaced_with_hyphens(self):
        assert normalize_name("my-app.v2") == "my-app-v2"

    def test_underscores_replaced_with_hyphens(self):
        assert normalize_name("simple_graph_name") == "simple-graph-name"

    def test_leading_trailing_stripped(self):
        assert normalize_name("--my-app..") == "my-app"

    def test_empty_string_returns_app(self):
        assert normalize_name("") == "app"

    def test_none_returns_app(self):
        assert normalize_name(None) == "app"

    def test_all_invalid_chars_returns_app(self):
        assert normalize_name("!!!") == "app"


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


class _FakeRunner:
    def run(self, coro):
        return asyncio.run(coro)


class TestValidatePrebuiltImage:
    def test_accepts_linux_amd64(self, monkeypatch):
        calls = []

        async def fake_subp_exec(*args, **kwargs):
            calls.append((args, kwargs))
            return "linux/amd64\n", None

        monkeypatch.setattr(deploy_mod, "subp_exec", fake_subp_exec)

        _validate_prebuilt_image(_FakeRunner(), "repo/app:tag", verbose=False)

        assert calls == [
            (
                (
                    "docker",
                    "image",
                    "inspect",
                    "--format",
                    "{{.Os}}/{{.Architecture}}",
                    "repo/app:tag",
                ),
                {"verbose": False, "collect": True},
            )
        ]

    def test_missing_docker_binary_raises_actionable_error(self, monkeypatch):
        async def fake_subp_exec(*args, **kwargs):
            raise FileNotFoundError("docker")

        monkeypatch.setattr(deploy_mod, "subp_exec", fake_subp_exec)

        with pytest.raises(click.ClickException, match="Docker is required"):
            _validate_prebuilt_image(_FakeRunner(), "repo/app:tag", verbose=False)

    def test_missing_image_raises_actionable_error(self, monkeypatch):
        async def fake_subp_exec(*args, **kwargs):
            raise click.exceptions.Exit(1)

        monkeypatch.setattr(deploy_mod, "subp_exec", fake_subp_exec)

        with pytest.raises(click.ClickException, match="not found locally"):
            _validate_prebuilt_image(_FakeRunner(), "missing:tag", verbose=False)

    def test_rejects_non_amd64_platform(self, monkeypatch):
        async def fake_subp_exec(*args, **kwargs):
            return "linux/arm64\n", None

        monkeypatch.setattr(deploy_mod, "subp_exec", fake_subp_exec)

        with pytest.raises(click.ClickException, match="requires linux/amd64"):
            _validate_prebuilt_image(_FakeRunner(), "repo/app:arm", verbose=False)


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
        c = HostBackendClient(
            "https://api.example.com",
            "test-key",
            transport=httpx.MockTransport(handler),
        )
        return c

    def _make_eu_client(self, handler):
        c = HostBackendClient(
            "https://eu.api.host.langchain.com",
            "test-key",
            transport=httpx.MockTransport(handler),
        )
        return c

    def test_success_passes_through(self):
        client = self._make_client(
            lambda req: httpx.Response(200, json={"resources": [{"id": "dep-1"}]})
        )
        result = _call_host_backend_with_optional_tenant(
            client, lambda c: c.list_deployments()
        )
        assert result == [{"id": "dep-1"}]

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
        lines = [line for line in buf.getvalue().splitlines() if line.strip()]
        return [json.loads(line) for line in lines]

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
        lines = [line.strip() for line in captured.out.splitlines() if line.strip()]
        assert "Deployment successful!" in lines
        assert "URL: https://app.test" in lines


# ---------------------------------------------------------------------------
# --no-input guard on _create_host_backend_client
# ---------------------------------------------------------------------------


class TestCreateHostBackendClientNoInput:
    def test_raises_when_no_api_key_and_no_input(self, monkeypatch, tmp_path):

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

        monkeypatch.setattr(deploy_mod, "_no_input", True)
        monkeypatch.setenv("LANGSMITH_API_KEY", "lsv2_test")

        client = _create_host_backend_client(
            host_url="https://api.example.com",
            api_key=None,
            env_vars={},
        )
        assert client is not None


class TestCreateHostBackendClientEndpoint:
    def test_langsmith_endpoint_from_project_env_selects_self_hosted_control_plane(
        self, monkeypatch
    ):
        monkeypatch.setenv("LANGSMITH_API_KEY", "lsv2_test")
        monkeypatch.delenv("LANGSMITH_ENDPOINT", raising=False)

        client = _create_host_backend_client(
            host_url=None,
            api_key=None,
            env_vars={"LANGSMITH_ENDPOINT": "https://smith.example.com/api/v1"},
        )

        assert client.base_url == "https://smith.example.com/api-host"

    def test_explicit_host_url_wins_over_langsmith_endpoint(self, monkeypatch):
        monkeypatch.setenv("LANGSMITH_API_KEY", "lsv2_test")
        monkeypatch.setenv("LANGSMITH_ENDPOINT", "https://smith.example.com/api/v1")

        client = _create_host_backend_client(
            host_url="https://custom.host.com", api_key=None, env_vars={}
        )

        assert client.base_url == "https://custom.host.com"


class TestDockerBuildCommand:
    @pytest.mark.parametrize(
        ("machine", "verbose", "expected"),
        [
            pytest.param(
                "x86_64",
                False,
                DockerBuildCommand(("docker", "build"), ()),
                id="amd64_host_builds_natively",
            ),
            pytest.param(
                "arm64",
                False,
                DockerBuildCommand(
                    ("docker", "buildx", "build"),
                    ("--platform", "linux/amd64", "--load", "--progress=quiet"),
                ),
                id="other_hosts_cross_build_quietly",
            ),
            pytest.param(
                "arm64",
                True,
                DockerBuildCommand(
                    ("docker", "buildx", "build"),
                    ("--platform", "linux/amd64", "--load"),
                ),
                id="verbose_cross_build_keeps_progress_output",
            ),
        ],
    )
    def test_for_host_targets_the_deployment_platform(self, machine, verbose, expected):
        assert DockerBuildCommand.for_host(machine, verbose=verbose) == expected


class TestSelectSource:
    OPTIONS = {
        "push_to": None,
        "image": None,
        "image_name": None,
        "tag": None,
        "remote_build_flag": None,
        "placement": RequestedPlacement(),
        "selector": ByName("my-app"),
    }
    REPOSITORY = "registry.example.com/app"

    @pytest.mark.parametrize(
        ("flags", "docker_available", "expected"),
        [
            pytest.param(
                {"push_to": REPOSITORY},
                True,
                CustomerRegistrySource(
                    reference=ImageReference(REPOSITORY, "latest"),
                    prebuilt_image=None,
                    requested_placement=RequestedPlacement(),
                ),
                id="push_to_selects_the_external_source_with_the_default_tag",
            ),
            pytest.param(
                {"push_to": f"{REPOSITORY}:v2"},
                True,
                CustomerRegistrySource(
                    reference=ImageReference(REPOSITORY, "v2"),
                    prebuilt_image=None,
                    requested_placement=RequestedPlacement(),
                ),
                id="push_to_keeps_a_tag_given_in_the_reference",
            ),
            pytest.param(
                {"push_to": REPOSITORY, "tag": "v3"},
                True,
                CustomerRegistrySource(
                    reference=ImageReference(REPOSITORY, "v3"),
                    prebuilt_image=None,
                    requested_placement=RequestedPlacement(),
                ),
                id="tag_flag_composes_with_push_to",
            ),
            pytest.param(
                {"push_to": REPOSITORY, "image": "app:dev"},
                False,
                CustomerRegistrySource(
                    reference=ImageReference(REPOSITORY, "latest"),
                    prebuilt_image="app:dev",
                    requested_placement=RequestedPlacement(),
                ),
                id="prebuilt_image_is_retagged_for_push_to_without_docker_checks",
            ),
            pytest.param(
                {
                    "push_to": REPOSITORY,
                    "placement": RequestedPlacement("listener-1", "agents"),
                },
                True,
                CustomerRegistrySource(
                    reference=ImageReference(REPOSITORY, "latest"),
                    prebuilt_image=None,
                    requested_placement=RequestedPlacement("listener-1", "agents"),
                ),
                id="push_to_carries_the_requested_placement",
            ),
            pytest.param(
                {"remote_build_flag": True},
                True,
                RemoteBuildSource(),
                id="remote_flag_selects_the_source_upload",
            ),
            pytest.param(
                {},
                False,
                RemoteBuildSource(),
                id="no_local_docker_falls_back_to_the_source_upload",
            ),
            pytest.param(
                {},
                True,
                ManagedRegistrySource(
                    prebuilt_image=None, image_name=None, tag="latest"
                ),
                id="local_docker_selects_the_internal_docker_source",
            ),
            pytest.param(
                {"image": "app:dev", "tag": "v1"},
                False,
                ManagedRegistrySource(
                    prebuilt_image="app:dev", image_name=None, tag="v1"
                ),
                id="prebuilt_image_forces_the_internal_docker_source",
            ),
        ],
    )
    def test_flags_select_one_source(
        self, monkeypatch, mocker, flags, docker_available, expected
    ):
        mocker.patch(
            "langgraph_cli.deploy._get_emitter", return_value=mocker.MagicMock()
        )
        monkeypatch.setattr(
            deploy_mod,
            "can_build_locally",
            lambda: (True, None) if docker_available else (False, "Docker is required"),
        )

        assert _select_source(**{**self.OPTIONS, **flags}) == expected

    def test_push_to_build_requires_local_docker(self, monkeypatch):
        monkeypatch.setattr(
            deploy_mod, "can_build_locally", lambda: (False, "Docker is required")
        )

        with pytest.raises(click.UsageError, match="Docker is required"):
            _select_source(**{**self.OPTIONS, "push_to": self.REPOSITORY})

    @pytest.mark.parametrize(
        ("flags", "message"),
        [
            pytest.param(
                {"push_to": REPOSITORY, "remote_build_flag": True},
                "--push-to cannot be combined with --remote.",
                id="push_to_with_remote",
            ),
            pytest.param(
                {"push_to": f"{REPOSITORY}:v1", "tag": "v2"},
                "already includes a tag",
                id="push_to_with_a_tag_and_the_tag_flag",
            ),
            pytest.param(
                {"push_to": f"{REPOSITORY}@sha256:abc"},
                "not a digest",
                id="push_to_with_a_digest",
            ),
            pytest.param(
                {"image": "app:dev", "remote_build_flag": True},
                "--image cannot be combined with --remote builds.",
                id="image_with_remote",
            ),
            pytest.param(
                {"placement": RequestedPlacement(listener_id="listener-1")},
                "only apply when creating a deployment with --push-to",
                id="listener_without_push_to",
            ),
            pytest.param(
                {"placement": RequestedPlacement(k8s_namespace="agents")},
                "only apply when creating a deployment with --push-to",
                id="namespace_without_push_to",
            ),
        ],
    )
    def test_conflicting_flags_are_rejected(self, monkeypatch, flags, message):
        monkeypatch.setattr(deploy_mod, "can_build_locally", lambda: (True, None))

        with pytest.raises(click.UsageError, match=message):
            _select_source(**{**self.OPTIONS, **flags})


class TestResolvePushedImageDigest:
    """Tests for ``_resolve_pushed_image_digest`` — runner is mocked to
    return the ``(stdout, stderr)`` tuple that ``subp_exec(collect=True)``
    would produce.
    """

    @staticmethod
    def _runner(stdout: str | None) -> MagicMock:
        # Close the unawaited subp_exec coroutine to silence gc warnings.
        runner = MagicMock()

        def _run(coro, *args, **kwargs):
            if hasattr(coro, "close"):
                coro.close()
            return (stdout, "")

        runner.run.side_effect = _run
        return runner

    def test_happy_path_returns_digest(self):
        runner = self._runner('["us-central1-docker.pkg.dev/proj/repo@sha256:abc123"]')
        out = _resolve_pushed_image_digest(
            runner,
            remote_image="us-central1-docker.pkg.dev/proj/repo:latest",
            docker_config_dir=None,
            verbose=False,
        )
        assert out == "us-central1-docker.pkg.dev/proj/repo@sha256:abc123"

    def test_filters_to_matching_repo(self):
        # Same image ID can hold digests for multiple repos — pick the one
        # matching the just-pushed repo.
        runner = self._runner(
            json.dumps(
                [
                    "other-registry.example.com/some/repo@sha256:000000",
                    "us-central1-docker.pkg.dev/proj/repo@sha256:abc123",
                ]
            )
        )
        out = _resolve_pushed_image_digest(
            runner,
            remote_image="us-central1-docker.pkg.dev/proj/repo:v1.2.3",
            docker_config_dir=None,
            verbose=False,
        )
        assert out == "us-central1-docker.pkg.dev/proj/repo@sha256:abc123"

    def test_registry_port_without_tag_still_resolves_the_digest(self):
        runner = self._runner('["localhost:5000/repo@sha256:abc123"]')
        out = _resolve_pushed_image_digest(
            runner,
            remote_image="localhost:5000/repo",
            docker_config_dir=None,
            verbose=False,
        )
        assert out == "localhost:5000/repo@sha256:abc123"

    def test_empty_repodigests_falls_back_with_warning(self, mocker):
        emitter = mocker.MagicMock()
        mocker.patch("langgraph_cli.deploy._get_emitter", return_value=emitter)
        runner = self._runner("[]")
        remote = "us-central1-docker.pkg.dev/proj/repo:latest"
        out = _resolve_pushed_image_digest(
            runner,
            remote_image=remote,
            docker_config_dir=None,
            verbose=False,
        )
        assert out == remote
        assert emitter.warn.called
        assert remote in emitter.warn.call_args.args[0]

    def test_null_repodigests_falls_back_with_warning(self, mocker):
        # ``docker inspect --format '{{json .RepoDigests}}'`` emits ``null``
        # when the field is absent.
        emitter = mocker.MagicMock()
        mocker.patch("langgraph_cli.deploy._get_emitter", return_value=emitter)
        runner = self._runner("null")
        remote = "us-central1-docker.pkg.dev/proj/repo:latest"
        out = _resolve_pushed_image_digest(
            runner,
            remote_image=remote,
            docker_config_dir=None,
            verbose=False,
        )
        assert out == remote
        assert emitter.warn.called

    def test_no_matching_repo_falls_back_with_warning(self, mocker):
        # No matching digest for the pushed repo — warn and fall back to the
        # tag-based ref rather than failing the deploy.
        emitter = mocker.MagicMock()
        mocker.patch("langgraph_cli.deploy._get_emitter", return_value=emitter)
        runner = self._runner('["other-registry.example.com/some/repo@sha256:000000"]')
        remote = "us-central1-docker.pkg.dev/proj/repo:latest"
        out = _resolve_pushed_image_digest(
            runner,
            remote_image=remote,
            docker_config_dir=None,
            verbose=False,
        )
        assert out == remote
        assert emitter.warn.called

    def test_registry_with_port_in_host(self):
        # Only the rightmost ``:`` (the ``:latest`` tag) should be stripped.
        runner = self._runner('["localhost:5000/repo@sha256:deadbeef"]')
        out = _resolve_pushed_image_digest(
            runner,
            remote_image="localhost:5000/repo:latest",
            docker_config_dir=None,
            verbose=False,
        )
        assert out == "localhost:5000/repo@sha256:deadbeef"

    @staticmethod
    def _capturing_runner(stdout: str) -> tuple[MagicMock, dict]:
        """Like ``_runner`` but exposes the coroutine for arg introspection.
        Caller must close ``captured["coro"]``.
        """
        runner = MagicMock()
        captured: dict = {}

        def _run(coro, *args, **kwargs):
            captured["coro"] = coro
            return (stdout, "")

        runner.run.side_effect = _run
        return runner, captured

    def test_passes_docker_config_dir(self):
        runner, captured = self._capturing_runner(
            '["us-central1-docker.pkg.dev/proj/repo@sha256:abc"]'
        )
        _resolve_pushed_image_digest(
            runner,
            remote_image="us-central1-docker.pkg.dev/proj/repo:latest",
            docker_config_dir="/tmp/some-cfg",
            verbose=False,
        )
        frame_locals = captured["coro"].cr_frame.f_locals
        assert frame_locals["cmd"] == "docker"
        assert "--config" in frame_locals["args"]
        cfg_idx = frame_locals["args"].index("--config")
        assert frame_locals["args"][cfg_idx + 1] == "/tmp/some-cfg"
        captured["coro"].close()

    def test_omits_docker_config_dir_when_none(self):
        runner, captured = self._capturing_runner(
            '["us-central1-docker.pkg.dev/proj/repo@sha256:abc"]'
        )
        _resolve_pushed_image_digest(
            runner,
            remote_image="us-central1-docker.pkg.dev/proj/repo:latest",
            docker_config_dir=None,
            verbose=False,
        )
        frame_locals = captured["coro"].cr_frame.f_locals
        assert "--config" not in frame_locals["args"]
        captured["coro"].close()


class TestListener:
    @pytest.mark.parametrize(
        ("resource", "expected"),
        [
            pytest.param(
                {
                    "id": "listener-1",
                    "compute_id": "prod-cluster",
                    "compute_config": {"k8s_namespaces": ["agents", "agents-staging"]},
                },
                Listener("listener-1", "prod-cluster", ("agents", "agents-staging")),
                id="reads_id_cluster_and_namespaces",
            ),
            pytest.param(
                {"id": "listener-1", "compute_id": "c", "compute_config": {}},
                Listener("listener-1", "c", ()),
                id="missing_namespaces",
            ),
            pytest.param(
                {"id": "listener-1", "compute_id": "c", "compute_config": None},
                Listener("listener-1", "c", ()),
                id="null_compute_config",
            ),
            pytest.param(
                {"id": "listener-1"},
                Listener("listener-1", "", ()),
                id="only_an_id",
            ),
        ],
    )
    def test_from_resource_reads_the_control_plane_shape(self, resource, expected):
        assert Listener.from_resource(resource) == expected


ONE_NAMESPACE = Listener("listener-1", "prod-cluster", ("agents",))
TWO_NAMESPACES = Listener("listener-2", "multi-cluster", ("agents", "agents-staging"))
NO_NAMESPACE = Listener("listener-3", "broken-cluster", ())


class TestRequestedPlacement:
    @pytest.mark.parametrize(
        ("request_", "listeners", "expected"),
        [
            pytest.param(
                RequestedPlacement(), (), Unplaced(), id="no_listeners_no_request"
            ),
            pytest.param(
                RequestedPlacement(),
                (ONE_NAMESPACE,),
                OnListener("listener-1", "agents"),
                id="uses_the_only_possible_answer",
            ),
            pytest.param(
                RequestedPlacement(k8s_namespace="agents-staging"),
                (TWO_NAMESPACES,),
                OnListener("listener-2", "agents-staging"),
                id="namespace_alone_picks_the_only_listener",
            ),
        ],
    )
    def test_resolves_to_a_placement(self, request_, listeners, expected):
        assert request_.among(listeners) == expected

    @pytest.mark.parametrize(
        ("request_", "listeners", "message"),
        [
            pytest.param(
                RequestedPlacement(listener_id="listener-1"),
                (),
                "no listeners",
                id="workspace_has_no_listeners",
            ),
            pytest.param(
                RequestedPlacement(),
                (ONE_NAMESPACE, TWO_NAMESPACES),
                "--listener-id",
                id="several_listeners_need_a_choice",
            ),
            pytest.param(
                RequestedPlacement(k8s_namespace="agents"),
                (ONE_NAMESPACE, TWO_NAMESPACES),
                "--listener-id",
                id="namespace_alone_is_ambiguous_with_several_listeners",
            ),
            pytest.param(
                RequestedPlacement(k8s_namespace="agents"),
                (),
                "no listeners",
                id="namespace_without_any_listener",
            ),
            pytest.param(
                RequestedPlacement(),
                (TWO_NAMESPACES,),
                "--k8s-namespace",
                id="several_namespaces_need_a_choice",
            ),
        ],
    )
    def test_refuses_and_names_the_choices(self, request_, listeners, message):
        with pytest.raises(click.UsageError, match=message):
            request_.among(listeners)

    def test_the_error_lists_every_listener_with_its_cluster_and_namespaces(self):
        with pytest.raises(click.UsageError) as error:
            RequestedPlacement().among((ONE_NAMESPACE, TWO_NAMESPACES))

        assert "listener-1" in error.value.message
        assert "prod-cluster" in error.value.message
        assert "agents-staging" in error.value.message

    @pytest.mark.parametrize(
        ("placement", "expected"),
        [
            pytest.param(Unplaced(), {}, id="unplaced_adds_nothing"),
            pytest.param(
                OnListener("listener-1", "agents"),
                {
                    "listener_id": "listener-1",
                    "listener_config": {"k8s_namespace": "agents"},
                },
                id="placed_carries_listener_and_namespace",
            ),
        ],
    )
    def test_source_config_matches_the_control_plane_shape(self, placement, expected):
        assert placement.source_config() == expected


def test_finding_a_deployment_by_name_narrows_the_search_for_every_server_version():
    seen: dict = {}

    def handler(req: httpx.Request) -> httpx.Response:
        seen["params"] = dict(req.url.params)
        return httpx.Response(
            200,
            json={"resources": [{"id": "dep-1", "name": "agent", "source": "github"}]},
        )

    client = HostBackendClient(
        "https://api.example.com", "key", transport=httpx.MockTransport(handler)
    )

    found = find_deployment_by_name(client, "agent")

    assert seen["params"] == {
        "name": "agent",
        "name_contains": "agent",
        "limit": "100",
    }
    assert found == ExistingDeployment("dep-1", "github")


def test_a_server_that_ignores_the_exact_name_filter_never_matches_another_deployment():
    client = HostBackendClient(
        "https://api.example.com",
        "key",
        transport=httpx.MockTransport(
            lambda req: httpx.Response(
                200,
                json={
                    "resources": [
                        {
                            "id": "dep-other",
                            "name": "another-teams-agent",
                            "source": "external_docker",
                        }
                    ]
                },
            )
        ),
    )

    assert find_deployment_by_name(client, "brand-new-agent") is None


def test_a_full_page_without_a_match_refuses_to_claim_the_name_is_free():
    page = [
        {"id": f"dep-{index}", "name": f"other-agent-{index}"} for index in range(100)
    ]
    client = HostBackendClient(
        "https://api.example.com",
        "key",
        transport=httpx.MockTransport(
            lambda req: httpx.Response(200, json={"resources": page})
        ),
    )

    with pytest.raises(click.ClickException, match="--deployment-id"):
        find_deployment_by_name(client, "brand-new-agent")


def test_a_partial_page_without_a_match_means_the_name_is_free():
    client = HostBackendClient(
        "https://api.example.com",
        "key",
        transport=httpx.MockTransport(
            lambda req: httpx.Response(
                200, json={"resources": [{"id": "dep-1", "name": "other"}]}
            )
        ),
    )

    assert find_deployment_by_name(client, "brand-new-agent") is None


@pytest.mark.parametrize(
    "resource",
    [
        pytest.param({"compute_id": "c"}, id="no_id"),
        pytest.param({"id": ""}, id="empty_id"),
    ],
)
def test_a_listener_without_an_id_is_refused(resource):
    with pytest.raises(HostBackendError, match="without an id"):
        Listener.from_resource(resource)


def test_a_deployment_id_with_listener_flags_is_refused_without_probing_docker(
    monkeypatch,
):
    def explode() -> tuple[bool, str | None]:
        raise AssertionError("docker must not be probed for an argv-only conflict")

    monkeypatch.setattr(deploy_mod, "can_build_locally", explode)

    with pytest.raises(click.UsageError, match="--deployment-id"):
        _select_source(
            push_to="registry.example.com/app",
            image=None,
            image_name=None,
            tag=None,
            remote_build_flag=None,
            placement=RequestedPlacement(listener_id="listener-1"),
            selector=ById("dep-1"),
        )


class TestPlacementOnAKnownListener:
    @pytest.mark.parametrize(
        ("request_", "listener", "expected"),
        [
            pytest.param(
                RequestedPlacement(listener_id="listener-1"),
                ONE_NAMESPACE,
                OnListener("listener-1", "agents"),
                id="the_only_namespace_is_used",
            ),
            pytest.param(
                RequestedPlacement(listener_id="listener-2", k8s_namespace="agents"),
                TWO_NAMESPACES,
                OnListener("listener-2", "agents"),
                id="the_chosen_namespace_is_used",
            ),
        ],
    )
    def test_places_on_the_listener(self, request_, listener, expected):
        assert request_.on(listener) == expected

    @pytest.mark.parametrize(
        ("request_", "listener", "message"),
        [
            pytest.param(
                RequestedPlacement(listener_id="listener-2"),
                TWO_NAMESPACES,
                "--k8s-namespace",
                id="several_namespaces_need_a_choice",
            ),
            pytest.param(
                RequestedPlacement(listener_id="listener-2", k8s_namespace="nope"),
                TWO_NAMESPACES,
                "does not serve namespace",
                id="unknown_namespace",
            ),
            pytest.param(
                RequestedPlacement(listener_id="listener-3"),
                NO_NAMESPACE,
                "serves no namespaces",
                id="listener_without_namespaces",
            ),
        ],
    )
    def test_refuses_and_names_the_namespaces(self, request_, listener, message):
        with pytest.raises(click.UsageError, match=message):
            request_.on(listener)
