import asyncio
import json
import uuid
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from functools import partial
from pathlib import Path

import click.exceptions
import httpx
import pytest
from click.testing import CliRunner, Result

import langgraph_cli.archive as archive_module
import langgraph_cli.deploy as deploy_module
from langgraph_cli.cli import cli
from langgraph_cli.host_backend import (
    CONTROL_PLANE_PATH,
    HostBackendClient,
    LangSmithApiClient,
)
from langgraph_cli.image_reference import ImageReference

CONTROL_PLANE_URL = "https://control-plane.example.com"
CLOUD_CONTROL_PLANE_URL = "https://api.host.langchain.com"
REGISTRY_URL = "https://registry.example.com/team"
PUSH_TOKEN = "push-token"
PUSHED_IMAGE = "registry.example.com/team/my-app:latest"
PUSHED_DIGEST = "registry.example.com/team/my-app@sha256:abc123"
PUSH_REPOSITORY = "registry.example.com/team/agent"
EXTERNAL_IMAGE = f"{PUSH_REPOSITORY}:latest"
EXTERNAL_DIGEST = f"{PUSH_REPOSITORY}@sha256:abc123"
LISTENER_ID = "11111111-1111-4111-8111-111111111111"
OTHER_LISTENER_ID = "22222222-2222-4222-8222-222222222222"
PAGE_TWO_LISTENER_ID = "33333333-3333-4333-8333-333333333333"
UNKNOWN_LISTENER_ID = "99999999-9999-4999-8999-999999999999"
LISTENER = {
    "id": LISTENER_ID,
    "compute_id": "prod-cluster",
    "compute_config": {"k8s_namespaces": ["agents"]},
}
OTHER_LISTENER = {
    "id": OTHER_LISTENER_ID,
    "compute_id": "other-cluster",
    "compute_config": {"k8s_namespaces": ["agents"]},
}
TWO_NAMESPACE_LISTENER = {
    "id": LISTENER_ID,
    "compute_id": "prod-cluster",
    "compute_config": {"k8s_namespaces": ["agents", "agents-staging"]},
}
CREATED_ID = "dep-created"
TRACKED_PACKAGES = ["langgraph:1.0.0"]
SIGNED_UPLOAD_URL = "https://storage.example.com/signed"
ARCHIVE = ("/tmp/src.tgz", 2048, "langgraph.json")
OBJECT_PATH = "tarballs/src.tgz"
PLATFORM_FORMAT = "{{.Os}}/{{.Architecture}}"
DIGESTS_FORMAT = "{{json .RepoDigests}}"
CONTEXT_HOST_FORMAT = "{{.Endpoints.docker.Host}}"
CONTEXT_HOST = "unix:///Users/dev/.docker/run/docker.sock"
NOT_A_CLI_DEPLOYMENT = (
    "push token is only available for 'internal_docker' source deployments"
)
LISTENER_REQUIRED = (
    "Source configuration error: 'source_config.listener_id' is required "
    f"for workspace with available listener IDs: ['{LISTENER_ID}']"
)
DATA_PLANE_URL = "https://langsmith.customer.example"
WORKSPACE_ID = "workspace-1"
BYOC_WORKSPACE = {"id": WORKSPACE_ID, "data_plane_url": DATA_PLANE_URL}
SAAS_WORKSPACE = {"id": "saas-workspace", "data_plane_url": None}
SCALE_TO_ZERO = "deployments_scale_to_zero_supported"
DEDICATED = "deployments_dedicated_supported"
CLI_BUILDS = "deployments_cli_builds_supported"
BYOC_FLAGS = {CLI_BUILDS: True, SCALE_TO_ZERO: True, DEDICATED: True}
REVISION_ID = "rev-1"
DISCOVER_WORKSPACES = "GET /api/v1/workspaces"
DATA_PLANE_INFO = "GET /api/v1/info"
LIST_DEPLOYMENTS = "GET /v2/deployments"
LIST_LISTENERS = "GET /v2/listeners"
CREATE_DEPLOYMENT = "POST /v2/deployments"


def _push_token(deployment_id: str) -> str:
    return f"POST /v2/deployments/{deployment_id}/push-token"


def _upload_url(deployment_id: str) -> str:
    return f"POST /v2/deployments/{deployment_id}/upload-url"


def _patch(deployment_id: str) -> str:
    return f"PATCH /v2/deployments/{deployment_id}"


def _get(deployment_id: str) -> str:
    return f"GET /v2/deployments/{deployment_id}"


def _looks_like_a_uuid(value: str) -> bool:
    try:
        uuid.UUID(value)
    except ValueError:
        return False
    return True


@dataclass
class ControlPlaneDouble:
    timeline: list[str]
    existing_deployments: list[dict] = field(default_factory=list)
    push_token_status: int = 200
    push_token_extra: dict = field(default_factory=dict)
    upload_url_extra: dict = field(default_factory=dict)
    create_error: str | None = None
    create_status: int = 400
    listeners: list[dict] = field(default_factory=list)
    listeners_by_id: dict[str, dict] = field(default_factory=dict)
    workspaces: list[dict] = field(default_factory=list)
    workspaces_status: int = 200
    requires_workspace: bool = False
    instance_flags: dict[str, bool] = field(default_factory=dict)
    bodies: dict[str, dict] = field(default_factory=dict)
    requests: list[httpx.Request] = field(default_factory=list)

    def handle(self, request: httpx.Request) -> httpx.Response:
        route = f"{request.method} {self._path(request)}"
        self.timeline.append(route)
        self.requests.append(request)
        if request.content:
            self.bodies[route] = json.loads(request.content)
        return self._respond(request)

    def host_backend_requests(self) -> list[httpx.Request]:
        return [r for r in self.requests if not r.url.path.startswith("/api/v1/")]

    @staticmethod
    def _path(request: httpx.Request) -> str:
        return request.url.path.removeprefix(CONTROL_PLANE_PATH)

    def _respond(self, request: httpx.Request) -> httpx.Response:
        method, path = request.method, self._path(request)
        if (method, path) == ("GET", "/api/v1/workspaces"):
            return httpx.Response(self.workspaces_status, json=self.workspaces)
        if (method, path) == ("GET", "/api/v1/info"):
            return httpx.Response(200, json={"instance_flags": self.instance_flags})
        if self.requires_workspace and "X-Tenant-ID" not in request.headers:
            return httpx.Response(
                403, json={"detail": "This API key requires workspace specification"}
            )
        if path.endswith("/revisions"):
            return httpx.Response(200, json={"resources": [{"id": REVISION_ID}]})
        if path.endswith("/build_logs"):
            return httpx.Response(200, json={"logs": []})
        if (method, path) == ("GET", "/v2/listeners"):
            offset = int(request.url.params.get("offset", 0))
            limit = int(request.url.params.get("limit", 20))
            return httpx.Response(
                200, json={"resources": self.listeners[offset : offset + limit]}
            )
        if method == "GET" and path.startswith("/v2/listeners/"):
            listener_id = path.rsplit("/", 1)[-1]
            if not _looks_like_a_uuid(listener_id):
                return httpx.Response(
                    422,
                    json={
                        "detail": [
                            {"type": "uuid_parsing", "loc": ["path", "listener_id"]}
                        ]
                    },
                )
            known = {listener["id"]: listener for listener in self.listeners}
            known.update(self.listeners_by_id)
            if listener_id not in known:
                return httpx.Response(
                    404, json={"detail": f"Listener ID {listener_id} not found."}
                )
            return httpx.Response(200, json=known[listener_id])
        if (method, path) == ("GET", "/v2/deployments"):
            name = request.url.params.get("name")
            return httpx.Response(
                200,
                json={
                    "resources": [
                        deployment
                        for deployment in self.existing_deployments
                        if name is None or deployment.get("name") == name
                    ]
                },
            )
        if (method, path) == ("POST", "/v2/deployments"):
            if self.create_error is not None:
                return httpx.Response(
                    self.create_status, json={"detail": self.create_error}
                )
            return httpx.Response(201, json={"id": CREATED_ID, "tenant_id": "tenant-1"})
        if path.endswith("/push-token"):
            if self.push_token_status != 200:
                return httpx.Response(self.push_token_status, text=NOT_A_CLI_DEPLOYMENT)
            return httpx.Response(
                200,
                json={
                    "token": PUSH_TOKEN,
                    "registry_url": REGISTRY_URL,
                    **self.push_token_extra,
                },
            )
        if path.endswith("/upload-url"):
            return httpx.Response(
                200,
                json={
                    "upload_url": SIGNED_UPLOAD_URL,
                    "object_path": OBJECT_PATH,
                    **self.upload_url_extra,
                },
            )
        if method == "PATCH":
            return httpx.Response(200, json={"tenant_id": "tenant-1"})
        if method == "GET":
            deployment_id = path.rsplit("/", 1)[-1]
            return httpx.Response(
                200,
                json=next(
                    d for d in self.existing_deployments if d["id"] == deployment_id
                ),
            )
        raise AssertionError(f"unexpected control plane call: {method} {path}")

    def install(self, monkeypatch: pytest.MonkeyPatch) -> None:
        transport = httpx.MockTransport(self.handle)
        for client_class in (HostBackendClient, LangSmithApiClient):
            monkeypatch.setattr(
                deploy_module,
                client_class.__name__,
                partial(client_class, transport=transport),
            )


@dataclass
class DockerCommand:
    args: tuple[str, ...]
    kwargs: dict


@dataclass
class DockerDouble:
    timeline: list[str]
    failing_pushes: int = 0
    context_lookup_fails: bool = False
    context_host: str = CONTEXT_HOST
    builds: list[dict] = field(default_factory=list)
    commands: list[DockerCommand] = field(default_factory=list)

    def verbs(self) -> list[str]:
        return [event for event in self.timeline if event.startswith("docker ")]

    def command(self, verb: str) -> DockerCommand:
        return next(c for c in self.commands if verb in c.args)

    def build_docker_image(
        self,
        runner: object,
        set_message: Callable[[str], None],
        config: Path,
        config_json: dict,
        base_image: str | None,
        api_version: str | None,
        pull: bool,
        tag: str,
        passthrough: tuple[str, ...] = (),
        install_command: str | None = None,
        build_command: str | None = None,
        docker_command: tuple[str, ...] | None = None,
        extra_flags: tuple[str, ...] = (),
        verbose: bool = True,
    ) -> None:
        self.timeline.append("docker build")
        self.builds.append(
            {
                "tag": tag,
                "docker_command": tuple(docker_command or ("docker", "build")),
                "extra_flags": tuple(extra_flags),
            }
        )

    async def subp_exec(
        self, *args: str, **kwargs: object
    ) -> tuple[str | None, str | None]:
        self.commands.append(DockerCommand(args=args, kwargs=kwargs))
        self.timeline.append(f"docker {self._verb(args)}")
        if "push" in args and self.failing_pushes > 0:
            self.failing_pushes -= 1
            raise click.exceptions.Exit(1)
        if CONTEXT_HOST_FORMAT in args:
            if self.context_lookup_fails:
                raise click.exceptions.Exit(1)
            return f"{self.context_host}\n", None
        if PLATFORM_FORMAT in args:
            return "linux/amd64\n", None
        if DIGESTS_FORMAT in args:
            repository = ImageReference.parse(args[-1]).repository
            return json.dumps([f"{repository}@sha256:abc123"]), None
        return None, None

    @staticmethod
    def _verb(args: tuple[str, ...]) -> str:
        if CONTEXT_HOST_FORMAT in args:
            return "inspect-context"
        if PLATFORM_FORMAT in args:
            return "inspect-platform"
        if DIGESTS_FORMAT in args:
            return "inspect-digest"
        return next(verb for verb in ("login", "tag", "push", "pull") if verb in args)


class _AsyncioRunner:
    def run(self, coro):
        return asyncio.run(coro)


@contextmanager
def _fake_runner() -> Iterator[_AsyncioRunner]:
    yield _AsyncioRunner()


@dataclass
class DeployProject:
    control_plane: ControlPlaneDouble
    docker: DockerDouble
    timeline: list[str]
    uploads: list[tuple[str, str, int, dict[str, str] | None]]

    def run(self, *args: str, host_url: str | None = CONTROL_PLANE_URL) -> Result:
        host_args = ("--host-url", host_url) if host_url else ()
        return CliRunner().invoke(
            cli,
            [
                "deploy",
                "--api-key",
                "test-key",
                *host_args,
                "--name",
                "my-app",
                "--no-input",
                "--no-wait",
                *args,
            ],
        )


@pytest.fixture
def deploy_project(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> DeployProject:
    (tmp_path / "langgraph.json").write_text(
        json.dumps({"graphs": {"agent": "agent.py:graph"}, "dependencies": ["."]})
    )
    monkeypatch.chdir(tmp_path)
    for name in (
        "LANGSMITH_TENANT_ID",
        "LANGSMITH_ENDPOINT",
        "LANGGRAPH_HOST_URL",
        "DOCKER_HOST",
    ):
        monkeypatch.delenv(name, raising=False)

    timeline: list[str] = []
    control_plane = ControlPlaneDouble(timeline)
    docker = DockerDouble(timeline)
    uploads: list[tuple[str, str, int, dict[str, str] | None]] = []

    @contextmanager
    def fake_create_archive(config_path: Path, config: dict) -> Iterator[tuple]:
        timeline.append("create_archive")
        yield ARCHIVE

    def fake_upload(
        signed_url: str,
        file_path: str,
        file_size: int,
        upload_headers: dict[str, str] | None,
    ) -> None:
        timeline.append("upload_archive")
        uploads.append((signed_url, file_path, file_size, upload_headers))

    monkeypatch.setattr(deploy_module, "_no_input", False)
    monkeypatch.setattr(deploy_module, "_emitter", None)
    control_plane.install(monkeypatch)
    monkeypatch.setattr(deploy_module, "build_docker_image", docker.build_docker_image)
    monkeypatch.setattr(deploy_module, "subp_exec", docker.subp_exec)
    monkeypatch.setattr(deploy_module, "Runner", _fake_runner)
    monkeypatch.setattr(deploy_module, "can_build_locally", lambda: (True, None))
    monkeypatch.setattr(
        deploy_module,
        "find_tracked_packages",
        lambda config, config_json: TRACKED_PACKAGES,
    )
    monkeypatch.setattr(deploy_module.platform, "machine", lambda: "x86_64")
    monkeypatch.setattr(archive_module, "create_archive", fake_create_archive)
    monkeypatch.setattr(deploy_module, "_upload_source_tarball", fake_upload)
    return DeployProject(control_plane, docker, timeline, uploads)


@pytest.mark.parametrize(
    ("command", "expected"),
    [
        ([], ["--agent-server-id", "--agent-server-type"]),
        (["logs"], ["--agent-server-id"]),
        (["delete"], ["AGENT_SERVER_ID"]),
        (["revisions", "list"], ["AGENT_SERVER_ID"]),
        (["metrics"], ["AGENT_SERVER_ID"]),
    ],
)
def test_deploy_help_uses_agent_server_names(command: list[str], expected: list[str]):
    result = CliRunner().invoke(cli, ["deploy", *command, "--help"])

    assert result.exit_code == 0, result.output
    assert all(name in result.output for name in expected)
    assert "Agent Server" in result.output
    assert "--deployment-" not in result.output
    assert "DEPLOYMENT_ID" not in result.output


@pytest.mark.parametrize("command", [[], ["logs"]])
@pytest.mark.parametrize("option", ["--agent-server-id", "--deployment-id"])
def test_missing_agent_server_id_uses_public_option_name(
    command: list[str], option: str
):
    result = CliRunner().invoke(cli, ["deploy", *command, option])

    assert result.exit_code == 2, result.output
    assert "Option '--agent-server-id' requires an argument" in result.output
    assert "--deployment-id" not in result.output


@pytest.mark.parametrize("option", ["--agent-server-type", "--deployment-type"])
@pytest.mark.parametrize("values", [[], ["invalid"]])
def test_invalid_agent_server_type_uses_public_option_name(
    option: str, values: list[str]
):
    result = CliRunner().invoke(cli, ["deploy", option, *values])

    assert result.exit_code == 2, result.output
    assert "'--agent-server-type'" in result.output
    assert "--deployment-type" not in result.output


@pytest.mark.parametrize("option", ["--agent-server-id", "--deployment-id"])
def test_deploy_logs_accepts_agent_server_id_aliases(
    deploy_project, mocker, option: str
):
    fetch_logs = mocker.patch.object(
        HostBackendClient, "get_deploy_logs", return_value={"logs": []}
    )

    result = CliRunner().invoke(
        cli, ["deploy", "logs", "--api-key", "test-key", option, "dep-existing"]
    )

    assert result.exit_code == 0, result.output
    fetch_logs.assert_called_once_with(
        "dep-existing", {"limit": 100, "order": "desc"}, None
    )


@pytest.mark.parametrize("build_mode", ["--no-remote", "--remote"])
@pytest.mark.parametrize(
    ("detail", "expects_dev_hint"),
    [
        ("Scale to zero deployments are not supported in AWS environments", True),
        ("No eligible cluster is currently available to host this deployment.", False),
    ],
)
def test_create_failure_preserves_backend_error_and_only_hints_for_aws_rejection(
    deploy_project: DeployProject, build_mode: str, detail: str, expects_dev_hint: bool
) -> None:
    deploy_project.control_plane.create_error = detail

    result = deploy_project.run(build_mode)

    assert result.exit_code == 1, result.output
    assert detail in result.output
    assert ("Use --agent-server-type dev instead." in result.output) is expects_dev_hint
    assert deploy_project.control_plane.bodies[CREATE_DEPLOYMENT]["source_config"] == {
        "deployment_type": "dev_zero"
    }
    assert deploy_project.timeline == [LIST_DEPLOYMENTS, CREATE_DEPLOYMENT]


@pytest.mark.parametrize("build_mode", ["--no-remote", "--remote"])
def test_aws_serverless_creation_succeeds_when_backend_accepts_it(
    deploy_project: DeployProject, build_mode: str
) -> None:
    result = deploy_project.run(
        build_mode, host_url="https://aws.api.host.langchain.com"
    )

    assert result.exit_code == 0, result.output
    assert deploy_project.control_plane.bodies[CREATE_DEPLOYMENT]["source_config"] == {
        "deployment_type": "dev_zero"
    }


def test_first_local_deploy_creates_then_builds_pushes_and_updates_in_order(
    deploy_project: DeployProject,
) -> None:
    result = deploy_project.run("--no-remote")

    assert result.exit_code == 0, result.output
    assert deploy_project.timeline == [
        LIST_DEPLOYMENTS,
        CREATE_DEPLOYMENT,
        "docker build",
        _push_token(CREATED_ID),
        "docker inspect-context",
        "docker login",
        "docker tag",
        "docker push",
        "docker inspect-digest",
        _patch(CREATED_ID),
    ]
    assert "Agent Server updated" in result.output


@pytest.mark.parametrize(
    ("deployment_type", "wire_type"),
    [
        ("serverless", "dev_zero"),
        ("dedicated", "prod"),
        ("dev", "dev"),
        ("prod", "prod"),
    ],
)
@pytest.mark.parametrize("build_mode", ["--no-remote", "--remote"])
@pytest.mark.parametrize("option", ["--agent-server-type", "--deployment-type"])
def test_deployment_type_aliases(
    deploy_project: DeployProject,
    deployment_type: str,
    wire_type: str,
    build_mode: str,
    option: str,
) -> None:
    result = deploy_project.run(build_mode, option, deployment_type)

    assert result.exit_code == 0, result.output
    assert deploy_project.control_plane.bodies[CREATE_DEPLOYMENT]["source_config"] == {
        "deployment_type": wire_type
    }


@pytest.mark.parametrize(
    ("build_mode", "source"),
    [("--no-remote", "internal_docker"), ("--remote", "internal_source")],
)
def test_first_deploy_defaults_to_serverless(
    deploy_project: DeployProject, build_mode: str, source: str
) -> None:
    result = deploy_project.run(build_mode)

    assert result.exit_code == 0, result.output
    assert deploy_project.control_plane.bodies[CREATE_DEPLOYMENT] == {
        "name": "my-app",
        "source": source,
        "source_config": {"deployment_type": "dev_zero"},
        "source_revision_config": {},
        "secrets": [],
    }


@pytest.mark.parametrize(
    ("machine", "expected_command", "expected_flags"),
    [
        pytest.param(
            "arm64",
            ("docker", "buildx", "build"),
            ("--platform", "linux/amd64", "--load", "--progress=plain"),
            id="apple_silicon_cross_builds_for_linux_amd64",
        ),
        pytest.param(
            "x86_64",
            ("docker", "build"),
            (),
            id="amd64_host_uses_plain_docker_build",
        ),
    ],
)
def test_local_build_targets_linux_amd64(
    deploy_project: DeployProject,
    monkeypatch: pytest.MonkeyPatch,
    machine: str,
    expected_command: tuple[str, ...],
    expected_flags: tuple[str, ...],
) -> None:
    monkeypatch.setattr(deploy_module.platform, "machine", lambda: machine)

    deploy_project.run("--no-remote")

    build = deploy_project.docker.builds[0]
    assert build["tag"].startswith("langgraph-deploy-tmp:")
    assert (build["docker_command"], build["extra_flags"]) == (
        expected_command,
        expected_flags,
    )


def test_local_deploy_logs_in_with_the_control_plane_push_token(
    deploy_project: DeployProject,
) -> None:
    deploy_project.run("--no-remote")

    login = deploy_project.docker.command("login")
    assert login.args[:2] == ("docker", "--config")
    assert login.args[-5:] == (
        "login",
        "-u",
        "oauth2accesstoken",
        "--password-stdin",
        "registry.example.com",
    )
    assert login.kwargs["input"] == f"{PUSH_TOKEN}\n"


def test_local_deploy_logs_in_and_pushes_through_the_active_docker_context(
    deploy_project: DeployProject,
) -> None:
    """The temporary config has no current context, so without the active
    context's endpoint Docker would push through /var/run/docker.sock."""
    result = deploy_project.run("--no-remote")

    assert result.exit_code == 0, result.output
    for verb in ("login", "push"):
        args = deploy_project.docker.command(verb).args
        assert args[3:5] == ("--host", CONTEXT_HOST), verb


def test_local_deploy_leaves_an_explicit_docker_host_to_docker(
    deploy_project: DeployProject, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("DOCKER_HOST", "tcp://127.0.0.1:2375")

    result = deploy_project.run("--no-remote")

    assert result.exit_code == 0, result.output
    for verb in ("login", "push"):
        assert "--host" not in deploy_project.docker.command(verb).args, verb


def test_local_deploy_keeps_dockers_default_when_the_context_lookup_fails(
    deploy_project: DeployProject,
) -> None:
    deploy_project.docker.context_lookup_fails = True

    result = deploy_project.run("--no-remote")

    assert result.exit_code == 0, result.output
    assert "--host" not in deploy_project.docker.command("push").args


def test_local_deploy_ignores_a_context_without_a_docker_endpoint(
    deploy_project: DeployProject,
) -> None:
    deploy_project.docker.context_host = "<no value>"

    result = deploy_project.run("--no-remote")

    assert result.exit_code == 0, result.output
    assert "--host" not in deploy_project.docker.command("push").args


def test_local_deploy_tells_the_control_plane_it_honors_registry_usernames(
    deploy_project: DeployProject,
) -> None:
    deploy_project.run("--no-remote")

    assert deploy_project.control_plane.bodies[_push_token(CREATED_ID)] == {
        "supports_registry_username": True
    }


ECR_REPOSITORY = "123456789012.dkr.ecr.us-east-2.amazonaws.com/lsd/agent-1"
ECR_PUSH_TOKEN_FIELDS = {"username": "AWS", "repository_url": ECR_REPOSITORY}


def test_local_deploy_logs_in_with_the_username_the_control_plane_returns(
    deploy_project: DeployProject,
) -> None:
    deploy_project.control_plane.push_token_extra = ECR_PUSH_TOKEN_FIELDS

    deploy_project.run("--no-remote")

    login = deploy_project.docker.command("login")
    assert login.args[-5:] == (
        "login",
        "-u",
        "AWS",
        "--password-stdin",
        "123456789012.dkr.ecr.us-east-2.amazonaws.com",
    )


def test_local_deploy_pushes_to_exactly_the_returned_repository(
    deploy_project: DeployProject,
) -> None:
    deploy_project.control_plane.push_token_extra = ECR_PUSH_TOKEN_FIELDS

    deploy_project.run("--no-remote")

    tag = deploy_project.docker.command("tag")
    assert tag.args[-1] == f"{ECR_REPOSITORY}:latest"


def test_local_deploy_tags_the_build_into_the_token_registry(
    deploy_project: DeployProject,
) -> None:
    deploy_project.run("--no-remote")

    built_tag = deploy_project.docker.builds[0]["tag"]
    assert deploy_project.docker.command("tag").args == (
        "docker",
        "tag",
        built_tag,
        PUSHED_IMAGE,
    )
    assert deploy_project.docker.command("push").args[-1] == PUSHED_IMAGE


def test_local_deploy_records_the_pushed_digest_and_tracked_packages(
    deploy_project: DeployProject,
) -> None:
    deploy_project.run("--no-remote")

    assert deploy_project.control_plane.bodies[_patch(CREATED_ID)] == {
        "revision_source": "internal_docker",
        "source_revision_config": {"image_uri": PUSHED_DIGEST},
        "secrets": [],
        "tracked_packages": TRACKED_PACKAGES,
    }


def test_status_link_points_at_the_langsmith_dashboard(
    deploy_project: DeployProject,
) -> None:
    result = deploy_project.run("--no-remote")

    assert (
        "View status: https://smith.langchain.com/o/tenant-1/host/deployments/dep-created"
        in result.output
    )


def test_prebuilt_image_is_validated_and_pushed_without_a_build(
    deploy_project: DeployProject,
) -> None:
    result = deploy_project.run("--image", "local/app:dev")

    assert result.exit_code == 0, result.output
    assert deploy_project.docker.builds == []
    assert deploy_project.docker.verbs() == [
        "docker inspect-platform",
        "docker inspect-context",
        "docker login",
        "docker tag",
        "docker push",
        "docker inspect-digest",
    ]
    assert deploy_project.docker.command("tag").args[2:] == (
        "local/app:dev",
        PUSHED_IMAGE,
    )


def test_push_is_retried_until_the_third_attempt(
    deploy_project: DeployProject,
) -> None:
    deploy_project.docker.failing_pushes = 2

    result = deploy_project.run("--no-remote")

    assert result.exit_code == 0, result.output
    assert deploy_project.docker.verbs().count("docker push") == 3


def test_three_failed_pushes_abort_before_the_deployment_is_updated(
    deploy_project: DeployProject,
) -> None:
    deploy_project.docker.failing_pushes = 3

    result = deploy_project.run("--no-remote")

    assert result.exit_code != 0
    assert _patch(CREATED_ID) not in deploy_project.timeline


def test_existing_deployment_matched_by_exact_name_is_updated_not_created(
    deploy_project: DeployProject,
) -> None:
    deploy_project.control_plane.existing_deployments = [
        {"id": "dep-other", "name": "my-app-2"},
        {"id": "dep-existing", "name": "my-app"},
    ]

    deploy_project.run("--no-remote")

    assert CREATE_DEPLOYMENT not in deploy_project.timeline
    assert _patch("dep-existing") in deploy_project.timeline


def test_deployment_not_created_by_the_cli_gets_an_actionable_error(
    deploy_project: DeployProject,
) -> None:
    deploy_project.control_plane.existing_deployments = [
        {"id": "dep-ui", "name": "my-app"}
    ]
    deploy_project.control_plane.push_token_status = 400

    result = deploy_project.run("--no-remote")

    assert result.exit_code != 0
    assert "was not created by 'langgraph deploy'" in result.output
    assert "docker login" not in deploy_project.timeline


def test_remote_build_creates_an_internal_source_deployment_and_uploads_the_archive(
    deploy_project: DeployProject,
) -> None:
    upload_headers = {"Content-Type": "application/gzip", "x-ms-blob-type": "BlockBlob"}
    deploy_project.control_plane.upload_url_extra = {"upload_headers": upload_headers}

    result = deploy_project.run("--remote", "--install-command", "yarn install")

    assert result.exit_code == 0, result.output
    assert deploy_project.timeline == [
        LIST_DEPLOYMENTS,
        CREATE_DEPLOYMENT,
        "create_archive",
        _upload_url(CREATED_ID),
        "upload_archive",
        _patch(CREATED_ID),
    ]
    assert deploy_project.control_plane.bodies[CREATE_DEPLOYMENT]["source"] == (
        "internal_source"
    )
    assert deploy_project.control_plane.bodies[_upload_url(CREATED_ID)] == {
        "supports_upload_headers": True
    }
    assert deploy_project.uploads == [
        (SIGNED_UPLOAD_URL, ARCHIVE[0], ARCHIVE[1], upload_headers)
    ]
    assert deploy_project.control_plane.bodies[_patch(CREATED_ID)] == {
        "revision_source": "internal_source",
        "source_revision_config": {
            "source_tarball_path": OBJECT_PATH,
            "langgraph_config_path": ARCHIVE[2],
        },
        "source_config": {"install_command": "yarn install"},
        "secrets": [],
        "tracked_packages": TRACKED_PACKAGES,
    }
    assert "Build triggered" in result.output


def test_push_to_builds_pushes_then_creates_an_external_deployment(
    deploy_project: DeployProject,
) -> None:
    result = deploy_project.run("--push-to", PUSH_REPOSITORY)

    assert result.exit_code == 0, result.output
    assert deploy_project.timeline == [
        LIST_DEPLOYMENTS,
        "docker build",
        "docker push",
        "docker inspect-digest",
        CREATE_DEPLOYMENT,
    ]
    assert deploy_project.control_plane.bodies[CREATE_DEPLOYMENT] == {
        "name": "my-app",
        "source": "external_docker",
        "source_config": {"resource_spec": {}},
        "source_revision_config": {"image_uri": EXTERNAL_DIGEST},
        "secrets": [],
    }
    assert "Agent Server created" in result.output


def test_push_to_builds_directly_with_the_push_reference(
    deploy_project: DeployProject,
) -> None:
    result = deploy_project.run("--push-to", PUSH_REPOSITORY)

    assert result.exit_code == 0, result.output
    assert deploy_project.docker.builds[0]["tag"] == EXTERNAL_IMAGE
    assert deploy_project.docker.command("push").args == (
        "docker",
        "push",
        EXTERNAL_IMAGE,
    )


def test_push_to_composes_with_the_tag_flag(deploy_project: DeployProject) -> None:
    result = deploy_project.run("--push-to", PUSH_REPOSITORY, "--tag", "v1")

    assert result.exit_code == 0, result.output
    assert deploy_project.docker.command("push").args[-1] == f"{PUSH_REPOSITORY}:v1"


def test_push_to_with_a_failing_push_creates_no_deployment(
    deploy_project: DeployProject,
) -> None:
    deploy_project.docker.failing_pushes = 3

    result = deploy_project.run("--push-to", PUSH_REPOSITORY)

    assert result.exit_code != 0
    assert CREATE_DEPLOYMENT not in deploy_project.timeline


def test_verbose_never_echoes_the_push_token(deploy_project: DeployProject) -> None:
    result = deploy_project.run("--no-remote", "--verbose")

    assert result.exit_code == 0, result.output
    assert deploy_project.docker.command("login").kwargs["verbose"] is False


def test_push_to_retags_a_prebuilt_image_instead_of_building(
    deploy_project: DeployProject,
) -> None:
    result = deploy_project.run(
        "--image", "local/app:dev", "--push-to", PUSH_REPOSITORY
    )

    assert result.exit_code == 0, result.output
    assert deploy_project.docker.builds == []
    assert deploy_project.docker.verbs() == [
        "docker inspect-platform",
        "docker tag",
        "docker push",
        "docker inspect-digest",
    ]
    assert deploy_project.docker.command("tag").args == (
        "docker",
        "tag",
        "local/app:dev",
        EXTERNAL_IMAGE,
    )


def test_push_to_updates_an_existing_external_deployment_with_the_new_image(
    deploy_project: DeployProject,
) -> None:
    deploy_project.control_plane.existing_deployments = [
        {"id": "dep-ext", "name": "my-app", "source": "external_docker"}
    ]

    result = deploy_project.run("--push-to", PUSH_REPOSITORY)

    assert result.exit_code == 0, result.output
    assert deploy_project.timeline == [
        LIST_DEPLOYMENTS,
        "docker build",
        "docker push",
        "docker inspect-digest",
        _patch("dep-ext"),
    ]
    assert deploy_project.control_plane.bodies[_patch("dep-ext")] == {
        "source_revision_config": {"image_uri": EXTERNAL_DIGEST},
        "secrets": [],
        "tracked_packages": TRACKED_PACKAGES,
    }


def test_push_to_rejects_a_non_external_deployment_before_any_docker_work(
    deploy_project: DeployProject,
) -> None:
    deploy_project.control_plane.existing_deployments = [
        {"id": "dep-cli", "name": "my-app", "source": "internal_docker"}
    ]

    result = deploy_project.run("--push-to", PUSH_REPOSITORY)

    assert result.exit_code != 0
    assert "cannot be updated with --push-to" in result.output
    assert deploy_project.docker.verbs() == []


@pytest.mark.parametrize("option", ["--agent-server-id", "--deployment-id"])
@pytest.mark.parametrize("equals", [False, True])
def test_push_to_with_deployment_id_fetches_the_deployment_once(
    deploy_project: DeployProject, option: str, equals: bool
) -> None:
    deploy_project.control_plane.existing_deployments = [
        {"id": "dep-ext", "name": "another-name", "source": "external_docker"}
    ]

    id_args = [f"{option}=dep-ext"] if equals else [option, "dep-ext"]
    result = deploy_project.run(*id_args, "--push-to", PUSH_REPOSITORY)

    assert result.exit_code == 0, result.output
    assert deploy_project.timeline == [
        _get("dep-ext"),
        "docker build",
        "docker push",
        "docker inspect-digest",
        _patch("dep-ext"),
    ]


def test_image_uri_creates_an_external_deployment_without_any_docker_work(
    deploy_project: DeployProject,
) -> None:
    result = deploy_project.run("--image-uri", EXTERNAL_DIGEST)

    assert result.exit_code == 0, result.output
    assert deploy_project.docker.verbs() == []
    assert deploy_project.timeline == [LIST_DEPLOYMENTS, CREATE_DEPLOYMENT]
    assert deploy_project.control_plane.bodies[CREATE_DEPLOYMENT][
        "source_revision_config"
    ] == {"image_uri": EXTERNAL_DIGEST}


def test_image_uri_updates_an_existing_external_deployment_without_any_docker_work(
    deploy_project: DeployProject,
) -> None:
    deploy_project.control_plane.existing_deployments = [
        {"id": "dep-ext", "name": "my-app", "source": "external_docker"}
    ]

    result = deploy_project.run("--image-uri", EXTERNAL_DIGEST)

    assert result.exit_code == 0, result.output
    assert deploy_project.docker.verbs() == []
    assert deploy_project.timeline == [LIST_DEPLOYMENTS, _patch("dep-ext")]
    assert deploy_project.control_plane.bodies[_patch("dep-ext")] == {
        "source_revision_config": {"image_uri": EXTERNAL_DIGEST},
        "secrets": [],
        "tracked_packages": TRACKED_PACKAGES,
    }


def test_image_uri_places_a_new_deployment_on_the_only_listener(
    deploy_project: DeployProject,
) -> None:
    deploy_project.control_plane.listeners = [LISTENER]

    result = deploy_project.run(
        "--image-uri", EXTERNAL_DIGEST, host_url=CLOUD_CONTROL_PLANE_URL
    )

    assert result.exit_code == 0, result.output
    assert deploy_project.docker.verbs() == []
    assert deploy_project.control_plane.bodies[CREATE_DEPLOYMENT]["source_config"] == {
        "resource_spec": {},
        "listener_id": LISTENER_ID,
        "listener_config": {"k8s_namespace": "agents"},
    }


def test_image_uri_rejects_a_non_external_deployment_before_any_docker_work(
    deploy_project: DeployProject,
) -> None:
    deploy_project.control_plane.existing_deployments = [
        {"id": "dep-cli", "name": "my-app", "source": "internal_docker"}
    ]

    result = deploy_project.run("--image-uri", EXTERNAL_DIGEST)

    assert result.exit_code != 0
    assert "cannot be updated with --push-to or --image-uri" in result.output
    assert deploy_project.docker.verbs() == []


@pytest.mark.parametrize(
    ("args", "message"),
    [
        pytest.param(
            ("--image-uri", EXTERNAL_DIGEST, "--push-to", PUSH_REPOSITORY),
            "--image-uri cannot be combined with --push-to.",
            id="with_push_to",
        ),
        pytest.param(
            ("--image-uri", EXTERNAL_DIGEST, "--image", "local/app:dev"),
            "--image-uri cannot be combined with --image.",
            id="with_image",
        ),
        pytest.param(
            ("--image-uri", EXTERNAL_DIGEST, "--tag", "v2"),
            "--image-uri cannot be combined with --tag.",
            id="with_tag",
        ),
        pytest.param(
            ("--image-uri", EXTERNAL_DIGEST, "--remote"),
            "--image-uri cannot be combined with --remote.",
            id="with_remote",
        ),
    ],
)
def test_image_uri_conflicting_flags_are_rejected_before_any_docker_work(
    deploy_project: DeployProject, args: tuple[str, ...], message: str
) -> None:
    result = deploy_project.run(*args)

    assert result.exit_code != 0
    assert message in result.output
    assert deploy_project.docker.verbs() == []
    assert deploy_project.timeline == []


def test_invalid_tag_fails_before_any_control_plane_call(
    deploy_project: DeployProject,
) -> None:
    result = deploy_project.run("--no-remote", "--tag", "not a tag")

    assert result.exit_code != 0
    assert "Image tag may only contain" in result.output
    assert deploy_project.timeline == []


def test_push_to_places_a_new_deployment_on_the_only_listener(
    deploy_project: DeployProject,
) -> None:
    deploy_project.control_plane.listeners = [LISTENER]

    result = deploy_project.run(
        "--push-to", PUSH_REPOSITORY, host_url=CLOUD_CONTROL_PLANE_URL
    )

    assert result.exit_code == 0, result.output
    assert deploy_project.timeline == [
        LIST_DEPLOYMENTS,
        LIST_LISTENERS,
        "docker build",
        "docker push",
        "docker inspect-digest",
        CREATE_DEPLOYMENT,
    ]
    assert deploy_project.control_plane.bodies[CREATE_DEPLOYMENT]["source_config"] == {
        "resource_spec": {},
        "listener_id": LISTENER_ID,
        "listener_config": {"k8s_namespace": "agents"},
    }
    assert f"Deploying through listener {LISTENER_ID} in namespace agents" in (
        result.output
    )


def test_push_to_places_a_new_deployment_on_the_chosen_listener(
    deploy_project: DeployProject,
) -> None:
    deploy_project.control_plane.listeners = [LISTENER, OTHER_LISTENER]

    result = deploy_project.run(
        "--push-to",
        PUSH_REPOSITORY,
        "--listener-id",
        OTHER_LISTENER_ID,
        "--k8s-namespace",
        "agents",
        host_url=CLOUD_CONTROL_PLANE_URL,
    )

    assert result.exit_code == 0, result.output
    assert deploy_project.control_plane.bodies[CREATE_DEPLOYMENT]["source_config"] == {
        "resource_spec": {},
        "listener_id": OTHER_LISTENER_ID,
        "listener_config": {"k8s_namespace": "agents"},
    }


@pytest.mark.parametrize(
    ("listeners", "args", "message"),
    [
        pytest.param(
            [LISTENER, OTHER_LISTENER], (), "--listener-id", id="two_listeners"
        ),
        pytest.param(
            [TWO_NAMESPACE_LISTENER], (), "--k8s-namespace", id="two_namespaces"
        ),
        pytest.param(
            [LISTENER],
            ("--k8s-namespace", "nope"),
            "does not serve namespace",
            id="unknown_namespace",
        ),
    ],
)
def test_push_to_refuses_an_unresolved_placement_before_any_docker_work(
    deploy_project: DeployProject, listeners, args, message
) -> None:
    deploy_project.control_plane.listeners = listeners

    result = deploy_project.run(
        "--push-to", PUSH_REPOSITORY, *args, host_url=CLOUD_CONTROL_PLANE_URL
    )

    assert result.exit_code != 0
    assert message in result.output
    assert deploy_project.docker.verbs() == []
    assert CREATE_DEPLOYMENT not in deploy_project.timeline


def test_self_hosted_control_plane_keeps_its_default_placement(
    deploy_project: DeployProject,
) -> None:
    deploy_project.control_plane.listeners = [LISTENER]

    result = deploy_project.run("--push-to", PUSH_REPOSITORY)

    assert result.exit_code == 0, result.output
    assert deploy_project.control_plane.bodies[CREATE_DEPLOYMENT]["source_config"] == {
        "resource_spec": {}
    }


def test_self_hosted_control_plane_places_when_asked(
    deploy_project: DeployProject,
) -> None:
    deploy_project.control_plane.listeners = [LISTENER]

    result = deploy_project.run(
        "--push-to", PUSH_REPOSITORY, "--listener-id", LISTENER_ID
    )

    assert result.exit_code == 0, result.output
    assert deploy_project.control_plane.bodies[CREATE_DEPLOYMENT]["source_config"] == {
        "resource_spec": {},
        "listener_id": LISTENER_ID,
        "listener_config": {"k8s_namespace": "agents"},
    }


def test_image_pull_secret_is_included_in_resource_spec_on_create(
    deploy_project: DeployProject,
) -> None:
    result = deploy_project.run(
        "--push-to",
        PUSH_REPOSITORY,
        "--image-pull-secret",
        "regcred",
        "--image-pull-secret",
        "other-secret",
    )

    assert result.exit_code == 0, result.output
    assert deploy_project.control_plane.bodies[CREATE_DEPLOYMENT]["source_config"] == {
        "resource_spec": {
            "image_pull_secrets": [
                {"name": "regcred"},
                {"name": "other-secret"},
            ],
        },
    }


def test_image_pull_secret_works_with_image_uri_too(
    deploy_project: DeployProject,
) -> None:
    result = deploy_project.run(
        "--image-uri", EXTERNAL_DIGEST, "--image-pull-secret", "regcred"
    )

    assert result.exit_code == 0, result.output
    assert deploy_project.control_plane.bodies[CREATE_DEPLOYMENT]["source_config"] == {
        "resource_spec": {"image_pull_secrets": [{"name": "regcred"}]},
    }


def test_image_pull_secret_without_push_to_or_image_uri_is_refused(
    deploy_project: DeployProject,
) -> None:
    result = deploy_project.run("--image-pull-secret", "regcred")

    assert result.exit_code != 0
    assert "only apply when creating an Agent Server with --push-to" in result.output
    assert deploy_project.timeline == []


def test_an_invalid_image_pull_secret_name_is_rejected_before_any_docker_work(
    deploy_project: DeployProject,
) -> None:
    result = deploy_project.run(
        "--push-to", PUSH_REPOSITORY, "--image-pull-secret", "Not_Valid!"
    )

    assert result.exit_code != 0
    assert "not a valid Kubernetes" in result.output
    assert deploy_project.docker.verbs() == []
    assert deploy_project.timeline == []


def test_image_pull_secret_cannot_update_an_existing_deployment(
    deploy_project: DeployProject,
) -> None:
    result = deploy_project.run(
        "--deployment-id",
        "dep-1",
        "--push-to",
        PUSH_REPOSITORY,
        "--image-pull-secret",
        "regcred",
    )

    assert result.exit_code != 0
    assert "--image-pull-secret" in result.output
    assert "dep-1" in result.output
    assert deploy_project.timeline == []


def test_image_pull_secret_cannot_update_a_deployment_found_by_name(
    deploy_project: DeployProject,
) -> None:
    deploy_project.control_plane.existing_deployments = [
        {"id": "dep-ext", "name": "my-app", "source": "external_docker"}
    ]

    result = deploy_project.run(
        "--image-uri", EXTERNAL_DIGEST, "--image-pull-secret", "regcred"
    )

    assert result.exit_code != 0
    assert "--image-pull-secret" in result.output
    assert deploy_project.docker.verbs() == []
    assert _patch("dep-ext") not in deploy_project.timeline


def test_updating_a_deployment_never_looks_up_listeners(
    deploy_project: DeployProject,
) -> None:
    deploy_project.control_plane.listeners = [LISTENER]
    deploy_project.control_plane.existing_deployments = [
        {"id": "dep-ext", "name": "my-app", "source": "external_docker"}
    ]

    result = deploy_project.run(
        "--push-to", PUSH_REPOSITORY, host_url=CLOUD_CONTROL_PLANE_URL
    )

    assert result.exit_code == 0, result.output
    assert LIST_LISTENERS not in deploy_project.timeline


def test_listener_flags_are_refused_for_a_deployment_id_without_any_call(
    deploy_project: DeployProject,
) -> None:
    result = deploy_project.run(
        "--push-to",
        PUSH_REPOSITORY,
        "--deployment-id",
        "dep-ext",
        "--k8s-namespace",
        "agents",
        host_url=CLOUD_CONTROL_PLANE_URL,
    )

    assert result.exit_code != 0
    assert "fixed when an Agent Server is created" in result.output
    assert deploy_project.timeline == []


def test_listener_flags_are_refused_on_an_existing_deployment(
    deploy_project: DeployProject,
) -> None:
    deploy_project.control_plane.listeners = [LISTENER]
    deploy_project.control_plane.existing_deployments = [
        {"id": "dep-ext", "name": "my-app", "source": "external_docker"}
    ]

    result = deploy_project.run(
        "--push-to",
        PUSH_REPOSITORY,
        "--listener-id",
        LISTENER_ID,
        host_url=CLOUD_CONTROL_PLANE_URL,
    )

    assert result.exit_code != 0
    assert "fixed when an Agent Server is created" in result.output
    assert deploy_project.docker.verbs() == []


def test_a_deployment_without_a_listener_announces_nothing(
    deploy_project: DeployProject,
) -> None:
    result = deploy_project.run("--push-to", PUSH_REPOSITORY)

    assert result.exit_code == 0, result.output
    assert "listener" not in result.output


def test_a_self_hosted_create_without_flags_never_looks_up_listeners(
    deploy_project: DeployProject,
) -> None:
    deploy_project.control_plane.listeners = [LISTENER]

    result = deploy_project.run("--push-to", PUSH_REPOSITORY)

    assert result.exit_code == 0, result.output
    assert LIST_LISTENERS not in deploy_project.timeline


def test_a_control_plane_that_demands_a_listener_names_the_flags(
    deploy_project: DeployProject,
) -> None:
    deploy_project.control_plane.create_error = LISTENER_REQUIRED

    result = deploy_project.run("--push-to", PUSH_REPOSITORY)

    assert result.exit_code != 0
    assert "--listener-id" in result.output
    assert "--k8s-namespace" in result.output
    assert LISTENER_ID in result.output
    assert "{" not in result.output
    assert "POST /v2/deployments failed" not in result.output


def test_listener_flags_without_push_to_make_no_call_at_all(
    deploy_project: DeployProject,
) -> None:
    result = deploy_project.run("--listener-id", LISTENER_ID)

    assert result.exit_code != 0
    assert "--push-to" in result.output
    assert deploy_project.timeline == []


def test_listeners_beyond_the_first_page_are_counted(
    deploy_project: DeployProject,
) -> None:
    deploy_project.control_plane.listeners = [
        {
            "id": str(uuid.UUID(int=index)),
            "compute_id": "cluster",
            "compute_config": {"k8s_namespaces": ["agents"]},
        }
        for index in range(101)
    ]

    result = deploy_project.run(
        "--push-to", PUSH_REPOSITORY, host_url=CLOUD_CONTROL_PLANE_URL
    )

    assert result.exit_code != 0
    assert "... and 91 more" in result.output
    assert "first 100" not in result.output


def test_a_managed_build_in_a_listener_workspace_points_at_push_to(
    deploy_project: DeployProject,
) -> None:
    deploy_project.control_plane.create_error = LISTENER_REQUIRED

    result = deploy_project.run("--no-remote")

    assert result.exit_code != 0
    assert "--push-to" in result.output
    assert deploy_project.docker.verbs() == []


@pytest.mark.parametrize(
    "args",
    [
        pytest.param(("--no-remote",), id="managed_build"),
        pytest.param(("--push-to", PUSH_REPOSITORY), id="push_to"),
    ],
)
def test_a_listener_requirement_links_the_listener_docs(
    deploy_project: DeployProject, args: tuple[str, ...]
) -> None:
    deploy_project.control_plane.create_error = LISTENER_REQUIRED

    result = deploy_project.run(*args)

    assert result.exit_code != 0
    assert "https://docs.langchain.com/langsmith/control-plane#listeners" in (
        result.output
    )


def test_a_managed_control_plane_without_listeners_creates_as_before(
    deploy_project: DeployProject,
) -> None:
    result = deploy_project.run(
        "--push-to", PUSH_REPOSITORY, host_url=CLOUD_CONTROL_PLANE_URL
    )

    assert result.exit_code == 0, result.output
    assert deploy_project.control_plane.bodies[CREATE_DEPLOYMENT]["source_config"] == {
        "resource_spec": {}
    }
    assert deploy_project.timeline.count(LIST_LISTENERS) == 1


def test_a_listener_without_an_id_is_reported_rather_than_ignored(
    deploy_project: DeployProject,
) -> None:
    deploy_project.control_plane.listeners = [
        {"compute_id": "broken", "compute_config": {"k8s_namespaces": ["agents"]}},
        LISTENER,
    ]

    result = deploy_project.run(
        "--push-to", PUSH_REPOSITORY, host_url=CLOUD_CONTROL_PLANE_URL
    )

    assert result.exit_code != 0
    assert "without an id" in result.output
    assert deploy_project.docker.verbs() == []


def _listener_route(listener_id: str) -> str:
    return f"GET /v2/listeners/{listener_id}"


def test_an_explicit_listener_is_fetched_by_id_not_searched(
    deploy_project: DeployProject,
) -> None:
    deploy_project.control_plane.listeners = [LISTENER, OTHER_LISTENER]

    result = deploy_project.run(
        "--push-to",
        PUSH_REPOSITORY,
        "--listener-id",
        OTHER_LISTENER_ID,
        host_url=CLOUD_CONTROL_PLANE_URL,
    )

    assert result.exit_code == 0, result.output
    assert _listener_route(OTHER_LISTENER_ID) in deploy_project.timeline
    assert LIST_LISTENERS not in deploy_project.timeline
    assert deploy_project.control_plane.bodies[CREATE_DEPLOYMENT]["source_config"] == {
        "resource_spec": {},
        "listener_id": OTHER_LISTENER_ID,
        "listener_config": {"k8s_namespace": "agents"},
    }


def test_an_explicit_listener_beyond_the_first_page_still_works(
    deploy_project: DeployProject,
) -> None:
    deploy_project.control_plane.listeners = [
        {
            "id": str(uuid.UUID(int=index)),
            "compute_id": "cluster",
            "compute_config": {"k8s_namespaces": ["agents"]},
        }
        for index in range(100)
    ]
    deploy_project.control_plane.listeners_by_id = {
        PAGE_TWO_LISTENER_ID: {
            "id": PAGE_TWO_LISTENER_ID,
            "compute_id": "far-cluster",
            "compute_config": {"k8s_namespaces": ["agents"]},
        }
    }

    result = deploy_project.run(
        "--push-to",
        PUSH_REPOSITORY,
        "--listener-id",
        PAGE_TWO_LISTENER_ID,
        host_url=CLOUD_CONTROL_PLANE_URL,
    )

    assert result.exit_code == 0, result.output
    assert deploy_project.control_plane.bodies[CREATE_DEPLOYMENT]["source_config"] == {
        "resource_spec": {},
        "listener_id": PAGE_TWO_LISTENER_ID,
        "listener_config": {"k8s_namespace": "agents"},
    }


def test_an_unknown_listener_names_the_ones_that_exist(
    deploy_project: DeployProject,
) -> None:
    deploy_project.control_plane.listeners = [LISTENER]

    result = deploy_project.run(
        "--push-to",
        PUSH_REPOSITORY,
        "--listener-id",
        UNKNOWN_LISTENER_ID,
        host_url=CLOUD_CONTROL_PLANE_URL,
    )

    assert result.exit_code != 0
    assert "was not found" in result.output
    assert LISTENER_ID in result.output
    assert "prod-cluster" in result.output
    assert deploy_project.docker.verbs() == []


def test_an_explicit_listener_in_a_workspace_without_any_is_refused(
    deploy_project: DeployProject,
) -> None:
    result = deploy_project.run(
        "--push-to",
        PUSH_REPOSITORY,
        "--listener-id",
        LISTENER_ID,
        host_url=CLOUD_CONTROL_PLANE_URL,
    )

    assert result.exit_code != 0
    assert "no listeners" in result.output
    assert deploy_project.docker.verbs() == []


def test_a_listener_id_that_is_not_an_identifier_still_names_the_real_ones(
    deploy_project: DeployProject,
) -> None:
    deploy_project.control_plane.listeners = [LISTENER]

    result = deploy_project.run(
        "--push-to",
        PUSH_REPOSITORY,
        "--listener-id",
        "not-a-listener",
        host_url=CLOUD_CONTROL_PLANE_URL,
    )

    assert result.exit_code != 0
    assert "was not found" in result.output
    assert LISTENER_ID in result.output
    assert "uuid_parsing" not in result.output


@pytest.fixture
def byoc_project(deploy_project: DeployProject) -> DeployProject:
    deploy_project.control_plane.workspaces = [BYOC_WORKSPACE]
    deploy_project.control_plane.instance_flags = dict(BYOC_FLAGS)
    return deploy_project


@pytest.mark.parametrize("build_mode", ["--no-remote", "--remote"])
def test_byoc_deploy_runs_on_the_discovered_data_plane_and_workspace(
    byoc_project: DeployProject, build_mode: str
) -> None:
    result = byoc_project.run(build_mode, host_url=None)

    assert result.exit_code == 0, result.output
    assert byoc_project.timeline[:4] == [
        DISCOVER_WORKSPACES,
        LIST_DEPLOYMENTS,
        DATA_PLANE_INFO,
        CREATE_DEPLOYMENT,
    ]
    host_backend_requests = byoc_project.control_plane.host_backend_requests()
    assert host_backend_requests
    assert all(
        str(r.url).startswith(f"{DATA_PLANE_URL}/api-host/v2/")
        and r.headers["X-Tenant-ID"] == WORKSPACE_ID
        for r in host_backend_requests
    )


@pytest.mark.parametrize(
    "langsmith_endpoint",
    [
        pytest.param(None, id="default_endpoint"),
        pytest.param(DATA_PLANE_URL, id="endpoint_is_the_data_plane"),
    ],
)
def test_byoc_status_link_points_at_langsmith_not_the_data_plane(
    byoc_project: DeployProject,
    monkeypatch: pytest.MonkeyPatch,
    langsmith_endpoint: str | None,
) -> None:
    if langsmith_endpoint:
        monkeypatch.setenv("LANGSMITH_ENDPOINT", langsmith_endpoint)

    result = byoc_project.run("--no-remote", host_url=None)

    assert (
        "View status: https://smith.langchain.com/o/tenant-1/host/deployments/dep-created"
        in result.output
    )


@pytest.mark.parametrize(
    ("instance_flags", "type_args", "source_config"),
    [
        pytest.param(
            BYOC_FLAGS,
            [],
            {"resource_spec": {}, "deployment_type": "dev_zero"},
            id="default_is_serverless_with_scale_to_zero",
        ),
        pytest.param(
            {CLI_BUILDS: True},
            [],
            {"resource_spec": {}},
            id="default_is_dev_without_scale_to_zero",
        ),
        pytest.param(
            BYOC_FLAGS,
            ["--agent-server-type", "dev"],
            {"resource_spec": {}},
            id="dev",
        ),
        pytest.param(
            BYOC_FLAGS,
            ["--agent-server-type", "dedicated"],
            {"resource_spec": None, "deployment_type": "prod"},
            id="dedicated",
        ),
        pytest.param(
            BYOC_FLAGS,
            ["--agent-server-type", "prod"],
            {"resource_spec": None, "deployment_type": "prod"},
            id="legacy_prod_is_dedicated",
        ),
    ],
)
@pytest.mark.parametrize("build_mode", ["--no-remote", "--remote"])
def test_byoc_create_sizes_the_agent_server_by_resource_spec(
    byoc_project: DeployProject,
    build_mode: str,
    instance_flags: dict[str, bool],
    type_args: list[str],
    source_config: dict,
) -> None:
    byoc_project.control_plane.instance_flags = instance_flags

    result = byoc_project.run(build_mode, *type_args, host_url=None)

    assert result.exit_code == 0, result.output
    assert (
        byoc_project.control_plane.bodies[CREATE_DEPLOYMENT]["source_config"]
        == source_config
    )


@pytest.mark.parametrize(
    ("instance_flags", "type_args", "expected"),
    [
        pytest.param(
            {CLI_BUILDS: True},
            ["--agent-server-type", "serverless"],
            "does not support serverless (scale to zero) Agent Servers",
            id="serverless_without_scale_to_zero",
        ),
        pytest.param(
            {CLI_BUILDS: True},
            ["--agent-server-type", "dedicated"],
            "does not support dedicated Agent Servers",
            id="dedicated_without_dedicated",
        ),
        pytest.param(
            {CLI_BUILDS: True},
            ["--agent-server-type", "prod"],
            "does not support dedicated Agent Servers",
            id="prod_without_dedicated",
        ),
        pytest.param(
            {SCALE_TO_ZERO: True, DEDICATED: True},
            [],
            "--push-to <registry>/<repository>",
            id="no_cli_builds",
        ),
    ],
)
@pytest.mark.parametrize("build_mode", ["--no-remote", "--remote"])
def test_byoc_refuses_unsupported_deploys_before_creating_anything(
    byoc_project: DeployProject,
    build_mode: str,
    instance_flags: dict[str, bool],
    type_args: list[str],
    expected: str,
) -> None:
    byoc_project.control_plane.instance_flags = instance_flags

    result = byoc_project.run(build_mode, *type_args, host_url=None)

    assert result.exit_code == 2, result.output
    assert expected in result.output
    assert CREATE_DEPLOYMENT not in byoc_project.timeline


@pytest.mark.parametrize("build_mode", ["--no-remote", "--remote"])
def test_byoc_redeploys_an_existing_dedicated_agent_server_without_dedicated_support(
    byoc_project: DeployProject, build_mode: str
) -> None:
    byoc_project.control_plane.instance_flags = {CLI_BUILDS: True}
    byoc_project.control_plane.existing_deployments = [
        {"id": "dep-existing", "name": "my-app"}
    ]

    result = byoc_project.run(
        build_mode, "--agent-server-type", "dedicated", host_url=None
    )

    assert result.exit_code == 0, result.output
    assert CREATE_DEPLOYMENT not in byoc_project.timeline


def test_byoc_without_cli_builds_names_both_external_image_flags(
    byoc_project: DeployProject,
) -> None:
    byoc_project.control_plane.instance_flags = {}

    result = byoc_project.run("--no-remote", host_url=None)

    assert "--push-to" in result.output
    assert "--image-uri" in result.output


def test_byoc_without_cli_builds_still_deploys_an_external_image(
    byoc_project: DeployProject,
) -> None:
    byoc_project.control_plane.instance_flags = {}

    result = byoc_project.run("--image-uri", EXTERNAL_DIGEST, host_url=None)

    assert result.exit_code == 0, result.output
    assert byoc_project.timeline == [
        DISCOVER_WORKSPACES,
        LIST_DEPLOYMENTS,
        CREATE_DEPLOYMENT,
    ]
    assert byoc_project.control_plane.bodies[CREATE_DEPLOYMENT]["source"] == (
        "external_docker"
    )


@pytest.mark.parametrize(
    ("workspaces", "workspaces_status"),
    [
        pytest.param([SAAS_WORKSPACE], 200, id="workspace_without_a_data_plane"),
        pytest.param([], 500, id="discovery_fails"),
    ],
)
def test_saas_deploy_pays_one_discovery_call_and_is_otherwise_unchanged(
    deploy_project: DeployProject, workspaces: list[dict], workspaces_status: int
) -> None:
    deploy_project.control_plane.workspaces = workspaces
    deploy_project.control_plane.workspaces_status = workspaces_status

    result = deploy_project.run("--no-remote", host_url=None)

    assert result.exit_code == 0, result.output
    assert deploy_project.timeline[:3] == [
        DISCOVER_WORKSPACES,
        LIST_DEPLOYMENTS,
        CREATE_DEPLOYMENT,
    ]
    assert DATA_PLANE_INFO not in deploy_project.timeline
    assert all(
        str(r.url).startswith(f"{CLOUD_CONTROL_PLANE_URL}/v2/")
        and "X-Tenant-ID" not in r.headers
        for r in deploy_project.control_plane.host_backend_requests()
    )
    assert deploy_project.control_plane.bodies[CREATE_DEPLOYMENT]["source_config"] == {
        "deployment_type": "dev_zero"
    }


def test_failed_discovery_says_it_falls_back_to_the_default_control_plane(
    deploy_project: DeployProject,
) -> None:
    deploy_project.control_plane.workspaces_status = 500

    result = deploy_project.run("--no-remote", host_url=None)

    assert result.exit_code == 0, result.output
    assert "using the default LangSmith control plane" in result.stderr


def test_host_url_on_a_data_plane_gets_the_drop_host_url_hint(
    byoc_project: DeployProject,
) -> None:
    detail = (
        "'source_config.resource_spec' is required for 'external_docker' source, "
        "and for every source on BYOC data planes, and must be 'null' for all "
        "other sources."
    )
    byoc_project.control_plane.create_error = detail

    result = byoc_project.run(
        "--no-remote", host_url=f"{DATA_PLANE_URL}{CONTROL_PLANE_PATH}"
    )

    assert result.exit_code == 1, result.output
    assert DATA_PLANE_INFO not in byoc_project.timeline
    assert detail in result.output
    assert "drop it so 'langgraph deploy' finds the data plane" in result.output


def test_host_url_overrides_discovery(byoc_project: DeployProject) -> None:
    result = byoc_project.run("--no-remote")

    assert result.exit_code == 0, result.output
    assert DISCOVER_WORKSPACES not in byoc_project.timeline
    assert all(
        str(r.url).startswith(CONTROL_PLANE_URL)
        for r in byoc_project.control_plane.requests
    )


@pytest.mark.parametrize("build_mode", ["--no-remote", "--remote"])
def test_an_older_data_plane_rejecting_the_source_gets_the_external_image_hint(
    deploy_project: DeployProject, build_mode: str
) -> None:
    detail = "'source' must be one of ['github', 'external_docker']"
    deploy_project.control_plane.create_error = detail

    result = deploy_project.run(build_mode)

    assert result.exit_code == 1, result.output
    assert detail in result.output
    assert "--push-to <registry>/<repository>" in result.output
    assert "--image-uri" in result.output


def test_byoc_build_logs_use_the_discovered_data_plane_and_workspace(
    byoc_project: DeployProject,
) -> None:
    result = CliRunner().invoke(
        cli,
        [
            "deploy",
            "logs",
            "--api-key",
            "test-key",
            "--type",
            "build",
            "--agent-server-id",
            "dep-existing",
        ],
    )

    assert result.exit_code == 0, result.output
    assert [
        (str(r.url).split("?")[0], r.headers["X-Tenant-ID"])
        for r in byoc_project.control_plane.host_backend_requests()
    ] == [
        (
            f"{DATA_PLANE_URL}/api-host/v2/deployments/dep-existing/revisions",
            WORKSPACE_ID,
        ),
        (
            f"{DATA_PLANE_URL}/api-host/v1/projects/dep-existing/revisions/"
            f"{REVISION_ID}/build_logs",
            WORKSPACE_ID,
        ),
    ]


def test_mixed_data_planes_without_a_tenant_id_list_through_saas(
    deploy_project: DeployProject,
) -> None:
    deploy_project.control_plane.workspaces = [BYOC_WORKSPACE, SAAS_WORKSPACE]

    result = CliRunner().invoke(
        cli, ["deploy", "list", "--json", "--api-key", "test-key"]
    )

    assert result.exit_code == 0, result.output
    assert deploy_project.timeline == [DISCOVER_WORKSPACES, LIST_DEPLOYMENTS]
    assert [
        str(r.url).split("?")[0]
        for r in deploy_project.control_plane.host_backend_requests()
    ] == [f"{CLOUD_CONTROL_PLANE_URL}/v2/deployments"]


def test_a_prompted_byoc_workspace_is_sent_to_its_data_plane(
    deploy_project: DeployProject,
) -> None:
    deploy_project.control_plane.workspaces = [SAAS_WORKSPACE, BYOC_WORKSPACE]
    deploy_project.control_plane.requires_workspace = True

    result = CliRunner().invoke(
        cli, ["deploy", "list", "--api-key", "test-key"], input=f"{WORKSPACE_ID}\n"
    )

    assert result.exit_code == 0, result.output
    assert [
        (str(r.url).split("?")[0], r.headers.get("X-Tenant-ID"))
        for r in deploy_project.control_plane.host_backend_requests()
    ] == [
        (f"{CLOUD_CONTROL_PLANE_URL}/v2/deployments", None),
        (f"{DATA_PLANE_URL}/api-host/v2/deployments", WORKSPACE_ID),
    ]


def test_an_empty_tenant_id_is_treated_as_unset(
    byoc_project: DeployProject, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("LANGSMITH_TENANT_ID", "")

    result = byoc_project.run("--no-remote", host_url=None)

    assert result.exit_code == 0, result.output
    assert all(
        r.headers["X-Tenant-ID"] == WORKSPACE_ID
        for r in byoc_project.control_plane.host_backend_requests()
    )


def test_a_data_plane_without_https_never_gets_the_api_key(
    deploy_project: DeployProject,
) -> None:
    deploy_project.control_plane.workspaces = [
        {"id": WORKSPACE_ID, "data_plane_url": "http://langsmith.customer.example"}
    ]

    result = deploy_project.run("--no-remote", host_url=None)

    assert result.exit_code == 1, result.output
    assert "must use https" in result.output
    assert deploy_project.timeline == [DISCOVER_WORKSPACES]


def test_a_tenant_id_outside_the_keys_workspaces_fails_before_any_host_call(
    byoc_project: DeployProject, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("LANGSMITH_TENANT_ID", "unknown-ws")

    result = byoc_project.run("--no-remote", host_url=None)

    assert result.exit_code == 1, result.output
    assert "unknown-ws" in result.output
    assert byoc_project.timeline == [DISCOVER_WORKSPACES]


def test_a_data_plane_refusing_local_pushes_shows_its_hint(
    byoc_project: DeployProject,
) -> None:
    hint = "Local image pushes are not supported on this data plane; use --remote."
    byoc_project.control_plane.create_status = 501
    byoc_project.control_plane.create_error = hint

    result = byoc_project.run("--no-remote", host_url=None)

    assert result.exit_code == 1, result.output
    assert hint in result.output
