import asyncio
import json
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path

import click.exceptions
import pytest
from click.testing import CliRunner, Result

import langgraph_cli.archive as archive_module
import langgraph_cli.deploy as deploy_module
from langgraph_cli.cli import cli
from langgraph_cli.host_backend import HostBackendError

CONTROL_PLANE_URL = "https://control-plane.example.com"
REGISTRY_URL = "https://registry.example.com/team"
PUSH_TOKEN = "push-token"
PUSHED_IMAGE = "registry.example.com/team/my-app:latest"
PUSHED_DIGEST = "registry.example.com/team/my-app@sha256:abc123"
CREATED_DEPLOYMENT_ID = "dep-created"
TRACKED_PACKAGES = ["langgraph:1.0.0"]
SIGNED_UPLOAD_URL = "https://storage.example.com/signed"
ARCHIVE = ("/tmp/src.tgz", 2048, "langgraph.json")
OBJECT_PATH = "tarballs/src.tgz"
PLATFORM_FORMAT = "{{.Os}}/{{.Architecture}}"
DIGESTS_FORMAT = "{{json .RepoDigests}}"


@dataclass
class ControlPlaneDouble:
    timeline: list[str]
    existing_deployments: list[dict] = field(default_factory=list)
    push_token_error: HostBackendError | None = None
    payloads: dict[str, dict] = field(default_factory=dict)

    def record(self, method: str, **payload: object) -> None:
        self.timeline.append(method)
        self.payloads[method] = payload

    def client_class(self) -> type:
        double = self

        class FakeHostBackendClient:
            def __init__(
                self, host_url: str, api_key: str, tenant_id: str | None = None
            ) -> None:
                self.base_url = host_url

            def list_deployments(self, name_contains: str = "") -> dict:
                double.record("list_deployments", name_contains=name_contains)
                return {"resources": double.existing_deployments}

            def get_deployment(self, deployment_id: str) -> dict:
                double.record("get_deployment", deployment_id=deployment_id)
                return next(
                    d for d in double.existing_deployments if d["id"] == deployment_id
                )

            def create_deployment(self, **payload: object) -> dict:
                double.record("create_deployment", **payload)
                return {"id": CREATED_DEPLOYMENT_ID}

            def request_push_token(self, deployment_id: str) -> dict:
                double.record("request_push_token", deployment_id=deployment_id)
                if double.push_token_error is not None:
                    raise double.push_token_error
                return {"token": PUSH_TOKEN, "registry_url": REGISTRY_URL}

            def update_deployment(
                self, deployment_id: str, image_uri: str, **payload: object
            ) -> dict:
                double.record(
                    "update_deployment",
                    deployment_id=deployment_id,
                    image_uri=image_uri,
                    **payload,
                )
                return {"tenant_id": "tenant-1"}

            def request_upload_url(self, deployment_id: str) -> dict:
                double.record("request_upload_url", deployment_id=deployment_id)
                return {"upload_url": SIGNED_UPLOAD_URL, "object_path": OBJECT_PATH}

            def update_deployment_internal_source(
                self, deployment_id: str, **payload: object
            ) -> dict:
                double.record(
                    "update_deployment_internal_source",
                    deployment_id=deployment_id,
                    **payload,
                )
                return {"tenant_id": "tenant-1"}

        return FakeHostBackendClient


@dataclass
class DockerCommand:
    args: tuple[str, ...]
    kwargs: dict


@dataclass
class DockerDouble:
    timeline: list[str]
    failing_pushes: int = 0
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
                "docker_command": docker_command,
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
        if PLATFORM_FORMAT in args:
            return "linux/amd64\n", None
        if DIGESTS_FORMAT in args:
            return json.dumps([PUSHED_DIGEST]), None
        return None, None

    @staticmethod
    def _verb(args: tuple[str, ...]) -> str:
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
    uploads: list[tuple[str, str, int]]

    def run(self, *args: str) -> Result:
        return CliRunner().invoke(
            cli,
            [
                "deploy",
                "--api-key",
                "test-key",
                "--host-url",
                CONTROL_PLANE_URL,
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
    for name in ("LANGSMITH_TENANT_ID", "LANGSMITH_ENDPOINT", "LANGGRAPH_HOST_URL"):
        monkeypatch.delenv(name, raising=False)

    timeline: list[str] = []
    control_plane = ControlPlaneDouble(timeline)
    docker = DockerDouble(timeline)
    uploads: list[tuple[str, str, int]] = []

    @contextmanager
    def fake_create_archive(config_path: Path, config: dict) -> Iterator[tuple]:
        timeline.append("create_archive")
        yield ARCHIVE

    def fake_upload(signed_url: str, file_path: str, file_size: int) -> None:
        timeline.append("upload_archive")
        uploads.append((signed_url, file_path, file_size))

    monkeypatch.setattr(deploy_module, "_no_input", False)
    monkeypatch.setattr(deploy_module, "_emitter", None)
    monkeypatch.setattr(
        deploy_module, "HostBackendClient", control_plane.client_class()
    )
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
    monkeypatch.setattr(deploy_module, "_upload_to_gcs", fake_upload)
    return DeployProject(control_plane, docker, timeline, uploads)


def test_first_local_deploy_creates_then_builds_pushes_and_updates_in_order(
    deploy_project: DeployProject,
) -> None:
    result = deploy_project.run("--no-remote")

    assert result.exit_code == 0, result.output
    assert deploy_project.timeline == [
        "list_deployments",
        "create_deployment",
        "docker build",
        "request_push_token",
        "docker login",
        "docker tag",
        "docker push",
        "docker inspect-digest",
        "update_deployment",
    ]
    assert "Deployment updated" in result.output


def test_first_local_deploy_creates_an_internal_docker_deployment(
    deploy_project: DeployProject,
) -> None:
    deploy_project.run("--no-remote")

    assert deploy_project.control_plane.payloads["create_deployment"] == {
        "name": "my-app",
        "deployment_type": "dev",
        "source": "internal_docker",
        "config_path": None,
        "secrets": [],
    }


@pytest.mark.parametrize(
    ("machine", "expected_command", "expected_flags"),
    [
        pytest.param(
            "arm64",
            ("docker", "buildx", "build"),
            ("--platform", "linux/amd64", "--load", "--progress=quiet"),
            id="apple_silicon_cross_builds_for_linux_amd64",
        ),
        pytest.param("x86_64", None, (), id="amd64_host_uses_plain_docker_build"),
    ],
)
def test_local_build_targets_linux_amd64(
    deploy_project: DeployProject,
    monkeypatch: pytest.MonkeyPatch,
    machine: str,
    expected_command: tuple[str, ...] | None,
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
    assert login.args[3:] == (
        "login",
        "-u",
        "oauth2accesstoken",
        "--password-stdin",
        "registry.example.com",
    )
    assert login.kwargs["input"] == f"{PUSH_TOKEN}\n"


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

    assert deploy_project.control_plane.payloads["update_deployment"] == {
        "deployment_id": CREATED_DEPLOYMENT_ID,
        "image_uri": PUSHED_DIGEST,
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
    assert "update_deployment" not in deploy_project.timeline


def test_existing_deployment_matched_by_exact_name_is_updated_not_created(
    deploy_project: DeployProject,
) -> None:
    deploy_project.control_plane.existing_deployments = [
        {"id": "dep-other", "name": "my-app-2"},
        {"id": "dep-existing", "name": "my-app"},
    ]

    deploy_project.run("--no-remote")

    assert "create_deployment" not in deploy_project.timeline
    assert (
        deploy_project.control_plane.payloads["update_deployment"]["deployment_id"]
        == "dep-existing"
    )


def test_deployment_not_created_by_the_cli_gets_an_actionable_error(
    deploy_project: DeployProject,
) -> None:
    deploy_project.control_plane.existing_deployments = [
        {"id": "dep-ui", "name": "my-app"}
    ]
    deploy_project.control_plane.push_token_error = HostBackendError(
        "push token is only available for 'internal_docker' source deployments",
        status_code=400,
    )

    result = deploy_project.run("--no-remote")

    assert result.exit_code != 0
    assert "was not created by 'langgraph deploy'" in result.output
    assert "docker login" not in deploy_project.timeline


def test_remote_build_creates_an_internal_source_deployment_and_uploads_the_archive(
    deploy_project: DeployProject,
) -> None:
    result = deploy_project.run("--remote", "--install-command", "yarn install")

    assert result.exit_code == 0, result.output
    assert deploy_project.timeline == [
        "list_deployments",
        "create_deployment",
        "create_archive",
        "request_upload_url",
        "upload_archive",
        "update_deployment_internal_source",
    ]
    assert deploy_project.control_plane.payloads["create_deployment"]["source"] == (
        "internal_source"
    )
    assert deploy_project.uploads == [(SIGNED_UPLOAD_URL, ARCHIVE[0], ARCHIVE[1])]
    assert deploy_project.control_plane.payloads[
        "update_deployment_internal_source"
    ] == {
        "deployment_id": CREATED_DEPLOYMENT_ID,
        "source_tarball_path": OBJECT_PATH,
        "config_path": ARCHIVE[2],
        "secrets": [],
        "install_command": "yarn install",
        "build_command": None,
        "tracked_packages": TRACKED_PACKAGES,
    }
    assert "Build triggered" in result.output
