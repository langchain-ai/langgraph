import asyncio
import json
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path

import click.exceptions
import httpx
import pytest
from click.testing import CliRunner, Result

import langgraph_cli.archive as archive_module
import langgraph_cli.deploy as deploy_module
from langgraph_cli.cli import cli
from langgraph_cli.host_backend import HostBackendClient
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
LISTENER = {
    "id": "listener-1",
    "compute_id": "prod-cluster",
    "compute_config": {"k8s_namespaces": ["agents"]},
}
OTHER_LISTENER = {
    "id": "listener-2",
    "compute_id": "other-cluster",
    "compute_config": {"k8s_namespaces": ["agents"]},
}
TWO_NAMESPACE_LISTENER = {
    "id": "listener-1",
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
NOT_A_CLI_DEPLOYMENT = (
    "push token is only available for 'internal_docker' source deployments"
)
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


@dataclass
class ControlPlaneDouble:
    timeline: list[str]
    existing_deployments: list[dict] = field(default_factory=list)
    push_token_status: int = 200
    create_error: str | None = None
    listeners: list[dict] = field(default_factory=list)
    bodies: dict[str, dict] = field(default_factory=dict)

    def handle(self, request: httpx.Request) -> httpx.Response:
        route = f"{request.method} {request.url.path}"
        self.timeline.append(route)
        if request.content:
            self.bodies[route] = json.loads(request.content)
        return self._respond(request.method, request.url.path)

    def _respond(self, method: str, path: str) -> httpx.Response:
        if (method, path) == ("GET", "/v2/listeners"):
            return httpx.Response(200, json={"resources": self.listeners})
        if (method, path) == ("GET", "/v2/deployments"):
            return httpx.Response(200, json={"resources": self.existing_deployments})
        if (method, path) == ("POST", "/v2/deployments"):
            if self.create_error is not None:
                return httpx.Response(400, text=self.create_error)
            return httpx.Response(201, json={"id": CREATED_ID, "tenant_id": "tenant-1"})
        if path.endswith("/push-token"):
            if self.push_token_status != 200:
                return httpx.Response(self.push_token_status, text=NOT_A_CLI_DEPLOYMENT)
            return httpx.Response(
                200, json={"token": PUSH_TOKEN, "registry_url": REGISTRY_URL}
            )
        if path.endswith("/upload-url"):
            return httpx.Response(
                200, json={"upload_url": SIGNED_UPLOAD_URL, "object_path": OBJECT_PATH}
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

    def client_factory(self) -> Callable[..., HostBackendClient]:
        transport = httpx.MockTransport(self.handle)

        def make(
            host_url: str, api_key: str, tenant_id: str | None = None
        ) -> HostBackendClient:
            return HostBackendClient(host_url, api_key, tenant_id, transport=transport)

        return make


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
        if PLATFORM_FORMAT in args:
            return "linux/amd64\n", None
        if DIGESTS_FORMAT in args:
            repository = ImageReference.parse(args[-1]).repository
            return json.dumps([f"{repository}@sha256:abc123"]), None
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

    def run(self, *args: str, host_url: str = CONTROL_PLANE_URL) -> Result:
        return CliRunner().invoke(
            cli,
            [
                "deploy",
                "--api-key",
                "test-key",
                "--host-url",
                host_url,
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
        deploy_module, "HostBackendClient", control_plane.client_factory()
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
        LIST_DEPLOYMENTS,
        CREATE_DEPLOYMENT,
        "docker build",
        _push_token(CREATED_ID),
        "docker login",
        "docker tag",
        "docker push",
        "docker inspect-digest",
        _patch(CREATED_ID),
    ]
    assert "Deployment updated" in result.output


def test_first_local_deploy_creates_an_internal_docker_deployment(
    deploy_project: DeployProject,
) -> None:
    deploy_project.run("--no-remote")

    assert deploy_project.control_plane.bodies[CREATE_DEPLOYMENT] == {
        "name": "my-app",
        "source": "internal_docker",
        "source_config": {"deployment_type": "dev"},
        "source_revision_config": {},
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
    assert deploy_project.uploads == [(SIGNED_UPLOAD_URL, ARCHIVE[0], ARCHIVE[1])]
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
        LIST_LISTENERS,
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
    assert "Deployment created" in result.output


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


def test_push_to_with_deployment_id_fetches_the_deployment_once(
    deploy_project: DeployProject,
) -> None:
    deploy_project.control_plane.existing_deployments = [
        {"id": "dep-ext", "name": "another-name", "source": "external_docker"}
    ]

    result = deploy_project.run(
        "--deployment-id", "dep-ext", "--push-to", PUSH_REPOSITORY
    )

    assert result.exit_code == 0, result.output
    assert deploy_project.timeline == [
        _get("dep-ext"),
        "docker build",
        "docker push",
        "docker inspect-digest",
        _patch("dep-ext"),
    ]


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
        "listener_id": "listener-1",
        "listener_config": {"k8s_namespace": "agents"},
    }


def test_push_to_places_a_new_deployment_on_the_chosen_listener(
    deploy_project: DeployProject,
) -> None:
    deploy_project.control_plane.listeners = [LISTENER, OTHER_LISTENER]

    result = deploy_project.run(
        "--push-to",
        PUSH_REPOSITORY,
        "--listener-id",
        "listener-2",
        "--k8s-namespace",
        "agents",
        host_url=CLOUD_CONTROL_PLANE_URL,
    )

    assert result.exit_code == 0, result.output
    assert deploy_project.control_plane.bodies[CREATE_DEPLOYMENT]["source_config"] == {
        "resource_spec": {},
        "listener_id": "listener-2",
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
            ("--listener-id", "listener-9"),
            "was not found",
            id="unknown_listener",
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
        "--push-to", PUSH_REPOSITORY, "--listener-id", "listener-1"
    )

    assert result.exit_code == 0, result.output
    assert deploy_project.control_plane.bodies[CREATE_DEPLOYMENT]["source_config"] == {
        "resource_spec": {},
        "listener_id": "listener-1",
        "listener_config": {"k8s_namespace": "agents"},
    }


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
        "listener-1",
        host_url=CLOUD_CONTROL_PLANE_URL,
    )

    assert result.exit_code != 0
    assert "fixed when the deployment is created" in result.output
    assert deploy_project.docker.verbs() == []


def test_automatic_placement_is_announced(deploy_project: DeployProject) -> None:
    deploy_project.control_plane.listeners = [LISTENER]

    result = deploy_project.run(
        "--push-to", PUSH_REPOSITORY, host_url=CLOUD_CONTROL_PLANE_URL
    )

    assert result.exit_code == 0, result.output
    assert "listener-1" in result.output
    assert "agents" in result.output


def test_a_deployment_without_a_listener_announces_nothing(
    deploy_project: DeployProject,
) -> None:
    result = deploy_project.run("--push-to", PUSH_REPOSITORY)

    assert result.exit_code == 0, result.output
    assert "listener" not in result.output
