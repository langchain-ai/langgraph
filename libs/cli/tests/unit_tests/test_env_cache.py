"""Generated Dockerfile cache keys; opt into the real BuildKit test locally."""

import hashlib
import os
import pathlib
import subprocess
import uuid

import pytest

from langgraph_cli.config import config_to_docker, validate_config


def generated_prefix(tmp_path: pathlib.Path, runtime: str) -> str:
    config = {
        "graphs": {"agent": "agent:graph"},
        "dockerfile_lines": [
            f"RUN --mount=type=secret,id=build_marker echo {tmp_path.name} >/dev/null; "
            "sha256sum /run/secrets/build_marker | cut -d ' ' -f 1 > /result"
        ],
    }
    if runtime == "node":
        config["node_version"] = "20"
        config["graphs"] = {"agent": "./agent.js:graph"}
    else:
        config.update(python_version="3.11", dependencies=["requests"])
        if runtime == "uv":
            config.pop("dependencies")
            config["source"] = {"kind": "uv", "root": "."}
            (tmp_path / "uv.lock").write_text("version = 1\n")
            (tmp_path / "pyproject.toml").write_text(
                '[project]\nname = "agent"\nversion = "0.1.0"\n'
                '[build-system]\nrequires = ["setuptools"]\n'
                'build-backend = "setuptools.build_meta"\n'
            )
    dockerfile, _ = config_to_docker(
        tmp_path / "langgraph.json",
        validate_config(config),
        base_image="langchain/langgraph-api",
    )
    # Keep the generated stage header and custom RUN exactly as emitted. Avoid
    # installing real graph dependencies: this test exercises the cache boundary.
    return dockerfile.split(" > /result", 1)[0] + " > /result\n"


@pytest.mark.parametrize("runtime", ["python", "uv", "node"])
def test_env_hash_in_build_stage_before_custom_run(tmp_path, runtime):
    prefix = generated_prefix(tmp_path, runtime)
    assert prefix.index("FROM ") < prefix.index("ARG USER_ENV_VARS_HASH")
    assert prefix.index("ARG USER_ENV_VARS_HASH") < prefix.index("RUN ")


@pytest.mark.skipif(
    os.environ.get("LANGGRAPH_TEST_BUILDKIT") != "1",
    reason="Set LANGGRAPH_TEST_BUILDKIT=1 with a local Docker BuildKit daemon",
)
@pytest.mark.parametrize("runtime", ["python", "uv", "node"])
def test_secret_hash_invalidates_real_buildkit_cache(tmp_path, runtime, request):
    prefix = generated_prefix(tmp_path, runtime)
    lines = prefix.splitlines()
    lines[0] = "FROM alpine:3.21"
    prefix = "\n".join(lines) + "\n"
    dockerfile = tmp_path / "Dockerfile"
    marker = tmp_path / "build-marker"
    (tmp_path / ".dockerignore").write_text("*\n!Dockerfile\n")
    output_counter = 0
    cache_image = "langgraph-cli-cache-test:" + uuid.uuid4().hex
    request.addfinalizer(
        lambda: subprocess.run(
            ["docker", "image", "rm", cache_image], capture_output=True, check=False
        )
    )

    def build(contents, value, *, pass_hash=True, export_image=False):
        nonlocal output_counter
        output_counter += 1
        marker.write_text(value)
        dockerfile.write_text(
            contents + "FROM scratch\nCOPY --from=0 /result /result\n"
        )
        output = tmp_path / f"output-{output_counter}"
        args = [
            "docker",
            "buildx",
            "build",
            "--progress=plain",
            "--secret",
            f"id=build_marker,src={marker}",
            "--output",
            f"type=local,dest={output}",
            "-f",
            str(dockerfile),
            str(tmp_path),
        ]
        if pass_hash:
            args.extend(
                [
                    "--build-arg",
                    f"USER_ENV_VARS_HASH={hashlib.sha256(value.encode()).hexdigest()}",
                ]
            )
        if output_counter > 1:
            args.extend(["--cache-from", cache_image])
        if export_image:
            # Export a legacy image with inline cache, then explicitly import it.
            image_args = args.copy()
            output_index = image_args.index("--output")
            del image_args[output_index : output_index + 2]
            image_args.extend(
                ["--load", "--tag", cache_image, "--cache-to", "type=inline"]
            )
            subprocess.run(image_args, capture_output=True, text=True, check=True)
        # List arguments only; all inputs are local public test markers.
        result = subprocess.run(args, capture_output=True, text=True, check=True)
        return (output / "result").read_text().strip(), result.stderr

    first = f"{runtime}-{tmp_path.name}-first"
    second = f"{runtime}-{tmp_path.name}-second"
    legacy = prefix.replace("ARG USER_ENV_VARS_HASH\n", "")
    legacy_output, _ = build(legacy, first, export_image=True)
    assert legacy_output == hashlib.sha256(first.encode()).hexdigest()
    stale_output, _ = build(legacy, second)
    assert stale_output == legacy_output  # Secret contents alone do not bust cache.

    fixed_output, _ = build(prefix, second)
    assert fixed_output == hashlib.sha256(second.encode()).hexdigest()
    assert fixed_output != stale_output  # Old cache is safe when adopting the ARG.
    repeated_output, repeated_log = build(prefix, second)
    assert repeated_output == fixed_output
    run_vertex = next(
        line.split()[0]
        for line in repeated_log.splitlines()
        if "] RUN " in line and "sha256sum" in line
    )
    assert f"{run_vertex} CACHED" in repeated_log
    changed_output, _ = build(prefix, first)
    assert changed_output == hashlib.sha256(first.encode()).hexdigest()
    assert changed_output != repeated_output

    # Ordinary CLI callers can still build without supplying the optional hash.
    unset_output, _ = build(prefix, first, pass_hash=False)
    assert unset_output == hashlib.sha256(first.encode()).hexdigest()
