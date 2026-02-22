import json
import os
import pathlib
from unittest.mock import patch

import pytest

from langgraph_cli.config import config_to_docker, validate_config

FIXTURES_ROOT = (
    pathlib.Path(__file__).parent.parent / "local_cache_validation" / "projects"
)
CACHE_ENV = "LANGGRAPH_CACHE_OPTIMIZE"
TIERS = ("off", "lock", "pyproject", "all")

pytestmark = pytest.mark.skipif(
    os.getenv("RUN_LOCAL_CACHE_VALIDATION") != "1",
    reason="Local-only cache optimization TDD tests.",
)


def _dockerfile_for_fixture(fixture_name: str, tier: str) -> str:
    config_path = FIXTURES_ROOT / fixture_name / "langgraph.json"
    with open(config_path, encoding="utf-8") as f:
        raw = json.load(f)
    config = validate_config(raw)

    with patch.dict(os.environ, {CACHE_ENV: tier}):
        dockerfile, _ = config_to_docker(
            config_path=config_path,
            config=config,
            base_image="langchain/langgraph-api",
        )
    return dockerfile


@pytest.mark.parametrize(
    "fixture_name,expected",
    [
        (
            "real_pyproject_lock",
            {"off": False, "lock": True, "pyproject": True, "all": True},
        ),
        (
            "real_pyproject_no_lock",
            {"off": False, "lock": False, "pyproject": True, "all": True},
        ),
        (
            "real_setup_py",
            {"off": False, "lock": False, "pyproject": False, "all": True},
        ),
        (
            "faux_package",
            {"off": False, "lock": False, "pyproject": False, "all": False},
        ),
        (
            "pip_installer_fallback",
            {"off": False, "lock": False, "pyproject": False, "all": False},
        ),
    ],
)
def test_tier_deferred_copy_matrix(
    fixture_name: str, expected: dict[str, bool]
) -> None:
    for tier in TIERS:
        dockerfile = _dockerfile_for_fixture(fixture_name, tier)
        has_no_deps_editable = "--no-deps -e ." in dockerfile
        assert has_no_deps_editable is expected[tier], (
            f"{fixture_name=} {tier=} expected deferred copy {expected[tier]}, "
            f"got {has_no_deps_editable}"
        )


def test_requirements_before_no_deps_for_eligible_tier() -> None:
    dockerfile = _dockerfile_for_fixture("real_pyproject_lock", "lock")
    reqs_idx = dockerfile.find("Installing from requirements.txt files")
    no_deps_idx = dockerfile.find("--no-deps -e .")
    assert reqs_idx != -1
    assert no_deps_idx != -1
    assert reqs_idx < no_deps_idx


def test_generation_command_selection_by_fixture() -> None:
    docker_lock = _dockerfile_for_fixture("real_pyproject_lock", "lock")
    assert "uv export --no-hashes --no-dev --no-emit-local" in docker_lock

    docker_pyproject = _dockerfile_for_fixture("real_pyproject_no_lock", "pyproject")
    assert "uv pip compile pyproject.toml" in docker_pyproject

    docker_setup = _dockerfile_for_fixture("real_setup_py", "all")
    assert "uv pip compile setup.py" in docker_setup


def test_pip_installer_fallback_ignores_tier() -> None:
    outputs = {
        tier: _dockerfile_for_fixture("pip_installer_fallback", tier) for tier in TIERS
    }
    first = outputs["off"]
    for tier, dockerfile in outputs.items():
        assert dockerfile == first, f"Expected identical dockerfile for {tier=}"
