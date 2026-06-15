from pathlib import Path

import tomllib

from langgraph.version import __version__


def test_version_matches_pyproject() -> None:
    pyproject_path = Path(__file__).resolve().parents[1] / "pyproject.toml"
    with pyproject_path.open("rb") as handle:
        project_config = tomllib.load(handle)

    assert __version__ == project_config["project"]["version"]
