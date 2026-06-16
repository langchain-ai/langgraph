try:
    import tomllib
except ImportError:
    import tomli as tomllib
from pathlib import Path


def test_inmem_extra_excludes_langgraph_api_dev_releases() -> None:
    pyproject = Path(__file__).parents[2] / "pyproject.toml"

    project = tomllib.loads(pyproject.read_text())["project"]
    inmem_dependencies = project["optional-dependencies"]["inmem"]

    langgraph_api = next(
        dependency
        for dependency in inmem_dependencies
        if dependency.startswith("langgraph-api")
    )
    assert "<0.12.0 ;" in langgraph_api
    assert "<0.12.0a0" not in langgraph_api
