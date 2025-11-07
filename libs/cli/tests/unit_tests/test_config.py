import copy
import json
import os
import pathlib
import tempfile
import textwrap

import click
import pytest

from langgraph_cli.config import (
    _BUILD_TOOLS,
    _get_pip_cleanup_lines,
    config_to_compose,
    config_to_docker,
    docker_tag,
    validate_config,
    validate_config_file,
)
from langgraph_cli.util import clean_empty_lines

FORMATTED_CLEANUP_LINES = _get_pip_cleanup_lines(
    install_cmd="uv pip install --system",
    to_uninstall=("pip", "setuptools", "wheel"),
    pip_installer="uv",
)

PATH_TO_CONFIG = pathlib.Path(__file__).parent / "test_config.json"


def test_validate_config():
    # minimal config
    expected_config = {
        "dependencies": ["."],
        "graphs": {
            "agent": "./agent.py:graph",
        },
    }
    actual_config = validate_config(expected_config)
    expected_config = {
        "base_image": None,
        "python_version": "3.11",
        "node_version": None,
        "pip_config_file": None,
        "pip_installer": "auto",
        "image_distro": "debian",
        "dockerfile_lines": [],
        "env": {},
        "store": None,
        "auth": None,
        "checkpointer": None,
        "http": None,
        "ui": None,
        "ui_config": None,
        "keep_pkg_tools": None,
        **expected_config,
    }
    assert actual_config == expected_config

    # full config
    env = ".env"
    expected_config = {
        "base_image": None,
        "python_version": "3.12",
        "node_version": None,
        "pip_config_file": "pipconfig.txt",
        "pip_installer": "auto",
        "image_distro": "debian",
        "dockerfile_lines": ["ARG meow"],
        "dependencies": [".", "langchain"],
        "graphs": {
            "agent": "./agent.py:graph",
        },
        "env": env,
        "store": None,
        "auth": None,
        "checkpointer": None,
        "http": None,
        "ui": None,
        "ui_config": None,
        "keep_pkg_tools": None,
    }
    actual_config = validate_config(expected_config)
    assert actual_config == expected_config
    expected_config["python_version"] = "3.13"
    actual_config = validate_config(expected_config)
    assert actual_config == expected_config

    # check wrong python version raises
    with pytest.raises(click.UsageError):
        validate_config({"python_version": "3.9"})

    # check missing dependencies key raises
    with pytest.raises(click.UsageError):
        validate_config(
            {"python_version": "3.9", "graphs": {"agent": "./agent.py:graph"}}
        )

    # check missing graphs key raises
    with pytest.raises(click.UsageError):
        validate_config({"python_version": "3.9", "dependencies": ["."]})

    with pytest.raises(click.UsageError) as exc_info:
        validate_config({"python_version": "3.11.0"})
    assert "Invalid Python version format" in str(exc_info.value)

    with pytest.raises(click.UsageError) as exc_info:
        validate_config({"python_version": "3"})
    assert "Invalid Python version format" in str(exc_info.value)

    with pytest.raises(click.UsageError) as exc_info:
        validate_config({"python_version": "abc.def"})
    assert "Invalid Python version format" in str(exc_info.value)

    with pytest.raises(click.UsageError) as exc_info:
        validate_config({"python_version": "3.10"})
    assert "Minimum required version" in str(exc_info.value)

    config = validate_config(
        {
            "python_version": "3.11-bullseye",
            "dependencies": ["."],
            "graphs": {"agent": "./agent.py:graph"},
        }
    )
    assert config["python_version"] == "3.11-bullseye"

    config = validate_config(
        {
            "python_version": "3.12-slim",
            "dependencies": ["."],
            "graphs": {"agent": "./agent.py:graph"},
        }
    )
    assert config["python_version"] == "3.12-slim"
    with pytest.raises(ValueError, match="Invalid http.app format"):
        validate_config(
            {
                "python_version": "3.12",
                "dependencies": ["."],
                "graphs": {"agent": "./agent.py:graph"},
                "http": {"app": "../../examples/my_app.py"},
            }
        )


def test_validate_config_image_distro():
    """Test validation of image_distro field."""
    # Valid image_distro values should work
    config = validate_config(
        {
            "python_version": "3.11",
            "dependencies": ["."],
            "graphs": {"agent": "./agent.py:graph"},
            "image_distro": "debian",
        }
    )
    assert config["image_distro"] == "debian"

    config = validate_config(
        {
            "python_version": "3.11",
            "dependencies": ["."],
            "graphs": {"agent": "./agent.py:graph"},
            "image_distro": "wolfi",
        }
    )
    assert config["image_distro"] == "wolfi"

    # Missing image_distro should default to 'debian'
    config = validate_config(
        {
            "python_version": "3.11",
            "dependencies": ["."],
            "graphs": {"agent": "./agent.py:graph"},
        }
    )
    assert config["image_distro"] == "debian"

    # Invalid image_distro values should raise error
    with pytest.raises(click.UsageError) as exc_info:
        validate_config(
            {
                "python_version": "3.11",
                "dependencies": ["."],
                "graphs": {"agent": "./agent.py:graph"},
                "image_distro": "ubuntu",
            }
        )
    assert "Invalid image_distro: 'ubuntu'" in str(exc_info.value)

    with pytest.raises(click.UsageError) as exc_info:
        validate_config(
            {
                "python_version": "3.11",
                "dependencies": ["."],
                "graphs": {"agent": "./agent.py:graph"},
                "image_distro": "alpine",
            }
        )
    assert "Invalid image_distro: 'alpine'" in str(exc_info.value)

    # Test base Node.js config with image distro
    config = validate_config(
        {
            "node_version": "20",
            "graphs": {"agent": "./agent.js:graph"},
            "image_distro": "wolfi",
        }
    )
    assert config["image_distro"] == "wolfi"

    # Test Node.js config with no distro specified
    config = validate_config(
        {
            "node_version": "20",
            "graphs": {"agent": "./agent.js:graph"},
        }
    )
    assert config["image_distro"] == "debian"


def test_validate_config_pip_installer():
    """Test validation of pip_installer field."""
    # Valid pip_installer values should work
    config = validate_config(
        {
            "python_version": "3.11",
            "dependencies": ["."],
            "graphs": {"agent": "./agent.py:graph"},
            "pip_installer": "auto",
        }
    )
    assert config["pip_installer"] == "auto"

    config = validate_config(
        {
            "python_version": "3.11",
            "dependencies": ["."],
            "graphs": {"agent": "./agent.py:graph"},
            "pip_installer": "pip",
        }
    )
    assert config["pip_installer"] == "pip"

    config = validate_config(
        {
            "python_version": "3.11",
            "dependencies": ["."],
            "graphs": {"agent": "./agent.py:graph"},
            "pip_installer": "uv",
        }
    )
    assert config["pip_installer"] == "uv"

    # Missing pip_installer should default to "auto"
    config = validate_config(
        {
            "python_version": "3.11",
            "dependencies": ["."],
            "graphs": {"agent": "./agent.py:graph"},
        }
    )
    assert config["pip_installer"] == "auto"

    # Invalid pip_installer values should raise error
    with pytest.raises(click.UsageError) as exc_info:
        validate_config(
            {
                "python_version": "3.11",
                "dependencies": ["."],
                "graphs": {"agent": "./agent.py:graph"},
                "pip_installer": "conda",
            }
        )
    assert "Invalid pip_installer: 'conda'" in str(exc_info.value)
    assert "Must be 'auto', 'pip', or 'uv'" in str(exc_info.value)

    with pytest.raises(click.UsageError) as exc_info:
        validate_config(
            {
                "python_version": "3.11",
                "dependencies": ["."],
                "graphs": {"agent": "./agent.py:graph"},
                "pip_installer": "invalid",
            }
        )
    assert "Invalid pip_installer: 'invalid'" in str(exc_info.value)


def test_validate_config_file():
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_path = pathlib.Path(tmpdir)

        config_path = tmpdir_path / "langgraph.json"

        node_config = {"node_version": "20", "graphs": {"agent": "./agent.js:graph"}}
        with open(config_path, "w") as f:
            json.dump(node_config, f)

        validate_config_file(config_path)

        package_json = {"name": "test", "engines": {"node": "20"}}
        with open(tmpdir_path / "package.json", "w") as f:
            json.dump(package_json, f)
        validate_config_file(config_path)

        package_json["engines"]["node"] = "20.18"
        with open(tmpdir_path / "package.json", "w") as f:
            json.dump(package_json, f)
        with pytest.raises(click.UsageError, match="Use major version only"):
            validate_config_file(config_path)

        package_json["engines"] = {"node": "18"}
        with open(tmpdir_path / "package.json", "w") as f:
            json.dump(package_json, f)
        with pytest.raises(click.UsageError, match="must be >= 20"):
            validate_config_file(config_path)

        package_json["engines"] = {"node": "20", "deno": "1.0"}
        with open(tmpdir_path / "package.json", "w") as f:
            json.dump(package_json, f)
        with pytest.raises(click.UsageError, match="Only 'node' engine is supported"):
            validate_config_file(config_path)

        with open(tmpdir_path / "package.json", "w") as f:
            f.write("{invalid json")
        with pytest.raises(click.UsageError, match="Invalid package.json"):
            validate_config_file(config_path)

        python_config = {
            "python_version": "3.11",
            "dependencies": ["."],
            "graphs": {"agent": "./agent.py:graph"},
        }
        with open(config_path, "w") as f:
            json.dump(python_config, f)

        validate_config_file(config_path)

        for package_content in [
            {"name": "test"},
            {"engines": {"node": "18"}},
            {"engines": {"node": "20", "deno": "1.0"}},
            "{invalid json",
        ]:
            with open(tmpdir_path / "package.json", "w") as f:
                if isinstance(package_content, dict):
                    json.dump(package_content, f)
                else:
                    f.write(package_content)
            validate_config_file(config_path)


def test_validate_config_multiplatform():
    # default node
    config = validate_config(
        {"dependencies": ["."], "graphs": {"js": "./js.mts:graph"}}
    )
    assert config["node_version"] == "20"
    assert config["python_version"] is None

    # default multiplatform
    config = validate_config(
        {
            "node_version": "22",
            "python_version": "3.12",
            "dependencies": ["."],
            "graphs": {"python": "./python.py:graph", "js": "./js.mts:graph"},
        }
    )
    assert config["node_version"] == "22"
    assert config["python_version"] == "3.12"

    # default multiplatform (full infer)
    graphs = {"python": "./python.py:graph", "js": "./js.mts:graph"}
    config = validate_config({"dependencies": ["."], "graphs": graphs})
    assert config["node_version"] == "20"
    assert config["python_version"] == "3.11"

    # default multiplatform (partial node)
    config = validate_config(
        {"node_version": "22", "dependencies": ["."], "graphs": graphs}
    )
    assert config["node_version"] == "22"
    assert config["python_version"] == "3.11"

    # default multiplatform (partial python)
    config = validate_config(
        {"python_version": "3.12", "dependencies": ["."], "graphs": graphs}
    )
    assert config["node_version"] == "20"
    assert config["python_version"] == "3.12"

    # no known extension (assumes python)
    config = validate_config(
        {
            "dependencies": ["./local", "./shared_utils"],
            "graphs": {"agent": "local.workflow:graph"},
            "env": ".env",
        }
    )
    assert config["node_version"] is None
    assert config["python_version"] == "3.11"


# config_to_docker
def test_config_to_docker_simple():
    graphs = {"agent": "./agent.py:graph"}
    actual_docker_stdin, additional_contexts = config_to_docker(
        PATH_TO_CONFIG,
        validate_config(
            {
                "dependencies": [".", "../../examples/graphs_reqs_a", "../../examples"],
                "graphs": graphs,
                "http": {"app": "../../examples/my_app.py:app"},
            }
        ),
        "langchain/langgraph-api",
    )
    expected_docker_stdin = f"""\
# syntax=docker/dockerfile:1.4
FROM langchain/langgraph-api:3.11
# -- Installing local requirements --
COPY --from=outer-requirements.txt requirements.txt /deps/outer-graphs_reqs_a/graphs_reqs_a/requirements.txt
RUN PYTHONDONTWRITEBYTECODE=1 uv pip install --system --no-cache-dir -c /api/constraints.txt -r /deps/outer-graphs_reqs_a/graphs_reqs_a/requirements.txt
# -- End of local requirements install --
# -- Adding local package ../../examples --
COPY --from=examples . /deps/examples
# -- End of local package ../../examples --
# -- Adding non-package dependency unit_tests --
ADD . /deps/outer-unit_tests/unit_tests
RUN set -ex && \\
    for line in '[project]' \\
                'name = "unit_tests"' \\
                'version = "0.1"' \\
                '[tool.setuptools.package-data]' \\
                '"*" = ["**/*"]' \\
                '[build-system]' \\
                'requires = ["setuptools>=61"]' \\
                'build-backend = "setuptools.build_meta"'; do \\
        echo "$line" >> /deps/outer-unit_tests/pyproject.toml; \\
    done
# -- End of non-package dependency unit_tests --
# -- Adding non-package dependency graphs_reqs_a --
COPY --from=outer-graphs_reqs_a . /deps/outer-graphs_reqs_a/graphs_reqs_a
RUN set -ex && \\
    for line in '[project]' \\
                'name = "graphs_reqs_a"' \\
                'version = "0.1"' \\
                '[tool.setuptools.package-data]' \\
                '"*" = ["**/*"]' \\
                '[build-system]' \\
                'requires = ["setuptools>=61"]' \\
                'build-backend = "setuptools.build_meta"'; do \\
        echo "$line" >> /deps/outer-graphs_reqs_a/pyproject.toml; \\
    done
# -- End of non-package dependency graphs_reqs_a --
# -- Installing all local dependencies --
RUN for dep in /deps/*; do             echo "Installing $dep";             if [ -d "$dep" ]; then                 echo "Installing $dep";                 (cd "$dep" && PYTHONDONTWRITEBYTECODE=1 uv pip install --system --no-cache-dir -c /api/constraints.txt -e .);             fi;         done
# -- End of local dependencies install --
ENV LANGGRAPH_HTTP='{{"app": "/deps/examples/my_app.py:app"}}'
ENV LANGSERVE_GRAPHS='{{"agent": "/deps/outer-unit_tests/unit_tests/agent.py:graph"}}'
{FORMATTED_CLEANUP_LINES}
WORKDIR /deps/outer-unit_tests/unit_tests\
"""
    assert clean_empty_lines(actual_docker_stdin) == expected_docker_stdin

    assert additional_contexts == {
        "outer-graphs_reqs_a": str(
            (pathlib.Path(__file__).parent / "../../examples/graphs_reqs_a").resolve()
        ),
        "examples": str((pathlib.Path(__file__).parent / "../../examples").resolve()),
    }


def test_config_to_docker_outside_path():
    graphs = {"agent": "./agent.py:graph"}
    actual_docker_stdin, additional_contexts = config_to_docker(
        PATH_TO_CONFIG,
        validate_config({"dependencies": [".", ".."], "graphs": graphs}),
        "langchain/langgraph-api",
    )
    expected_docker_stdin = (
        """\
# syntax=docker/dockerfile:1.4
FROM langchain/langgraph-api:3.11
# -- Adding non-package dependency unit_tests --
ADD . /deps/outer-unit_tests/unit_tests
RUN set -ex && \\
    for line in '[project]' \\
                'name = "unit_tests"' \\
                'version = "0.1"' \\
                '[tool.setuptools.package-data]' \\
                '"*" = ["**/*"]' \\
                '[build-system]' \\
                'requires = ["setuptools>=61"]' \\
                'build-backend = "setuptools.build_meta"'; do \\
        echo "$line" >> /deps/outer-unit_tests/pyproject.toml; \\
    done
# -- End of non-package dependency unit_tests --
# -- Adding non-package dependency tests --
COPY --from=outer-tests . /deps/outer-tests/tests
RUN set -ex && \\
    for line in '[project]' \\
                'name = "tests"' \\
                'version = "0.1"' \\
                '[tool.setuptools.package-data]' \\
                '"*" = ["**/*"]' \\
                '[build-system]' \\
                'requires = ["setuptools>=61"]' \\
                'build-backend = "setuptools.build_meta"'; do \\
        echo "$line" >> /deps/outer-tests/pyproject.toml; \\
    done
# -- End of non-package dependency tests --
# -- Installing all local dependencies --
RUN for dep in /deps/*; do             echo "Installing $dep";             if [ -d "$dep" ]; then                 echo "Installing $dep";                 (cd "$dep" && PYTHONDONTWRITEBYTECODE=1 uv pip install --system --no-cache-dir -c /api/constraints.txt -e .);             fi;         done
# -- End of local dependencies install --
ENV LANGSERVE_GRAPHS='{"agent": "/deps/outer-unit_tests/unit_tests/agent.py:graph"}'
"""
        + FORMATTED_CLEANUP_LINES
        + """
WORKDIR /deps/outer-unit_tests/unit_tests\
"""
    )
    assert clean_empty_lines(actual_docker_stdin) == expected_docker_stdin
    assert additional_contexts == {
        "outer-tests": str(pathlib.Path(__file__).parent.parent.absolute()),
    }


def test_config_to_docker_pipconfig():
    graphs = {"agent": "./agent.py:graph"}
    actual_docker_stdin, additional_contexts = config_to_docker(
        PATH_TO_CONFIG,
        validate_config(
            {
                "dependencies": ["."],
                "graphs": graphs,
                "pip_config_file": "pipconfig.txt",
            }
        ),
        "langchain/langgraph-api",
    )
    expected_docker_stdin = (
        """\
FROM langchain/langgraph-api:3.11
ADD pipconfig.txt /pipconfig.txt
# -- Adding non-package dependency unit_tests --
ADD . /deps/outer-unit_tests/unit_tests
RUN set -ex && \\
    for line in '[project]' \\
                'name = "unit_tests"' \\
                'version = "0.1"' \\
                '[tool.setuptools.package-data]' \\
                '"*" = ["**/*"]' \\
                '[build-system]' \\
                'requires = ["setuptools>=61"]' \\
                'build-backend = "setuptools.build_meta"'; do \\
        echo "$line" >> /deps/outer-unit_tests/pyproject.toml; \\
    done
# -- End of non-package dependency unit_tests --
# -- Installing all local dependencies --
RUN for dep in /deps/*; do             echo "Installing $dep";             if [ -d "$dep" ]; then                 echo "Installing $dep";                 (cd "$dep" && PIP_CONFIG_FILE=/pipconfig.txt PYTHONDONTWRITEBYTECODE=1 uv pip install --system --no-cache-dir -c /api/constraints.txt -e .);             fi;         done
# -- End of local dependencies install --
ENV LANGSERVE_GRAPHS='{"agent": "/deps/outer-unit_tests/unit_tests/agent.py:graph"}'
"""
        + FORMATTED_CLEANUP_LINES
        + """
WORKDIR /deps/outer-unit_tests/unit_tests\
"""
    )
    assert clean_empty_lines(actual_docker_stdin) == expected_docker_stdin
    assert additional_contexts == {}


def test_config_to_docker_invalid_inputs():
    # test missing local dependencies
    with pytest.raises(FileNotFoundError):
        graphs = {"agent": "tests/unit_tests/agent.py:graph"}
        config_to_docker(
            PATH_TO_CONFIG,
            validate_config({"dependencies": ["./missing"], "graphs": graphs}),
            "langchain/langgraph-api",
        )

    # test missing local module
    with pytest.raises(FileNotFoundError):
        graphs = {"agent": "./missing_agent.py:graph"}
        config_to_docker(
            PATH_TO_CONFIG,
            validate_config({"dependencies": ["."], "graphs": graphs}),
            "langchain/langgraph-api",
        )


def test_config_to_docker_local_deps():
    graphs = {"agent": "./graphs/agent.py:graph"}
    actual_docker_stdin, additional_contexts = config_to_docker(
        PATH_TO_CONFIG,
        validate_config(
            {
                "dependencies": ["./graphs"],
                "graphs": graphs,
            }
        ),
        "langchain/langgraph-api-custom",
    )
    expected_docker_stdin = f"""\
FROM langchain/langgraph-api-custom:3.11
# -- Adding non-package dependency graphs --
ADD ./graphs /deps/outer-graphs/src
RUN set -ex && \\
    for line in '[project]' \\
                'name = "graphs"' \\
                'version = "0.1"' \\
                '[tool.setuptools.package-data]' \\
                '"*" = ["**/*"]' \\
                '[build-system]' \\
                'requires = ["setuptools>=61"]' \\
                'build-backend = "setuptools.build_meta"'; do \\
        echo "$line" >> /deps/outer-graphs/pyproject.toml; \\
    done
# -- End of non-package dependency graphs --
# -- Installing all local dependencies --
RUN for dep in /deps/*; do             echo "Installing $dep";             if [ -d "$dep" ]; then                 echo "Installing $dep";                 (cd "$dep" && PYTHONDONTWRITEBYTECODE=1 uv pip install --system --no-cache-dir -c /api/constraints.txt -e .);             fi;         done
# -- End of local dependencies install --
ENV LANGSERVE_GRAPHS='{{"agent": "/deps/outer-graphs/src/agent.py:graph"}}'
{FORMATTED_CLEANUP_LINES}\
"""
    assert clean_empty_lines(actual_docker_stdin) == expected_docker_stdin
    assert additional_contexts == {}


def test_config_to_docker_pyproject():
    pyproject_str = """[project]
name = "custom"
version = "0.1"
dependencies = ["langchain"]"""
    pyproject_path = "tests/unit_tests/pyproject.toml"
    with open(pyproject_path, "w") as f:
        f.write(pyproject_str)

    graphs = {"agent": "./graphs/agent.py:graph"}
    actual_docker_stdin, additional_contexts = config_to_docker(
        PATH_TO_CONFIG,
        validate_config(
            {
                "dependencies": ["."],
                "graphs": graphs,
            }
        ),
        "langchain/langgraph-api",
    )
    os.remove(pyproject_path)
    expected_docker_stdin = (
        """FROM langchain/langgraph-api:3.11
# -- Adding local package . --
ADD . /deps/unit_tests
# -- End of local package . --
# -- Installing all local dependencies --
RUN for dep in /deps/*; do             echo "Installing $dep";             if [ -d "$dep" ]; then                 echo "Installing $dep";                 (cd "$dep" && PYTHONDONTWRITEBYTECODE=1 uv pip install --system --no-cache-dir -c /api/constraints.txt -e .);             fi;         done
# -- End of local dependencies install --
ENV LANGSERVE_GRAPHS='{"agent": "/deps/unit_tests/graphs/agent.py:graph"}'
"""
        + FORMATTED_CLEANUP_LINES
        + "\n"
        + "WORKDIR /deps/unit_tests"
        ""
    )
    assert clean_empty_lines(actual_docker_stdin) == expected_docker_stdin
    assert additional_contexts == {}


def test_config_to_docker_end_to_end():
    graphs = {"agent": "./graphs/agent.py:graph"}
    actual_docker_stdin, additional_contexts = config_to_docker(
        PATH_TO_CONFIG,
        validate_config(
            {
                "python_version": "3.12",
                "dependencies": ["./graphs/", "langchain", "langchain_openai"],
                "graphs": graphs,
                "pip_config_file": "pipconfig.txt",
                "dockerfile_lines": ["ARG meow", "ARG foo"],
            }
        ),
        "langchain/langgraph-api",
    )
    expected_docker_stdin = f"""FROM langchain/langgraph-api:3.12
ARG meow
ARG foo
ADD pipconfig.txt /pipconfig.txt
RUN PIP_CONFIG_FILE=/pipconfig.txt PYTHONDONTWRITEBYTECODE=1 uv pip install --system --no-cache-dir -c /api/constraints.txt langchain langchain_openai
# -- Adding non-package dependency graphs --
ADD ./graphs/ /deps/outer-graphs/src
RUN set -ex && \\
    for line in '[project]' \\
                'name = "graphs"' \\
                'version = "0.1"' \\
                '[tool.setuptools.package-data]' \\
                '"*" = ["**/*"]' \\
                '[build-system]' \\
                'requires = ["setuptools>=61"]' \\
                'build-backend = "setuptools.build_meta"'; do \\
        echo "$line" >> /deps/outer-graphs/pyproject.toml; \\
    done
# -- End of non-package dependency graphs --
# -- Installing all local dependencies --
RUN for dep in /deps/*; do             echo "Installing $dep";             if [ -d "$dep" ]; then                 echo "Installing $dep";                 (cd "$dep" && PIP_CONFIG_FILE=/pipconfig.txt PYTHONDONTWRITEBYTECODE=1 uv pip install --system --no-cache-dir -c /api/constraints.txt -e .);             fi;         done
# -- End of local dependencies install --
ENV LANGSERVE_GRAPHS='{{"agent": "/deps/outer-graphs/src/agent.py:graph"}}'
{FORMATTED_CLEANUP_LINES}"""
    assert clean_empty_lines(actual_docker_stdin) == expected_docker_stdin
    assert additional_contexts == {}


# node.js build used for LangSmith Deployment
def test_config_to_docker_nodejs():
    graphs = {"agent": "./graphs/agent.js:graph"}
    actual_docker_stdin, additional_contexts = config_to_docker(
        PATH_TO_CONFIG,
        validate_config(
            {
                "node_version": "20",
                "graphs": graphs,
                "dockerfile_lines": ["ARG meow", "ARG foo"],
                "auth": {"path": "./graphs/auth.mts:auth"},
                "ui": {"agent": "./graphs/agent.ui.jsx"},
                "ui_config": {"shared": ["nuqs"]},
            }
        ),
        "langchain/langgraphjs-api",
    )
    expected_docker_stdin = """FROM langchain/langgraphjs-api:20
ARG meow
ARG foo
ADD . /deps/unit_tests
RUN cd /deps/unit_tests && npm i
ENV LANGGRAPH_AUTH='{"path": "./graphs/auth.mts:auth"}'
ENV LANGGRAPH_UI='{"agent": "./graphs/agent.ui.jsx"}'
ENV LANGGRAPH_UI_CONFIG='{"shared": ["nuqs"]}'
ENV LANGSERVE_GRAPHS='{"agent": "./graphs/agent.js:graph"}'
WORKDIR /deps/unit_tests
RUN (test ! -f /api/langgraph_api/js/build.mts && echo "Prebuild script not found, skipping") || tsx /api/langgraph_api/js/build.mts"""

    assert clean_empty_lines(actual_docker_stdin) == expected_docker_stdin
    assert additional_contexts == {}


def test_config_to_docker_nodejs_internal_docker_tag():
    graphs = {"agent": "./graphs/agent.js:graph"}
    actual_docker_stdin, additional_contexts = config_to_docker(
        PATH_TO_CONFIG,
        validate_config(
            {
                "node_version": "20",
                "graphs": graphs,
                "dockerfile_lines": ["ARG meow", "ARG foo"],
                "auth": {"path": "./graphs/auth.mts:auth"},
                "ui": {"agent": "./graphs/agent.ui.jsx"},
                "ui_config": {"shared": ["nuqs"]},
                "_INTERNAL_docker_tag": "my-tag",
            }
        ),
        "langchain/langgraphjs-api",
    )
    expected_docker_stdin = """FROM langchain/langgraphjs-api:my-tag
ARG meow
ARG foo
ADD . /deps/unit_tests
RUN cd /deps/unit_tests && npm i
ENV LANGGRAPH_AUTH='{"path": "./graphs/auth.mts:auth"}'
ENV LANGGRAPH_UI='{"agent": "./graphs/agent.ui.jsx"}'
ENV LANGGRAPH_UI_CONFIG='{"shared": ["nuqs"]}'
ENV LANGSERVE_GRAPHS='{"agent": "./graphs/agent.js:graph"}'
WORKDIR /deps/unit_tests
RUN (test ! -f /api/langgraph_api/js/build.mts && echo "Prebuild script not found, skipping") || tsx /api/langgraph_api/js/build.mts"""

    assert clean_empty_lines(actual_docker_stdin) == expected_docker_stdin
    assert additional_contexts == {}


def test_config_to_docker_gen_ui_python():
    graphs = {"agent": "./agent.py:graph"}
    actual_docker_stdin, additional_contexts = config_to_docker(
        PATH_TO_CONFIG,
        validate_config(
            {
                "dependencies": ["."],
                "graphs": graphs,
                "ui": {"agent": "./graphs/agent.ui.jsx"},
                "ui_config": {"shared": ["nuqs"]},
            }
        ),
        "langchain/langgraph-api",
    )

    expected_docker_stdin = f"""FROM langchain/langgraph-api:3.11
RUN /storage/install-node.sh
# -- Adding non-package dependency unit_tests --
ADD . /deps/outer-unit_tests/unit_tests
RUN set -ex && \\
    for line in '[project]' \\
                'name = "unit_tests"' \\
                'version = "0.1"' \\
                '[tool.setuptools.package-data]' \\
                '"*" = ["**/*"]' \\
                '[build-system]' \\
                'requires = ["setuptools>=61"]' \\
                'build-backend = "setuptools.build_meta"'; do \\
        echo "$line" >> /deps/outer-unit_tests/pyproject.toml; \\
    done
# -- End of non-package dependency unit_tests --
# -- Installing all local dependencies --
RUN for dep in /deps/*; do             echo "Installing $dep";             if [ -d "$dep" ]; then                 echo "Installing $dep";                 (cd "$dep" && PYTHONDONTWRITEBYTECODE=1 uv pip install --system --no-cache-dir -c /api/constraints.txt -e .);             fi;         done
# -- End of local dependencies install --
ENV LANGGRAPH_UI='{{"agent": "./graphs/agent.ui.jsx"}}'
ENV LANGGRAPH_UI_CONFIG='{{"shared": ["nuqs"]}}'
ENV LANGSERVE_GRAPHS='{{"agent": "/deps/outer-unit_tests/unit_tests/agent.py:graph"}}'
# -- Installing JS dependencies --
ENV NODE_VERSION=20
RUN cd /deps/outer-unit_tests/unit_tests && npm i && tsx /api/langgraph_api/js/build.mts
# -- End of JS dependencies install --
{FORMATTED_CLEANUP_LINES}
WORKDIR /deps/outer-unit_tests/unit_tests"""

    assert clean_empty_lines(actual_docker_stdin) == expected_docker_stdin
    assert additional_contexts == {}


def test_config_to_docker_multiplatform():
    graphs = {
        "python": "./multiplatform/python.py:graph",
        "js": "./multiplatform/js.mts:graph",
    }
    actual_docker_stdin, additional_contexts = config_to_docker(
        PATH_TO_CONFIG,
        validate_config(
            {"node_version": "22", "dependencies": ["."], "graphs": graphs}
        ),
        "langchain/langgraph-api",
    )

    expected_docker_stdin = f"""FROM langchain/langgraph-api:3.11
RUN /storage/install-node.sh
# -- Adding non-package dependency unit_tests --
ADD . /deps/outer-unit_tests/unit_tests
RUN set -ex && \\
    for line in '[project]' \\
                'name = "unit_tests"' \\
                'version = "0.1"' \\
                '[tool.setuptools.package-data]' \\
                '"*" = ["**/*"]' \\
                '[build-system]' \\
                'requires = ["setuptools>=61"]' \\
                'build-backend = "setuptools.build_meta"'; do \\
        echo "$line" >> /deps/outer-unit_tests/pyproject.toml; \\
    done
# -- End of non-package dependency unit_tests --
# -- Installing all local dependencies --
RUN for dep in /deps/*; do             echo "Installing $dep";             if [ -d "$dep" ]; then                 echo "Installing $dep";                 (cd "$dep" && PYTHONDONTWRITEBYTECODE=1 uv pip install --system --no-cache-dir -c /api/constraints.txt -e .);             fi;         done
# -- End of local dependencies install --
ENV LANGSERVE_GRAPHS='{{"python": "/deps/outer-unit_tests/unit_tests/multiplatform/python.py:graph", "js": "/deps/outer-unit_tests/unit_tests/multiplatform/js.mts:graph"}}'
# -- Installing JS dependencies --
ENV NODE_VERSION=22
RUN cd /deps/outer-unit_tests/unit_tests && npm i && tsx /api/langgraph_api/js/build.mts
# -- End of JS dependencies install --
{FORMATTED_CLEANUP_LINES}
WORKDIR /deps/outer-unit_tests/unit_tests"""

    assert clean_empty_lines(actual_docker_stdin) == expected_docker_stdin
    assert additional_contexts == {}


def test_config_to_docker_pip_installer():
    """Test that pip_installer setting affects the generated Dockerfile."""
    graphs = {"agent": "./graphs/agent.py:graph"}
    base_config = {
        "python_version": "3.11",
        "dependencies": ["."],
        "graphs": graphs,
    }

    # Test default (auto) behavior with UV-supporting image
    config_auto = validate_config(
        {**copy.deepcopy(base_config), "pip_installer": "auto"}
    )
    docker_auto, _ = config_to_docker(
        PATH_TO_CONFIG, config_auto, "langchain/langgraph-api:0.2.47"
    )
    assert "uv pip install --system " in docker_auto
    assert "rm /usr/bin/uv /usr/bin/uvx" in docker_auto

    # Test explicit pip setting
    config_pip = validate_config({**copy.deepcopy(base_config), "pip_installer": "pip"})
    docker_pip, _ = config_to_docker(
        PATH_TO_CONFIG, config_pip, "langchain/langgraph-api:0.2.47"
    )
    assert "uv pip install --system " not in docker_pip
    assert "pip install" in docker_pip
    assert "rm /usr/bin/uv" not in docker_pip

    # Test explicit uv setting
    config_uv = validate_config({**copy.deepcopy(base_config), "pip_installer": "uv"})
    docker_uv, _ = config_to_docker(
        PATH_TO_CONFIG, config_uv, "langchain/langgraph-api:0.2.47"
    )
    assert "uv pip install --system " in docker_uv
    assert "rm /usr/bin/uv /usr/bin/uvx" in docker_uv

    # Test auto behavior with older image (should use pip)
    config_auto_old = validate_config(
        {**copy.deepcopy(base_config), "pip_installer": "auto"}
    )
    docker_auto_old, _ = config_to_docker(
        PATH_TO_CONFIG, config_auto_old, "langchain/langgraph-api:0.2.46"
    )
    assert "uv pip install --system " not in docker_auto_old
    assert "pip install" in docker_auto_old
    assert "rm /usr/bin/uv" not in docker_auto_old

    # Test that missing pip_installer defaults to auto behavior
    config_default = validate_config(copy.deepcopy(base_config))
    docker_default, _ = config_to_docker(
        PATH_TO_CONFIG, config_default, "langchain/langgraph-api:0.2.47"
    )
    assert "uv pip install --system " in docker_default


def test_config_retain_build_tools():
    graphs = {"agent": "./graphs/agent.py:graph"}
    base_config = {
        "python_version": "3.11",
        "dependencies": ["."],
        "graphs": graphs,
    }
    config_true = validate_config(
        {**copy.deepcopy(base_config), "keep_pkg_tools": True}
    )
    docker_true, _ = config_to_docker(
        PATH_TO_CONFIG, config_true, "langchain/langgraph-api:0.2.47"
    )
    assert not any(
        "/usr/local/lib/python*/site-packages/" + pckg + "*" in docker_true
        for pckg in _BUILD_TOOLS
    )
    assert "RUN pip uninstall -y pip setuptools wheel" not in docker_true
    config_false = validate_config(
        {**copy.deepcopy(base_config), "keep_pkg_tools": False}
    )
    docker_false, _ = config_to_docker(
        PATH_TO_CONFIG, config_false, "langchain/langgraph-api:0.2.47"
    )
    assert all(
        "/usr/local/lib/python*/site-packages/" + pckg + "*" in docker_false
        for pckg in _BUILD_TOOLS
    )
    assert "RUN pip uninstall -y pip setuptools wheel" in docker_false
    config_list = validate_config(
        {**copy.deepcopy(base_config), "keep_pkg_tools": ["pip", "setuptools"]}
    )
    docker_list, _ = config_to_docker(
        PATH_TO_CONFIG, config_list, "langchain/langgraph-api:0.2.47"
    )
    assert all(
        "/usr/local/lib/python*/site-packages/" + pckg + "*" in docker_list
        for pckg in ("wheel",)
    )
    assert not any(
        "/usr/local/lib/python*/site-packages/" + pckg + "*" in docker_list
        for pckg in ("pip", "setuptools")
    )
    assert "RUN pip uninstall -y wheel" in docker_list
    assert "RUN pip uninstall -y pip setuptools" not in docker_list


# config_to_compose
def test_config_to_compose_simple_config():
    graphs = {"agent": "./agent.py:graph"}
    # Create a properly indented version of FORMATTED_CLEANUP_LINES for compose files
    expected_compose_stdin = f"""
        pull_policy: build
        build:
            context: .
            dockerfile_inline: |
                FROM langchain/langgraph-api:3.11
                # -- Adding non-package dependency unit_tests --
                ADD . /deps/outer-unit_tests/unit_tests
                RUN set -ex && \\
                    for line in '[project]' \\
                                'name = "unit_tests"' \\
                                'version = "0.1"' \\
                                '[tool.setuptools.package-data]' \\
                                '"*" = ["**/*"]' \\
                                '[build-system]' \\
                                'requires = ["setuptools>=61"]' \\
                                'build-backend = "setuptools.build_meta"'; do \\
                        echo "$line" >> /deps/outer-unit_tests/pyproject.toml; \\
                    done
                # -- End of non-package dependency unit_tests --
                # -- Installing all local dependencies --
                RUN for dep in /deps/*; do             echo "Installing $dep";             if [ -d "$dep" ]; then                 echo "Installing $dep";                 (cd "$dep" && PYTHONDONTWRITEBYTECODE=1 uv pip install --system --no-cache-dir -c /api/constraints.txt -e .);             fi;         done
                # -- End of local dependencies install --
                ENV LANGSERVE_GRAPHS='{{"agent": "/deps/outer-unit_tests/unit_tests/agent.py:graph"}}'
{textwrap.indent(textwrap.dedent(FORMATTED_CLEANUP_LINES), "                ")}
                WORKDIR /deps/outer-unit_tests/unit_tests
        """
    actual_compose_stdin = config_to_compose(
        PATH_TO_CONFIG,
        validate_config({"dependencies": ["."], "graphs": graphs}),
        "langchain/langgraph-api",
    )
    assert (
        clean_empty_lines(actual_compose_stdin).strip()
        == expected_compose_stdin.strip()
    )


def test_config_to_compose_env_vars():
    graphs = {"agent": "./agent.py:graph"}
    expected_compose_stdin = f"""                        OPENAI_API_KEY: "key"
        
        pull_policy: build
        build:
            context: .
            dockerfile_inline: |
                FROM langchain/langgraph-api-custom:3.11
                # -- Adding non-package dependency unit_tests --
                ADD . /deps/outer-unit_tests/unit_tests
                RUN set -ex && \\
                    for line in '[project]' \\
                                'name = "unit_tests"' \\
                                'version = "0.1"' \\
                                '[tool.setuptools.package-data]' \\
                                '"*" = ["**/*"]' \\
                                '[build-system]' \\
                                'requires = ["setuptools>=61"]' \\
                                'build-backend = "setuptools.build_meta"'; do \\
                        echo "$line" >> /deps/outer-unit_tests/pyproject.toml; \\
                    done
                # -- End of non-package dependency unit_tests --
                # -- Installing all local dependencies --
                RUN for dep in /deps/*; do             echo "Installing $dep";             if [ -d "$dep" ]; then                 echo "Installing $dep";                 (cd "$dep" && PYTHONDONTWRITEBYTECODE=1 uv pip install --system --no-cache-dir -c /api/constraints.txt -e .);             fi;         done
                # -- End of local dependencies install --
                ENV LANGSERVE_GRAPHS='{{"agent": "/deps/outer-unit_tests/unit_tests/agent.py:graph"}}'
{textwrap.indent(textwrap.dedent(FORMATTED_CLEANUP_LINES), "                ")}
                WORKDIR /deps/outer-unit_tests/unit_tests
        """
    openai_api_key = "key"
    actual_compose_stdin = config_to_compose(
        PATH_TO_CONFIG,
        validate_config(
            {
                "dependencies": ["."],
                "graphs": graphs,
                "env": {"OPENAI_API_KEY": openai_api_key},
            }
        ),
        "langchain/langgraph-api-custom",
    )
    assert clean_empty_lines(actual_compose_stdin) == expected_compose_stdin


def test_config_to_compose_env_file():
    graphs = {"agent": "./agent.py:graph"}
    expected_compose_stdin = f"""\
        env_file: .env
        pull_policy: build
        build:
            context: .
            dockerfile_inline: |
                FROM langchain/langgraph-api:3.11
                # -- Adding non-package dependency unit_tests --
                ADD . /deps/outer-unit_tests/unit_tests
                RUN set -ex && \\
                    for line in '[project]' \\
                                'name = "unit_tests"' \\
                                'version = "0.1"' \\
                                '[tool.setuptools.package-data]' \\
                                '"*" = ["**/*"]' \\
                                '[build-system]' \\
                                'requires = ["setuptools>=61"]' \\
                                'build-backend = "setuptools.build_meta"'; do \\
                        echo "$line" >> /deps/outer-unit_tests/pyproject.toml; \\
                    done
                # -- End of non-package dependency unit_tests --
                # -- Installing all local dependencies --
                RUN for dep in /deps/*; do             echo "Installing $dep";             if [ -d "$dep" ]; then                 echo "Installing $dep";                 (cd "$dep" && PYTHONDONTWRITEBYTECODE=1 uv pip install --system --no-cache-dir -c /api/constraints.txt -e .);             fi;         done
                # -- End of local dependencies install --
                ENV LANGSERVE_GRAPHS='{{"agent": "/deps/outer-unit_tests/unit_tests/agent.py:graph"}}'
{textwrap.indent(textwrap.dedent(FORMATTED_CLEANUP_LINES), "                ")}
                WORKDIR /deps/outer-unit_tests/unit_tests
        """
    actual_compose_stdin = config_to_compose(
        PATH_TO_CONFIG,
        validate_config({"dependencies": ["."], "graphs": graphs, "env": ".env"}),
        "langchain/langgraph-api",
    )
    assert clean_empty_lines(actual_compose_stdin) == expected_compose_stdin


def test_config_to_compose_watch():
    graphs = {"agent": "./agent.py:graph"}
    expected_compose_stdin = f"""\
        
        pull_policy: build
        build:
            context: .
            dockerfile_inline: |
                FROM langchain/langgraph-api:3.11
                # -- Adding non-package dependency unit_tests --
                ADD . /deps/outer-unit_tests/unit_tests
                RUN set -ex && \\
                    for line in '[project]' \\
                                'name = "unit_tests"' \\
                                'version = "0.1"' \\
                                '[tool.setuptools.package-data]' \\
                                '"*" = ["**/*"]' \\
                                '[build-system]' \\
                                'requires = ["setuptools>=61"]' \\
                                'build-backend = "setuptools.build_meta"'; do \\
                        echo "$line" >> /deps/outer-unit_tests/pyproject.toml; \\
                    done
                # -- End of non-package dependency unit_tests --
                # -- Installing all local dependencies --
                RUN for dep in /deps/*; do             echo "Installing $dep";             if [ -d "$dep" ]; then                 echo "Installing $dep";                 (cd "$dep" && PYTHONDONTWRITEBYTECODE=1 uv pip install --system --no-cache-dir -c /api/constraints.txt -e .);             fi;         done
                # -- End of local dependencies install --
                ENV LANGSERVE_GRAPHS='{{"agent": "/deps/outer-unit_tests/unit_tests/agent.py:graph"}}'
{textwrap.indent(textwrap.dedent(FORMATTED_CLEANUP_LINES), "                ")}
                WORKDIR /deps/outer-unit_tests/unit_tests
        
        develop:
            watch:
                - path: test_config.json
                  action: rebuild
                - path: .
                  action: rebuild\
"""
    actual_compose_stdin = config_to_compose(
        PATH_TO_CONFIG,
        validate_config({"dependencies": ["."], "graphs": graphs}),
        "langchain/langgraph-api",
        watch=True,
    )
    assert clean_empty_lines(actual_compose_stdin) == expected_compose_stdin


def test_config_to_compose_end_to_end():
    # test all of the above + langgraph API path
    graphs = {"agent": "./agent.py:graph"}
    expected_compose_stdin = f"""\
        env_file: .env
        pull_policy: build
        build:
            context: .
            dockerfile_inline: |
                FROM langchain/langgraph-api:3.11
                # -- Adding non-package dependency unit_tests --
                ADD . /deps/outer-unit_tests/unit_tests
                RUN set -ex && \\
                    for line in '[project]' \\
                                'name = "unit_tests"' \\
                                'version = "0.1"' \\
                                '[tool.setuptools.package-data]' \\
                                '"*" = ["**/*"]' \\
                                '[build-system]' \\
                                'requires = ["setuptools>=61"]' \\
                                'build-backend = "setuptools.build_meta"'; do \\
                        echo "$line" >> /deps/outer-unit_tests/pyproject.toml; \\
                    done
                # -- End of non-package dependency unit_tests --
                # -- Installing all local dependencies --
                RUN for dep in /deps/*; do             echo "Installing $dep";             if [ -d "$dep" ]; then                 echo "Installing $dep";                 (cd "$dep" && PYTHONDONTWRITEBYTECODE=1 uv pip install --system --no-cache-dir -c /api/constraints.txt -e .);             fi;         done
                # -- End of local dependencies install --
                ENV LANGSERVE_GRAPHS='{{"agent": "/deps/outer-unit_tests/unit_tests/agent.py:graph"}}'
{textwrap.indent(textwrap.dedent(FORMATTED_CLEANUP_LINES), "                ")}
                WORKDIR /deps/outer-unit_tests/unit_tests
        
        develop:
            watch:
                - path: test_config.json
                  action: rebuild
                - path: .
                  action: rebuild\
"""
    actual_compose_stdin = config_to_compose(
        PATH_TO_CONFIG,
        validate_config({"dependencies": ["."], "graphs": graphs, "env": ".env"}),
        "langchain/langgraph-api",
        watch=True,
    )
    assert clean_empty_lines(actual_compose_stdin) == expected_compose_stdin


def test_docker_tag_image_distro():
    """Test docker_tag function with different image_distro configurations."""

    # Test 1: Default distro (debian) - no suffix
    config = validate_config(
        {
            "python_version": "3.11",
            "dependencies": ["."],
            "graphs": {"agent": "./agent.py:graph"},
        }
    )
    tag = docker_tag(config)
    assert tag == "langchain/langgraph-api:3.11"

    # Test 2: Explicit debian distro - no suffix (same as default)
    config = validate_config(
        {
            "python_version": "3.11",
            "dependencies": ["."],
            "graphs": {"agent": "./agent.py:graph"},
            "image_distro": "debian",
        }
    )
    tag = docker_tag(config)
    assert tag == "langchain/langgraph-api:3.11"

    # Test 3: Wolfi distro - should add suffix
    config = validate_config(
        {
            "python_version": "3.11",
            "dependencies": ["."],
            "graphs": {"agent": "./agent.py:graph"},
            "image_distro": "wolfi",
        }
    )
    tag = docker_tag(config)
    assert tag == "langchain/langgraph-api:3.11-wolfi"

    # Test 4: Node.js with default distro
    config = validate_config(
        {
            "node_version": "20",
            "graphs": {"agent": "./agent.js:graph"},
        }
    )
    tag = docker_tag(config)
    assert tag == "langchain/langgraphjs-api:20"

    # Test 5: Node.js with wolfi distro
    config = validate_config(
        {
            "node_version": "20",
            "graphs": {"agent": "./agent.js:graph"},
            "image_distro": "wolfi",
        }
    )
    tag = docker_tag(config)
    assert tag == "langchain/langgraphjs-api:20-wolfi"

    # Test 6: Custom base image with wolfi
    config = validate_config(
        {
            "python_version": "3.12",
            "dependencies": ["."],
            "graphs": {"agent": "./agent.py:graph"},
            "image_distro": "wolfi",
            "base_image": "my-registry/custom-image",
        }
    )
    tag = docker_tag(config, base_image="my-registry/custom-image")
    assert tag == "my-registry/custom-image:3.12-wolfi"


def test_docker_tag_multiplatform_with_distro():
    """Test docker_tag with multiplatform configs and image_distro."""

    # Test 1: Multiplatform (Python + Node) with wolfi
    config = validate_config(
        {
            "python_version": "3.11",
            "node_version": "20",
            "dependencies": ["."],
            "graphs": {"python": "./agent.py:graph", "js": "./agent.js:graph"},
            "image_distro": "wolfi",
        }
    )
    tag = docker_tag(config)
    # Should default to Python when both are present
    assert tag == "langchain/langgraph-api:3.11-wolfi"

    # Test 2: Node-only multiplatform with wolfi
    config = validate_config(
        {
            "node_version": "20",
            "graphs": {"js": "./agent.js:graph"},
            "image_distro": "wolfi",
        }
    )
    tag = docker_tag(config)
    assert tag == "langchain/langgraphjs-api:20-wolfi"


def test_docker_tag_different_python_versions_with_distro():
    """Test docker_tag with different Python versions and distros."""

    versions_and_expected = [
        ("3.11", "langchain/langgraph-api:3.11-wolfi"),
        ("3.12", "langchain/langgraph-api:3.12-wolfi"),
        ("3.13", "langchain/langgraph-api:3.13-wolfi"),
    ]

    for python_version, expected_tag in versions_and_expected:
        config = validate_config(
            {
                "python_version": python_version,
                "dependencies": ["."],
                "graphs": {"agent": "./agent.py:graph"},
                "image_distro": "wolfi",
            }
        )
        tag = docker_tag(config)
        assert tag == expected_tag, f"Failed for Python {python_version}"


def test_docker_tag_different_node_versions_with_distro():
    """Test docker_tag with different Node.js versions and distros."""

    versions_and_expected = [
        ("20", "langchain/langgraphjs-api:20-wolfi"),
        ("21", "langchain/langgraphjs-api:21-wolfi"),
        ("22", "langchain/langgraphjs-api:22-wolfi"),
    ]

    for node_version, expected_tag in versions_and_expected:
        config = validate_config(
            {
                "node_version": node_version,
                "graphs": {"agent": "./agent.js:graph"},
                "image_distro": "wolfi",
            }
        )
        tag = docker_tag(config)
        assert tag == expected_tag, f"Failed for Node.js {node_version}"


@pytest.mark.parametrize("in_config", [False, True])
def test_docker_tag_with_api_version(in_config: bool):
    """Test docker_tag function with api_version parameter."""

    # Test 1: Python config with api_version and default distro
    version = "0.2.74"
    config = validate_config(
        {
            "python_version": "3.11",
            "dependencies": ["."],
            "graphs": {"agent": "./agent.py:graph"},
            "api_version": version if in_config else None,
        }
    )
    tag = docker_tag(config, api_version=version if not in_config else None)
    assert tag == f"langchain/langgraph-api:{version}-py3.11"

    # Test 2: Python config with api_version and wolfi distro
    config = validate_config(
        {
            "python_version": "3.12",
            "dependencies": ["."],
            "graphs": {"agent": "./agent.py:graph"},
            "image_distro": "wolfi",
            "api_version": version if in_config else None,
        }
    )
    tag = docker_tag(config, api_version=version if not in_config else None)
    assert tag == f"langchain/langgraph-api:{version}-py3.12-wolfi"

    # Test 3: Node.js config with api_version and default distro
    config = validate_config(
        {
            "node_version": "20",
            "graphs": {"agent": "./agent.js:graph"},
            "api_version": version if in_config else None,
        }
    )
    tag = docker_tag(config, api_version=version if not in_config else None)
    assert tag == f"langchain/langgraphjs-api:{version}-node20"

    # Test 4: Node.js config with api_version and wolfi distro
    config = validate_config(
        {
            "node_version": "20",
            "graphs": {"agent": "./agent.js:graph"},
            "image_distro": "wolfi",
            "api_version": version if in_config else None,
        }
    )
    tag = docker_tag(config, api_version=version if not in_config else None)
    assert tag == f"langchain/langgraphjs-api:{version}-node20-wolfi"

    # Test 5: Custom base image with api_version
    config = validate_config(
        {
            "python_version": "3.11",
            "dependencies": ["."],
            "graphs": {"agent": "./agent.py:graph"},
            "base_image": "my-registry/custom-image",
            "api_version": version if in_config else None,
        }
    )
    tag = docker_tag(
        config,
        base_image="my-registry/custom-image",
        api_version=version if not in_config else None,
    )
    assert tag == f"my-registry/custom-image:{version}-py3.11"

    # Test 6: api_version with different Python versions
    for python_version in ["3.11", "3.12", "3.13"]:
        config = validate_config(
            {
                "python_version": python_version,
                "dependencies": ["."],
                "graphs": {"agent": "./agent.py:graph"},
                "api_version": version if in_config else None,
            }
        )
        tag = docker_tag(config, api_version=version if not in_config else None)
        assert tag == f"langchain/langgraph-api:{version}-py{python_version}"

    # Test 7: Without api_version should work as before
    config = validate_config(
        {
            "python_version": "3.11",
            "dependencies": ["."],
            "graphs": {"agent": "./agent.py:graph"},
        }
    )
    tag = docker_tag(config)
    assert tag == "langchain/langgraph-api:3.11"

    # Test 8: api_version with multiplatform config (should default to Python)
    config = validate_config(
        {
            "python_version": "3.11",
            "node_version": "20",
            "dependencies": ["."],
            "graphs": {"python": "./agent.py:graph", "js": "./agent.js:graph"},
            "api_version": version if in_config else None,
        }
    )
    tag = docker_tag(config, api_version=version if not in_config else None)
    assert tag == f"langchain/langgraph-api:{version}-py3.11"

    # Test 9: api_version with _INTERNAL_docker_tag should ignore api_version
    config = validate_config(
        {
            "python_version": "3.11",
            "dependencies": ["."],
            "graphs": {"agent": "./agent.py:graph"},
            "_INTERNAL_docker_tag": "internal-tag",
        }
    )
    tag = docker_tag(config, api_version="0.2.74")
    assert tag == "langchain/langgraph-api:internal-tag"

    # Test 10: api_version with langgraph-server base image should follow special format
    config = validate_config(
        {
            "python_version": "3.11",
            "dependencies": ["."],
            "graphs": {"agent": "./agent.py:graph"},
            "api_version": version if in_config else None,
        }
    )
    tag = docker_tag(
        config,
        base_image="langchain/langgraph-server",
        api_version=version if not in_config else None,
    )
    assert tag == f"langchain/langgraph-server:{version}-py3.11"


def test_config_to_docker_with_api_version():
    """Test config_to_docker function with api_version parameter."""

    # Test Python config with api_version
    graphs = {"agent": "./agent.py:graph"}
    actual_docker_stdin, additional_contexts = config_to_docker(
        PATH_TO_CONFIG,
        validate_config({"dependencies": ["."], "graphs": graphs}),
        "langchain/langgraph-api",
        api_version="0.2.74",
    )

    # Check that the FROM line uses the api_version
    lines = actual_docker_stdin.split("\n")
    from_line = lines[0]
    assert from_line == "FROM langchain/langgraph-api:0.2.74-py3.11"

    # Test Node.js config with api_version
    graphs = {"agent": "./agent.js:graph"}
    actual_docker_stdin, additional_contexts = config_to_docker(
        PATH_TO_CONFIG,
        validate_config({"node_version": "20", "graphs": graphs}),
        "langchain/langgraphjs-api",
        api_version="0.2.74",
    )

    # Check that the FROM line uses the api_version
    lines = actual_docker_stdin.split("\n")
    from_line = lines[0]
    assert from_line == "FROM langchain/langgraphjs-api:0.2.74-node20"


def test_config_to_compose_with_api_version():
    """Test config_to_compose function with api_version parameter."""

    # Test Python config with api_version
    config = validate_config(
        {
            "dependencies": ["."],
            "graphs": {"agent": "./agent.py:graph"},
        }
    )

    actual_compose_str = config_to_compose(
        PATH_TO_CONFIG,
        config,
        "langchain/langgraph-api",
        api_version="0.2.74",
    )

    # Check that the compose file includes the correct FROM line with api_version
    assert "FROM langchain/langgraph-api:0.2.74-py3.11" in actual_compose_str

    # Test Node.js config with api_version
    config = validate_config(
        {
            "node_version": "20",
            "graphs": {"agent": "./agent.js:graph"},
        }
    )

    actual_compose_str = config_to_compose(
        PATH_TO_CONFIG,
        config,
        "langchain/langgraphjs-api",
        api_version="0.2.74",
    )

    # Check that the compose file includes the correct FROM line with api_version
    assert "FROM langchain/langgraphjs-api:0.2.74-node20" in actual_compose_str
