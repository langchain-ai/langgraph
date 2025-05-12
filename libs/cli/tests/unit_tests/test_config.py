import json
import os
import pathlib
import tempfile
import textwrap

import click
import pytest

from langgraph_cli.config import (
    PIP_CLEANUP_LINES,
    config_to_compose,
    config_to_docker,
    validate_config,
    validate_config_file,
)
from langgraph_cli.util import clean_empty_lines

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
        "_INTERNAL_docker_tag": None,
        "base_image": None,
        "python_version": "3.11",
        "node_version": None,
        "pip_config_file": None,
        "dockerfile_lines": [],
        "env": {},
        "store": None,
        "auth": None,
        "checkpointer": None,
        "http": None,
        "ui": None,
        "ui_config": None,
        **expected_config,
    }
    assert actual_config == expected_config

    # full config
    env = ".env"
    expected_config = {
        "_INTERNAL_docker_tag": None,
        "base_image": None,
        "python_version": "3.12",
        "node_version": None,
        "pip_config_file": "pipconfig.txt",
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
    with pytest.raises(
        ValueError,
        match="Invalid http.app format",
    ):
        validate_config(
            {
                "python_version": "3.12",
                "dependencies": ["."],
                "graphs": {"agent": "./agent.py:graph"},
                "http": {"app": "../../examples/my_app.py"},
            }
        )


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
FROM langchain/langgraph-api:3.11
# -- Installing local requirements --
COPY --from=__outer_requirements.txt requirements.txt /deps/__outer_graphs_reqs_a/graphs_reqs_a/requirements.txt
RUN PYTHONDONTWRITEBYTECODE=1 pip install --no-cache-dir -c /api/constraints.txt -r /deps/__outer_graphs_reqs_a/graphs_reqs_a/requirements.txt
# -- End of local requirements install --
# -- Adding local package ../../examples --
COPY --from=examples . /deps/examples
# -- End of local package ../../examples --
# -- Adding non-package dependency unit_tests --
ADD . /deps/__outer_unit_tests/unit_tests
RUN set -ex && \\
    for line in '[project]' \\
                'name = "unit_tests"' \\
                'version = "0.1"' \\
                '[tool.setuptools.package-data]' \\
                '"*" = ["**/*"]'; do \\
        echo "$line" >> /deps/__outer_unit_tests/pyproject.toml; \\
    done
# -- End of non-package dependency unit_tests --
# -- Adding non-package dependency graphs_reqs_a --
COPY --from=__outer_graphs_reqs_a . /deps/__outer_graphs_reqs_a/graphs_reqs_a
RUN set -ex && \\
    for line in '[project]' \\
                'name = "graphs_reqs_a"' \\
                'version = "0.1"' \\
                '[tool.setuptools.package-data]' \\
                '"*" = ["**/*"]'; do \\
        echo "$line" >> /deps/__outer_graphs_reqs_a/pyproject.toml; \\
    done
# -- End of non-package dependency graphs_reqs_a --
# -- Installing all local dependencies --
RUN PYTHONDONTWRITEBYTECODE=1 pip install --no-cache-dir -c /api/constraints.txt -e /deps/*
# -- End of local dependencies install --
ENV LANGGRAPH_HTTP='{{"app": "/deps/examples/my_app.py:app"}}'
ENV LANGSERVE_GRAPHS='{{"agent": "/deps/__outer_unit_tests/unit_tests/agent.py:graph"}}'
{PIP_CLEANUP_LINES}
WORKDIR /deps/__outer_unit_tests/unit_tests\
"""
    assert clean_empty_lines(actual_docker_stdin) == expected_docker_stdin

    assert additional_contexts == {
        "__outer_graphs_reqs_a": str(
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
FROM langchain/langgraph-api:3.11
# -- Adding non-package dependency unit_tests --
ADD . /deps/__outer_unit_tests/unit_tests
RUN set -ex && \\
    for line in '[project]' \\
                'name = "unit_tests"' \\
                'version = "0.1"' \\
                '[tool.setuptools.package-data]' \\
                '"*" = ["**/*"]'; do \\
        echo "$line" >> /deps/__outer_unit_tests/pyproject.toml; \\
    done
# -- End of non-package dependency unit_tests --
# -- Adding non-package dependency tests --
COPY --from=__outer_tests . /deps/__outer_tests/tests
RUN set -ex && \\
    for line in '[project]' \\
                'name = "tests"' \\
                'version = "0.1"' \\
                '[tool.setuptools.package-data]' \\
                '"*" = ["**/*"]'; do \\
        echo "$line" >> /deps/__outer_tests/pyproject.toml; \\
    done
# -- End of non-package dependency tests --
# -- Installing all local dependencies --
RUN PYTHONDONTWRITEBYTECODE=1 pip install --no-cache-dir -c /api/constraints.txt -e /deps/*
# -- End of local dependencies install --
ENV LANGSERVE_GRAPHS='{"agent": "/deps/__outer_unit_tests/unit_tests/agent.py:graph"}'
"""
        + PIP_CLEANUP_LINES
        + """
WORKDIR /deps/__outer_unit_tests/unit_tests\
"""
    )
    assert clean_empty_lines(actual_docker_stdin) == expected_docker_stdin
    assert additional_contexts == {
        "__outer_tests": str(pathlib.Path(__file__).parent.parent.absolute()),
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
ADD . /deps/__outer_unit_tests/unit_tests
RUN set -ex && \\
    for line in '[project]' \\
                'name = "unit_tests"' \\
                'version = "0.1"' \\
                '[tool.setuptools.package-data]' \\
                '"*" = ["**/*"]'; do \\
        echo "$line" >> /deps/__outer_unit_tests/pyproject.toml; \\
    done
# -- End of non-package dependency unit_tests --
# -- Installing all local dependencies --
RUN PIP_CONFIG_FILE=/pipconfig.txt PYTHONDONTWRITEBYTECODE=1 pip install --no-cache-dir -c /api/constraints.txt -e /deps/*
# -- End of local dependencies install --
ENV LANGSERVE_GRAPHS='{"agent": "/deps/__outer_unit_tests/unit_tests/agent.py:graph"}'
"""
        + PIP_CLEANUP_LINES
        + """
WORKDIR /deps/__outer_unit_tests/unit_tests\
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
ADD ./graphs /deps/__outer_graphs/src
RUN set -ex && \\
    for line in '[project]' \\
                'name = "graphs"' \\
                'version = "0.1"' \\
                '[tool.setuptools.package-data]' \\
                '"*" = ["**/*"]'; do \\
        echo "$line" >> /deps/__outer_graphs/pyproject.toml; \\
    done
# -- End of non-package dependency graphs --
# -- Installing all local dependencies --
RUN PYTHONDONTWRITEBYTECODE=1 pip install --no-cache-dir -c /api/constraints.txt -e /deps/*
# -- End of local dependencies install --
ENV LANGSERVE_GRAPHS='{{"agent": "/deps/__outer_graphs/src/agent.py:graph"}}'
{PIP_CLEANUP_LINES}\
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
RUN PYTHONDONTWRITEBYTECODE=1 pip install --no-cache-dir -c /api/constraints.txt -e /deps/*
# -- End of local dependencies install --
ENV LANGSERVE_GRAPHS='{"agent": "/deps/unit_tests/graphs/agent.py:graph"}'
"""
        + PIP_CLEANUP_LINES
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
RUN PIP_CONFIG_FILE=/pipconfig.txt PYTHONDONTWRITEBYTECODE=1 pip install --no-cache-dir -c /api/constraints.txt langchain langchain_openai
# -- Adding non-package dependency graphs --
ADD ./graphs/ /deps/__outer_graphs/src
RUN set -ex && \\
    for line in '[project]' \\
                'name = "graphs"' \\
                'version = "0.1"' \\
                '[tool.setuptools.package-data]' \\
                '"*" = ["**/*"]'; do \\
        echo "$line" >> /deps/__outer_graphs/pyproject.toml; \\
    done
# -- End of non-package dependency graphs --
# -- Installing all local dependencies --
RUN PIP_CONFIG_FILE=/pipconfig.txt PYTHONDONTWRITEBYTECODE=1 pip install --no-cache-dir -c /api/constraints.txt -e /deps/*
# -- End of local dependencies install --
ENV LANGSERVE_GRAPHS='{{"agent": "/deps/__outer_graphs/src/agent.py:graph"}}'
{PIP_CLEANUP_LINES}"""
    assert clean_empty_lines(actual_docker_stdin) == expected_docker_stdin
    assert additional_contexts == {}


# node.js build used for LangGraph Cloud
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
ADD . /deps/__outer_unit_tests/unit_tests
RUN set -ex && \\
    for line in '[project]' \\
                'name = "unit_tests"' \\
                'version = "0.1"' \\
                '[tool.setuptools.package-data]' \\
                '"*" = ["**/*"]'; do \\
        echo "$line" >> /deps/__outer_unit_tests/pyproject.toml; \\
    done
# -- End of non-package dependency unit_tests --
# -- Installing all local dependencies --
RUN PYTHONDONTWRITEBYTECODE=1 pip install --no-cache-dir -c /api/constraints.txt -e /deps/*
# -- End of local dependencies install --
ENV LANGGRAPH_UI='{{"agent": "./graphs/agent.ui.jsx"}}'
ENV LANGGRAPH_UI_CONFIG='{{"shared": ["nuqs"]}}'
ENV LANGSERVE_GRAPHS='{{"agent": "/deps/__outer_unit_tests/unit_tests/agent.py:graph"}}'
# -- Installing JS dependencies --
ENV NODE_VERSION=20
RUN cd /deps/__outer_unit_tests/unit_tests && npm i && tsx /api/langgraph_api/js/build.mts
# -- End of JS dependencies install --
{PIP_CLEANUP_LINES}
WORKDIR /deps/__outer_unit_tests/unit_tests"""

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
ADD . /deps/__outer_unit_tests/unit_tests
RUN set -ex && \\
    for line in '[project]' \\
                'name = "unit_tests"' \\
                'version = "0.1"' \\
                '[tool.setuptools.package-data]' \\
                '"*" = ["**/*"]'; do \\
        echo "$line" >> /deps/__outer_unit_tests/pyproject.toml; \\
    done
# -- End of non-package dependency unit_tests --
# -- Installing all local dependencies --
RUN PYTHONDONTWRITEBYTECODE=1 pip install --no-cache-dir -c /api/constraints.txt -e /deps/*
# -- End of local dependencies install --
ENV LANGSERVE_GRAPHS='{{"python": "/deps/__outer_unit_tests/unit_tests/multiplatform/python.py:graph", "js": "/deps/__outer_unit_tests/unit_tests/multiplatform/js.mts:graph"}}'
# -- Installing JS dependencies --
ENV NODE_VERSION=22
RUN cd /deps/__outer_unit_tests/unit_tests && npm i && tsx /api/langgraph_api/js/build.mts
# -- End of JS dependencies install --
{PIP_CLEANUP_LINES}
WORKDIR /deps/__outer_unit_tests/unit_tests"""

    assert clean_empty_lines(actual_docker_stdin) == expected_docker_stdin
    assert additional_contexts == {}


# config_to_compose
def test_config_to_compose_simple_config():
    graphs = {"agent": "./agent.py:graph"}
    # Create a properly indented version of PIP_CLEANUP_LINES for compose files
    expected_compose_stdin = f"""
        pull_policy: build
        build:
            context: .
            dockerfile_inline: |
                FROM langchain/langgraph-api:3.11
                # -- Adding non-package dependency unit_tests --
                ADD . /deps/__outer_unit_tests/unit_tests
                RUN set -ex && \\
                    for line in '[project]' \\
                                'name = "unit_tests"' \\
                                'version = "0.1"' \\
                                '[tool.setuptools.package-data]' \\
                                '"*" = ["**/*"]'; do \\
                        echo "$line" >> /deps/__outer_unit_tests/pyproject.toml; \\
                    done
                # -- End of non-package dependency unit_tests --
                # -- Installing all local dependencies --
                RUN PYTHONDONTWRITEBYTECODE=1 pip install --no-cache-dir -c /api/constraints.txt -e /deps/*
                # -- End of local dependencies install --
                ENV LANGSERVE_GRAPHS='{{"agent": "/deps/__outer_unit_tests/unit_tests/agent.py:graph"}}'
{textwrap.indent(textwrap.dedent(PIP_CLEANUP_LINES), "                ")}
                WORKDIR /deps/__outer_unit_tests/unit_tests
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
                ADD . /deps/__outer_unit_tests/unit_tests
                RUN set -ex && \\
                    for line in '[project]' \\
                                'name = "unit_tests"' \\
                                'version = "0.1"' \\
                                '[tool.setuptools.package-data]' \\
                                '"*" = ["**/*"]'; do \\
                        echo "$line" >> /deps/__outer_unit_tests/pyproject.toml; \\
                    done
                # -- End of non-package dependency unit_tests --
                # -- Installing all local dependencies --
                RUN PYTHONDONTWRITEBYTECODE=1 pip install --no-cache-dir -c /api/constraints.txt -e /deps/*
                # -- End of local dependencies install --
                ENV LANGSERVE_GRAPHS='{{"agent": "/deps/__outer_unit_tests/unit_tests/agent.py:graph"}}'
{textwrap.indent(textwrap.dedent(PIP_CLEANUP_LINES), "                ")}
                WORKDIR /deps/__outer_unit_tests/unit_tests
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
                ADD . /deps/__outer_unit_tests/unit_tests
                RUN set -ex && \\
                    for line in '[project]' \\
                                'name = "unit_tests"' \\
                                'version = "0.1"' \\
                                '[tool.setuptools.package-data]' \\
                                '"*" = ["**/*"]'; do \\
                        echo "$line" >> /deps/__outer_unit_tests/pyproject.toml; \\
                    done
                # -- End of non-package dependency unit_tests --
                # -- Installing all local dependencies --
                RUN PYTHONDONTWRITEBYTECODE=1 pip install --no-cache-dir -c /api/constraints.txt -e /deps/*
                # -- End of local dependencies install --
                ENV LANGSERVE_GRAPHS='{{"agent": "/deps/__outer_unit_tests/unit_tests/agent.py:graph"}}'
{textwrap.indent(textwrap.dedent(PIP_CLEANUP_LINES), "                ")}
                WORKDIR /deps/__outer_unit_tests/unit_tests
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
                ADD . /deps/__outer_unit_tests/unit_tests
                RUN set -ex && \\
                    for line in '[project]' \\
                                'name = "unit_tests"' \\
                                'version = "0.1"' \\
                                '[tool.setuptools.package-data]' \\
                                '"*" = ["**/*"]'; do \\
                        echo "$line" >> /deps/__outer_unit_tests/pyproject.toml; \\
                    done
                # -- End of non-package dependency unit_tests --
                # -- Installing all local dependencies --
                RUN PYTHONDONTWRITEBYTECODE=1 pip install --no-cache-dir -c /api/constraints.txt -e /deps/*
                # -- End of local dependencies install --
                ENV LANGSERVE_GRAPHS='{{"agent": "/deps/__outer_unit_tests/unit_tests/agent.py:graph"}}'
{textwrap.indent(textwrap.dedent(PIP_CLEANUP_LINES), "                ")}
                WORKDIR /deps/__outer_unit_tests/unit_tests
        
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
                ADD . /deps/__outer_unit_tests/unit_tests
                RUN set -ex && \\
                    for line in '[project]' \\
                                'name = "unit_tests"' \\
                                'version = "0.1"' \\
                                '[tool.setuptools.package-data]' \\
                                '"*" = ["**/*"]'; do \\
                        echo "$line" >> /deps/__outer_unit_tests/pyproject.toml; \\
                    done
                # -- End of non-package dependency unit_tests --
                # -- Installing all local dependencies --
                RUN PYTHONDONTWRITEBYTECODE=1 pip install --no-cache-dir -c /api/constraints.txt -e /deps/*
                # -- End of local dependencies install --
                ENV LANGSERVE_GRAPHS='{{"agent": "/deps/__outer_unit_tests/unit_tests/agent.py:graph"}}'
{textwrap.indent(textwrap.dedent(PIP_CLEANUP_LINES), "                ")}
                WORKDIR /deps/__outer_unit_tests/unit_tests
        
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
