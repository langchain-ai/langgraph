import json
import os
import pathlib
import tempfile

import click
import pytest

from langgraph_cli.config import (
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
    expected_config = {
        "python_version": "3.11",
        "pip_config_file": None,
        "dockerfile_lines": [],
        "env": {},
        "store": None,
        "auth": None,
        **expected_config,
    }
    actual_config = validate_config(expected_config)
    assert actual_config == expected_config

    # full config
    env = ".env"
    expected_config = {
        "python_version": "3.12",
        "pip_config_file": "pipconfig.txt",
        "dockerfile_lines": ["ARG meow"],
        "dependencies": [".", "langchain"],
        "graphs": {
            "agent": "./agent.py:graph",
        },
        "env": env,
        "store": None,
        "auth": None,
    }
    actual_config = validate_config(expected_config)
    assert actual_config == expected_config
    expected_config["python_version"] = "3.13"
    actual_config = validate_config(expected_config)
    assert actual_config == expected_config

    # check wrong python version raises
    with pytest.raises(click.UsageError):
        validate_config(
            {
                "python_version": "3.9",
            }
        )

    # check missing dependencies key raises
    with pytest.raises(click.UsageError):
        validate_config(
            {"python_version": "3.9", "graphs": {"agent": "./agent.py:graph"}},
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


# config_to_docker
def test_config_to_docker_simple():
    graphs = {"agent": "./agent.py:graph"}
    actual_docker_stdin = config_to_docker(
        PATH_TO_CONFIG,
        validate_config({"dependencies": ["."], "graphs": graphs}),
        "langchain/langgraph-api",
    )
    expected_docker_stdin = """\
FROM langchain/langgraph-api:3.11
ADD . /deps/__outer_unit_tests/unit_tests
RUN set -ex && \\
    for line in '[project]' \\
                'name = "unit_tests"' \\
                'version = "0.1"' \\
                '[tool.setuptools.package-data]' \\
                '"*" = ["**/*"]'; do \\
        echo "$line" >> /deps/__outer_unit_tests/pyproject.toml; \\
    done
RUN PYTHONDONTWRITEBYTECODE=1 pip install --no-cache-dir -c /api/constraints.txt -e /deps/*
ENV LANGSERVE_GRAPHS='{"agent": "/deps/__outer_unit_tests/unit_tests/agent.py:graph"}'
WORKDIR /deps/__outer_unit_tests/unit_tests\
"""
    assert clean_empty_lines(actual_docker_stdin) == expected_docker_stdin


def test_config_to_docker_pipconfig():
    graphs = {"agent": "./agent.py:graph"}
    actual_docker_stdin = config_to_docker(
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
    expected_docker_stdin = """\
FROM langchain/langgraph-api:3.11
ADD pipconfig.txt /pipconfig.txt
ADD . /deps/__outer_unit_tests/unit_tests
RUN set -ex && \\
    for line in '[project]' \\
                'name = "unit_tests"' \\
                'version = "0.1"' \\
                '[tool.setuptools.package-data]' \\
                '"*" = ["**/*"]'; do \\
        echo "$line" >> /deps/__outer_unit_tests/pyproject.toml; \\
    done
RUN PIP_CONFIG_FILE=/pipconfig.txt PYTHONDONTWRITEBYTECODE=1 pip install --no-cache-dir -c /api/constraints.txt -e /deps/*
ENV LANGSERVE_GRAPHS='{"agent": "/deps/__outer_unit_tests/unit_tests/agent.py:graph"}'
WORKDIR /deps/__outer_unit_tests/unit_tests\
"""
    assert clean_empty_lines(actual_docker_stdin) == expected_docker_stdin


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
    actual_docker_stdin = config_to_docker(
        PATH_TO_CONFIG,
        validate_config(
            {
                "dependencies": ["./graphs"],
                "graphs": graphs,
            }
        ),
        "langchain/langgraph-api-custom",
    )
    expected_docker_stdin = """\
FROM langchain/langgraph-api-custom:3.11
ADD ./graphs /deps/__outer_graphs/src
RUN set -ex && \\
    for line in '[project]' \\
                'name = "graphs"' \\
                'version = "0.1"' \\
                '[tool.setuptools.package-data]' \\
                '"*" = ["**/*"]'; do \\
        echo "$line" >> /deps/__outer_graphs/pyproject.toml; \\
    done
RUN PYTHONDONTWRITEBYTECODE=1 pip install --no-cache-dir -c /api/constraints.txt -e /deps/*
ENV LANGSERVE_GRAPHS='{"agent": "/deps/__outer_graphs/src/agent.py:graph"}'\
"""
    assert clean_empty_lines(actual_docker_stdin) == expected_docker_stdin


def test_config_to_docker_pyproject():
    pyproject_str = """[project]
name = "custom"
version = "0.1"
dependencies = ["langchain"]"""
    pyproject_path = "tests/unit_tests/pyproject.toml"
    with open(pyproject_path, "w") as f:
        f.write(pyproject_str)

    graphs = {"agent": "./graphs/agent.py:graph"}
    actual_docker_stdin = config_to_docker(
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
    expected_docker_stdin = """FROM langchain/langgraph-api:3.11
ADD . /deps/unit_tests
RUN PYTHONDONTWRITEBYTECODE=1 pip install --no-cache-dir -c /api/constraints.txt -e /deps/*
ENV LANGSERVE_GRAPHS='{"agent": "/deps/unit_tests/graphs/agent.py:graph"}'
WORKDIR /deps/unit_tests"""
    assert clean_empty_lines(actual_docker_stdin) == expected_docker_stdin


def test_config_to_docker_end_to_end():
    graphs = {"agent": "./graphs/agent.py:graph"}
    actual_docker_stdin = config_to_docker(
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
    expected_docker_stdin = """FROM langchain/langgraph-api:3.12
ARG meow
ARG foo
ADD pipconfig.txt /pipconfig.txt
RUN PIP_CONFIG_FILE=/pipconfig.txt PYTHONDONTWRITEBYTECODE=1 pip install --no-cache-dir -c /api/constraints.txt langchain langchain_openai
ADD ./graphs/ /deps/__outer_graphs/src
RUN set -ex && \\
    for line in '[project]' \\
                'name = "graphs"' \\
                'version = "0.1"' \\
                '[tool.setuptools.package-data]' \\
                '"*" = ["**/*"]'; do \\
        echo "$line" >> /deps/__outer_graphs/pyproject.toml; \\
    done
RUN PIP_CONFIG_FILE=/pipconfig.txt PYTHONDONTWRITEBYTECODE=1 pip install --no-cache-dir -c /api/constraints.txt -e /deps/*
ENV LANGSERVE_GRAPHS='{"agent": "/deps/__outer_graphs/src/agent.py:graph"}'"""
    assert clean_empty_lines(actual_docker_stdin) == expected_docker_stdin


# node.js build used for LangGraph Cloud
def test_config_to_docker_nodejs():
    graphs = {"agent": "./graphs/agent.js:graph"}
    actual_docker_stdin = config_to_docker(
        PATH_TO_CONFIG,
        validate_config(
            {
                "node_version": "20",
                "graphs": graphs,
                "dockerfile_lines": ["ARG meow", "ARG foo"],
            }
        ),
        "langchain/langgraphjs-api",
    )
    expected_docker_stdin = """FROM langchain/langgraphjs-api:20
ARG meow
ARG foo
ADD . /deps/unit_tests
RUN cd /deps/unit_tests && npm i
ENV LANGSERVE_GRAPHS='{"agent": "./graphs/agent.js:graph"}'
WORKDIR /deps/unit_tests
RUN (test ! -f /api/langgraph_api/js/build.mts && echo "Prebuild script not found, skipping") || tsx /api/langgraph_api/js/build.mts"""

    assert clean_empty_lines(actual_docker_stdin) == expected_docker_stdin


# config_to_compose
def test_config_to_compose_simple_config():
    graphs = {"agent": "./agent.py:graph"}
    expected_compose_stdin = """\
        
        pull_policy: build
        build:
            context: .
            dockerfile_inline: |
                FROM langchain/langgraph-api:3.11
                ADD . /deps/__outer_unit_tests/unit_tests
                RUN set -ex && \\
                    for line in '[project]' \\
                                'name = "unit_tests"' \\
                                'version = "0.1"' \\
                                '[tool.setuptools.package-data]' \\
                                '"*" = ["**/*"]'; do \\
                        echo "$line" >> /deps/__outer_unit_tests/pyproject.toml; \\
                    done
                RUN PYTHONDONTWRITEBYTECODE=1 pip install --no-cache-dir -c /api/constraints.txt -e /deps/*
                ENV LANGSERVE_GRAPHS='{"agent": "/deps/__outer_unit_tests/unit_tests/agent.py:graph"}'
                WORKDIR /deps/__outer_unit_tests/unit_tests
        """
    actual_compose_stdin = config_to_compose(
        PATH_TO_CONFIG,
        validate_config({"dependencies": ["."], "graphs": graphs}),
        "langchain/langgraph-api",
    )
    assert clean_empty_lines(actual_compose_stdin) == expected_compose_stdin


def test_config_to_compose_env_vars():
    graphs = {"agent": "./agent.py:graph"}
    expected_compose_stdin = """                        OPENAI_API_KEY: "key"
        
        pull_policy: build
        build:
            context: .
            dockerfile_inline: |
                FROM langchain/langgraph-api-custom:3.11
                ADD . /deps/__outer_unit_tests/unit_tests
                RUN set -ex && \\
                    for line in '[project]' \\
                                'name = "unit_tests"' \\
                                'version = "0.1"' \\
                                '[tool.setuptools.package-data]' \\
                                '"*" = ["**/*"]'; do \\
                        echo "$line" >> /deps/__outer_unit_tests/pyproject.toml; \\
                    done
                RUN PYTHONDONTWRITEBYTECODE=1 pip install --no-cache-dir -c /api/constraints.txt -e /deps/*
                ENV LANGSERVE_GRAPHS='{"agent": "/deps/__outer_unit_tests/unit_tests/agent.py:graph"}'
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
    expected_compose_stdin = """\
        env_file: .env
        pull_policy: build
        build:
            context: .
            dockerfile_inline: |
                FROM langchain/langgraph-api:3.11
                ADD . /deps/__outer_unit_tests/unit_tests
                RUN set -ex && \\
                    for line in '[project]' \\
                                'name = "unit_tests"' \\
                                'version = "0.1"' \\
                                '[tool.setuptools.package-data]' \\
                                '"*" = ["**/*"]'; do \\
                        echo "$line" >> /deps/__outer_unit_tests/pyproject.toml; \\
                    done
                RUN PYTHONDONTWRITEBYTECODE=1 pip install --no-cache-dir -c /api/constraints.txt -e /deps/*
                ENV LANGSERVE_GRAPHS='{"agent": "/deps/__outer_unit_tests/unit_tests/agent.py:graph"}'
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
    expected_compose_stdin = """\
        
        pull_policy: build
        build:
            context: .
            dockerfile_inline: |
                FROM langchain/langgraph-api:3.11
                ADD . /deps/__outer_unit_tests/unit_tests
                RUN set -ex && \\
                    for line in '[project]' \\
                                'name = "unit_tests"' \\
                                'version = "0.1"' \\
                                '[tool.setuptools.package-data]' \\
                                '"*" = ["**/*"]'; do \\
                        echo "$line" >> /deps/__outer_unit_tests/pyproject.toml; \\
                    done
                RUN PYTHONDONTWRITEBYTECODE=1 pip install --no-cache-dir -c /api/constraints.txt -e /deps/*
                ENV LANGSERVE_GRAPHS='{"agent": "/deps/__outer_unit_tests/unit_tests/agent.py:graph"}'
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
    expected_compose_stdin = """\
        env_file: .env
        pull_policy: build
        build:
            context: .
            dockerfile_inline: |
                FROM langchain/langgraph-api:3.11
                ADD . /deps/__outer_unit_tests/unit_tests
                RUN set -ex && \\
                    for line in '[project]' \\
                                'name = "unit_tests"' \\
                                'version = "0.1"' \\
                                '[tool.setuptools.package-data]' \\
                                '"*" = ["**/*"]'; do \\
                        echo "$line" >> /deps/__outer_unit_tests/pyproject.toml; \\
                    done
                RUN PYTHONDONTWRITEBYTECODE=1 pip install --no-cache-dir -c /api/constraints.txt -e /deps/*
                ENV LANGSERVE_GRAPHS='{"agent": "/deps/__outer_unit_tests/unit_tests/agent.py:graph"}'
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
