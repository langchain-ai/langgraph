import os
import pathlib

import click
import pytest

from langgraph_cli.config import config_to_compose, config_to_docker, validate_config
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
    }
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
RUN pip install -c /api/constraints.txt -e /deps/*
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
RUN PIP_CONFIG_FILE=/pipconfig.txt pip install -c /api/constraints.txt -e /deps/*
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
RUN pip install -c /api/constraints.txt -e /deps/*
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
RUN pip install -c /api/constraints.txt -e /deps/*
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
RUN PIP_CONFIG_FILE=/pipconfig.txt pip install -c /api/constraints.txt langchain langchain_openai
ADD ./graphs/ /deps/__outer_graphs/src
RUN set -ex && \\
    for line in '[project]' \\
                'name = "graphs"' \\
                'version = "0.1"' \\
                '[tool.setuptools.package-data]' \\
                '"*" = ["**/*"]'; do \\
        echo "$line" >> /deps/__outer_graphs/pyproject.toml; \\
    done
RUN PIP_CONFIG_FILE=/pipconfig.txt pip install -c /api/constraints.txt -e /deps/*
ENV LANGSERVE_GRAPHS='{"agent": "/deps/__outer_graphs/src/agent.py:graph"}'"""
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
                RUN pip install -c /api/constraints.txt -e /deps/*
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
    expected_compose_stdin = """                        OPENAI_API_KEY: key
        
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
                RUN pip install -c /api/constraints.txt -e /deps/*
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
                RUN pip install -c /api/constraints.txt -e /deps/*
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
                RUN pip install -c /api/constraints.txt -e /deps/*
                ENV LANGSERVE_GRAPHS='{"agent": "/deps/__outer_unit_tests/unit_tests/agent.py:graph"}'
                WORKDIR /deps/__outer_unit_tests/unit_tests
        
        develop:
            watch:
                - path: tests/unit_tests/test_config.json
                  action: rebuild
                  ignore:
                    - .langgraph-data
                - path: tests/unit_tests
                  action: rebuild
                  ignore:
                    - .langgraph-data\
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
                RUN pip install -c /api/constraints.txt -e /deps/*
                ENV LANGSERVE_GRAPHS='{"agent": "/deps/__outer_unit_tests/unit_tests/agent.py:graph"}'
                WORKDIR /deps/__outer_unit_tests/unit_tests
        
        develop:
            watch:
                - path: tests/unit_tests/test_config.json
                  action: rebuild
                  ignore:
                    - .langgraph-data
                - path: tests/unit_tests
                  action: rebuild
                  ignore:
                    - .langgraph-data
                - path: path/to/langgraph/api
                  action: sync+restart
                  target: /api/langgraph_api\
"""
    actual_compose_stdin = config_to_compose(
        PATH_TO_CONFIG,
        validate_config({"dependencies": ["."], "graphs": graphs, "env": ".env"}),
        "langchain/langgraph-api",
        watch=True,
        langgraph_api_path="path/to/langgraph/api",
    )
    assert clean_empty_lines(actual_compose_stdin) == expected_compose_stdin
