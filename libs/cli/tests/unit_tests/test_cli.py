import pathlib

from langgraph_cli.cli import prepare_args_and_stdin
from langgraph_cli.config import Config, validate_config
from langgraph_cli.docker import DEFAULT_POSTGRES_URI, DockerCapabilities, Version
from langgraph_cli.util import clean_empty_lines

DEFAULT_DOCKER_CAPABILITIES = DockerCapabilities(
    version_docker=Version(26, 1, 1),
    version_compose=Version(2, 27, 0),
    healthcheck_start_interval=True,
)


def test_prepare_args_and_stdin():
    # this basically serves as an end-to-end test for using config and docker helpers
    config_path = pathlib.Path("./langgraph.json")
    config = validate_config(
        Config(dependencies=["."], graphs={"agent": "agent.py:graph"})
    )
    port = 8000
    debugger_port = 8001

    actual_args, actual_stdin = prepare_args_and_stdin(
        capabilities=DEFAULT_DOCKER_CAPABILITIES,
        config_path=config_path,
        config=config,
        docker_compose="custom-docker-compose.yml",
        port=port,
        debugger_port=debugger_port,
        watch=True,
    )

    expected_args = [
        "--project-directory",
        ".",
        "-f",
        "custom-docker-compose.yml",
        "-f",
        "-",
    ]
    expected_stdin = f"""volumes:
    langgraph-data:
        driver: local
services:
    langgraph-postgres:
        image: postgres:16
        ports:
            - "5433:5432"
        environment:
            POSTGRES_DB: postgres
            POSTGRES_USER: postgres
            POSTGRES_PASSWORD: postgres
        volumes:
            - langgraph-data:/var/lib/postgresql/data
        healthcheck:
            test: pg_isready -U postgres
            start_period: 10s
            timeout: 1s
            retries: 5
            interval: 60s
            start_interval: 1s
    langgraph-debugger:
        image: langchain/langgraph-debugger
        restart: on-failure
        ports:
            - "{debugger_port}:3968"
        depends_on:
            langgraph-postgres:
                condition: service_healthy
    langgraph-api:
        ports:
            - "8000:8000"
        depends_on:
            langgraph-postgres:
                condition: service_healthy
        environment:
            POSTGRES_URI: {DEFAULT_POSTGRES_URI}
        healthcheck:
            test: python /api/healthcheck.py
            interval: 60s
            start_interval: 1s
            start_period: 10s
        
        pull_policy: build
        build:
            context: .
            dockerfile_inline: |
                FROM langchain/langgraph-api:3.11
                ADD . /deps/
                RUN pip install -c /api/constraints.txt -e /deps/*
                ENV LANGSERVE_GRAPHS='{{"agent": "agent.py:graph"}}'
                WORKDIR /deps/
        
        develop:
            watch:
                - path: langgraph.json
                  action: rebuild
                - path: .
                  action: rebuild\
"""
    assert actual_args == expected_args
    assert clean_empty_lines(actual_stdin) == expected_stdin
