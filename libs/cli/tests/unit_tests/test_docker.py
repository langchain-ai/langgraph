from langgraph_cli.docker import (
    DEFAULT_POSTGRES_URI,
    DockerCapabilities,
    Version,
    compose,
)
from langgraph_cli.util import clean_empty_lines

DEFAULT_DOCKER_CAPABILITIES = DockerCapabilities(
    version_docker=Version(26, 1, 1),
    version_compose=Version(2, 27, 0),
    healthcheck_start_interval=False,
)


def test_compose_with_no_debugger_and_custom_db():
    port = 8123
    custom_postgres_uri = "custom_postgres_uri"
    actual_compose_str = compose(
        DEFAULT_DOCKER_CAPABILITIES, port=port, postgres_uri=custom_postgres_uri
    )
    expected_compose_str = f"""services:
    langgraph-api:
        restart: on-failure
        ports:
            - "{port}:8000"
        environment:
            POSTGRES_URI: {custom_postgres_uri}"""
    assert clean_empty_lines(actual_compose_str) == expected_compose_str


def test_compose_with_no_debugger_and_custom_db_with_healthcheck():
    port = 8123
    custom_postgres_uri = "custom_postgres_uri"
    actual_compose_str = compose(
        DEFAULT_DOCKER_CAPABILITIES._replace(healthcheck_start_interval=True),
        port=port,
        postgres_uri=custom_postgres_uri,
    )
    expected_compose_str = f"""services:
    langgraph-api:
        restart: on-failure
        ports:
            - "{port}:8000"
        environment:
            POSTGRES_URI: {custom_postgres_uri}
        healthcheck:
            test: python /api/healthcheck.py
            interval: 60s
            start_interval: 1s
            start_period: 10s"""
    assert clean_empty_lines(actual_compose_str) == expected_compose_str


def test_compose_with_debugger_and_custom_db():
    port = 8123
    custom_postgres_uri = "custom_postgres_uri"
    actual_compose_str = compose(
        DEFAULT_DOCKER_CAPABILITIES,
        port=port,
        postgres_uri=custom_postgres_uri,
    )
    expected_compose_str = f"""services:
    langgraph-api:
        restart: on-failure
        ports:
            - "{port}:8000"
        environment:
            POSTGRES_URI: {custom_postgres_uri}"""
    assert clean_empty_lines(actual_compose_str) == expected_compose_str


def test_compose_with_debugger_and_default_db():
    port = 8123
    actual_compose_str = compose(DEFAULT_DOCKER_CAPABILITIES, port=port)
    expected_compose_str = f"""volumes:
    langgraph-data:
        driver: local
services:
    langgraph-postgres:
        image: postgres:16
        restart: on-failure
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
            interval: 5s
    langgraph-api:
        restart: on-failure
        ports:
            - "{port}:8000"
        depends_on:
            langgraph-postgres:
                condition: service_healthy
        environment:
            POSTGRES_URI: {DEFAULT_POSTGRES_URI}"""
    assert clean_empty_lines(actual_compose_str) == expected_compose_str
