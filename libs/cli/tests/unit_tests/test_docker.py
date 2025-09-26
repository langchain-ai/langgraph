import pytest

from langgraph_cli.docker import (
    DEFAULT_POSTGRES_URI,
    DockerCapabilities,
    Version,
    _parse_version,
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
    langgraph-redis:
        image: redis:6
        healthcheck:
            test: redis-cli ping
            interval: 5s
            timeout: 1s
            retries: 5
    langgraph-api:
        ports:
            - "{port}:8000"
        depends_on:
            langgraph-redis:
                condition: service_healthy
        environment:
            REDIS_URI: redis://langgraph-redis:6379
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
    langgraph-redis:
        image: redis:6
        healthcheck:
            test: redis-cli ping
            interval: 5s
            timeout: 1s
            retries: 5
    langgraph-api:
        ports:
            - "{port}:8000"
        depends_on:
            langgraph-redis:
                condition: service_healthy
        environment:
            REDIS_URI: redis://langgraph-redis:6379
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
    langgraph-redis:
        image: redis:6
        healthcheck:
            test: redis-cli ping
            interval: 5s
            timeout: 1s
            retries: 5
    langgraph-api:
        ports:
            - "{port}:8000"
        depends_on:
            langgraph-redis:
                condition: service_healthy
        environment:
            REDIS_URI: redis://langgraph-redis:6379
            POSTGRES_URI: {custom_postgres_uri}"""
    assert clean_empty_lines(actual_compose_str) == expected_compose_str


def test_compose_with_debugger_and_default_db():
    port = 8123
    actual_compose_str = compose(DEFAULT_DOCKER_CAPABILITIES, port=port)
    expected_compose_str = f"""volumes:
    langgraph-data:
        driver: local
services:
    langgraph-redis:
        image: redis:6
        healthcheck:
            test: redis-cli ping
            interval: 5s
            timeout: 1s
            retries: 5
    langgraph-postgres:
        image: pgvector/pgvector:pg16
        ports:
            - "5433:5432"
        environment:
            POSTGRES_DB: postgres
            POSTGRES_USER: postgres
            POSTGRES_PASSWORD: postgres
        command:
            - postgres
            - -c
            - shared_preload_libraries=vector
        volumes:
            - langgraph-data:/var/lib/postgresql/data
        healthcheck:
            test: pg_isready -U postgres
            start_period: 10s
            timeout: 1s
            retries: 5
            interval: 5s
    langgraph-api:
        ports:
            - "{port}:8000"
        depends_on:
            langgraph-redis:
                condition: service_healthy
            langgraph-postgres:
                condition: service_healthy
        environment:
            REDIS_URI: redis://langgraph-redis:6379
            POSTGRES_URI: {DEFAULT_POSTGRES_URI}"""
    assert clean_empty_lines(actual_compose_str) == expected_compose_str


def test_compose_with_api_version():
    """Test compose function with api_version parameter."""
    port = 8123
    api_version = "0.2.74"

    actual_compose_str = compose(
        DEFAULT_DOCKER_CAPABILITIES, port=port, api_version=api_version
    )

    # The compose function should generate a compose file that doesn't directly
    # reference the api_version, since it's handled in the docker tag creation
    # when building the image. The compose function mainly sets up services.
    expected_compose_str = f"""volumes:
    langgraph-data:
        driver: local
services:
    langgraph-redis:
        image: redis:6
        healthcheck:
            test: redis-cli ping
            interval: 5s
            timeout: 1s
            retries: 5
    langgraph-postgres:
        image: pgvector/pgvector:pg16
        ports:
            - "5433:5432"
        environment:
            POSTGRES_DB: postgres
            POSTGRES_USER: postgres
            POSTGRES_PASSWORD: postgres
        command:
            - postgres
            - -c
            - shared_preload_libraries=vector
        volumes:
            - langgraph-data:/var/lib/postgresql/data
        healthcheck:
            test: pg_isready -U postgres
            start_period: 10s
            timeout: 1s
            retries: 5
            interval: 5s
    langgraph-api:
        ports:
            - "{port}:8000"
        depends_on:
            langgraph-redis:
                condition: service_healthy
            langgraph-postgres:
                condition: service_healthy
        environment:
            REDIS_URI: redis://langgraph-redis:6379
            POSTGRES_URI: {DEFAULT_POSTGRES_URI}"""
    assert clean_empty_lines(actual_compose_str) == expected_compose_str


def test_compose_with_api_version_and_base_image():
    """Test compose function with both api_version and base_image parameters."""
    port = 8123
    api_version = "1.0.0"
    base_image = "my-registry/custom-api"

    actual_compose_str = compose(
        DEFAULT_DOCKER_CAPABILITIES,
        port=port,
        api_version=api_version,
        base_image=base_image,
    )

    # Similar to the previous test - the compose function doesn't directly embed
    # the api_version or base_image into the compose file since those are handled
    # during the docker build process
    expected_compose_str = f"""volumes:
    langgraph-data:
        driver: local
services:
    langgraph-redis:
        image: redis:6
        healthcheck:
            test: redis-cli ping
            interval: 5s
            timeout: 1s
            retries: 5
    langgraph-postgres:
        image: pgvector/pgvector:pg16
        ports:
            - "5433:5432"
        environment:
            POSTGRES_DB: postgres
            POSTGRES_USER: postgres
            POSTGRES_PASSWORD: postgres
        command:
            - postgres
            - -c
            - shared_preload_libraries=vector
        volumes:
            - langgraph-data:/var/lib/postgresql/data
        healthcheck:
            test: pg_isready -U postgres
            start_period: 10s
            timeout: 1s
            retries: 5
            interval: 5s
    langgraph-api:
        ports:
            - "{port}:8000"
        depends_on:
            langgraph-redis:
                condition: service_healthy
            langgraph-postgres:
                condition: service_healthy
        environment:
            REDIS_URI: redis://langgraph-redis:6379
            POSTGRES_URI: {DEFAULT_POSTGRES_URI}"""
    assert clean_empty_lines(actual_compose_str) == expected_compose_str


def test_compose_with_api_version_and_custom_postgres():
    """Test compose function with api_version and custom postgres URI."""
    port = 8123
    api_version = "0.2.74"
    custom_postgres_uri = "postgresql://user:pass@external-db:5432/mydb"

    actual_compose_str = compose(
        DEFAULT_DOCKER_CAPABILITIES,
        port=port,
        api_version=api_version,
        postgres_uri=custom_postgres_uri,
    )

    expected_compose_str = f"""services:
    langgraph-redis:
        image: redis:6
        healthcheck:
            test: redis-cli ping
            interval: 5s
            timeout: 1s
            retries: 5
    langgraph-api:
        ports:
            - "{port}:8000"
        depends_on:
            langgraph-redis:
                condition: service_healthy
        environment:
            REDIS_URI: redis://langgraph-redis:6379
            POSTGRES_URI: {custom_postgres_uri}"""
    assert clean_empty_lines(actual_compose_str) == expected_compose_str


def test_compose_with_api_version_and_debugger():
    """Test compose function with api_version and debugger port."""
    port = 8123
    debugger_port = 8001
    api_version = "0.2.74"

    actual_compose_str = compose(
        DEFAULT_DOCKER_CAPABILITIES,
        port=port,
        api_version=api_version,
        debugger_port=debugger_port,
    )

    expected_compose_str = f"""volumes:
    langgraph-data:
        driver: local
services:
    langgraph-redis:
        image: redis:6
        healthcheck:
            test: redis-cli ping
            interval: 5s
            timeout: 1s
            retries: 5
    langgraph-postgres:
        image: pgvector/pgvector:pg16
        ports:
            - "5433:5432"
        environment:
            POSTGRES_DB: postgres
            POSTGRES_USER: postgres
            POSTGRES_PASSWORD: postgres
        command:
            - postgres
            - -c
            - shared_preload_libraries=vector
        volumes:
            - langgraph-data:/var/lib/postgresql/data
        healthcheck:
            test: pg_isready -U postgres
            start_period: 10s
            timeout: 1s
            retries: 5
            interval: 5s
    langgraph-debugger:
        image: langchain/langgraph-debugger
        restart: on-failure
        depends_on:
            langgraph-postgres:
                condition: service_healthy
        ports:
            - "{debugger_port}:3968"
    langgraph-api:
        ports:
            - "{port}:8000"
        depends_on:
            langgraph-redis:
                condition: service_healthy
            langgraph-postgres:
                condition: service_healthy
        environment:
            REDIS_URI: redis://langgraph-redis:6379
            POSTGRES_URI: {DEFAULT_POSTGRES_URI}"""
    assert clean_empty_lines(actual_compose_str) == expected_compose_str


@pytest.mark.parametrize(
    "input_str,expected",
    [
        ("1.2.3", Version(1, 2, 3)),
        ("v1.2.3", Version(1, 2, 3)),
        ("1.2.3-alpha", Version(1, 2, 3)),
        ("1.2.3+1", Version(1, 2, 3)),
        ("1.2.3-alpha+build", Version(1, 2, 3)),
        ("1.2", Version(1, 2, 0)),
        ("1", Version(1, 0, 0)),
        ("v28.1.1+1", Version(28, 1, 1)),
        ("2.0.0-beta.1+exp.sha.5114f85", Version(2, 0, 0)),
        ("v3.4.5-rc1+build.123", Version(3, 4, 5)),
    ],
)
def test_parse_version_w_edge_cases(input_str, expected):
    assert _parse_version(input_str) == expected
