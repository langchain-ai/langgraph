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


def test_parse_version_normal_versions():
    """Test _parse_version with normal semantic versions."""
    assert _parse_version("1.2.3") == Version(1, 2, 3)
    assert _parse_version("28.1.1") == Version(28, 1, 1)
    assert _parse_version("0.0.1") == Version(0, 0, 1)


def test_parse_version_with_v_prefix():
    """Test _parse_version with 'v' prefix versions."""
    assert _parse_version("v1.2.3") == Version(1, 2, 3)
    assert _parse_version("v28.1.1") == Version(28, 1, 1)
    assert _parse_version("v0.0.1") == Version(0, 0, 1)


def test_parse_version_prerelease():
    """Test _parse_version with prerelease versions."""
    assert _parse_version("1.2.3-alpha") == Version(1, 2, 3)
    assert _parse_version("1.2.3-beta.1") == Version(1, 2, 3)
    assert _parse_version("1.2.3-rc.1") == Version(1, 2, 3)
    assert _parse_version("28.1.1-alpha") == Version(28, 1, 1)


def test_parse_version_build_metadata():
    """Test _parse_version with build metadata versions."""
    assert _parse_version("1.2.3+1") == Version(1, 2, 3)
    assert _parse_version("1.2.3+build.1") == Version(1, 2, 3)
    assert _parse_version("28.1.1+1") == Version(28, 1, 1)  # This was the failing case
    assert _parse_version("1.2.3+20230101") == Version(1, 2, 3)


def test_parse_version_combined_prerelease_and_build():
    """Test _parse_version with combined prerelease and build metadata."""
    assert _parse_version("1.2.3-alpha+1") == Version(1, 2, 3)
    assert _parse_version("1.2.3-beta.1+build.1") == Version(1, 2, 3)
    assert _parse_version("28.1.1-rc+build") == Version(28, 1, 1)
    assert _parse_version("1.2.3-alpha.1+beta") == Version(1, 2, 3)


def test_parse_version_edge_cases():
    """Test _parse_version with edge cases and missing components."""
    # Missing patch version
    assert _parse_version("1.2") == Version(1, 2, 0)
    assert _parse_version("28.1") == Version(28, 1, 0)

    # Missing minor and patch versions
    assert _parse_version("1") == Version(1, 0, 0)
    assert _parse_version("28") == Version(28, 0, 0)

    # With 'v' prefix and missing components
    assert _parse_version("v1.2") == Version(1, 2, 0)
    assert _parse_version("v1") == Version(1, 0, 0)
