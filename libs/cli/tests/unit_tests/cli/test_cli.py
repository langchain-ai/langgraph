import json
import pathlib
import shutil
import tempfile
from contextlib import contextmanager
from pathlib import Path

from click.testing import CliRunner

from langgraph_cli.cli import cli, prepare_args_and_stdin
from langgraph_cli.config import Config, validate_config
from langgraph_cli.docker import DEFAULT_POSTGRES_URI, DockerCapabilities, Version
from langgraph_cli.util import clean_empty_lines

DEFAULT_DOCKER_CAPABILITIES = DockerCapabilities(
    version_docker=Version(26, 1, 1),
    version_compose=Version(2, 27, 0),
    healthcheck_start_interval=True,
)


@contextmanager
def temporary_config_folder(config_content: dict):
    # Create a temporary directory
    temp_dir = tempfile.mkdtemp()
    try:
        # Define the path for the config.json file
        config_path = Path(temp_dir) / "config.json"

        # Write the provided dictionary content to config.json
        with open(config_path, "w", encoding="utf-8") as config_file:
            json.dump(config_content, config_file)

        # Yield the temporary directory path for use within the context
        yield config_path.parent
    finally:
        # Cleanup the temporary directory and its contents
        shutil.rmtree(temp_dir)


def test_prepare_args_and_stdin() -> None:
    # this basically serves as an end-to-end test for using config and docker helpers
    config_path = pathlib.Path("./langgraph.json")
    config = validate_config(
        Config(dependencies=["."], graphs={"agent": "agent.py:graph"})
    )
    port = 8000
    debugger_port = 8001
    debugger_graph_url = f"http://127.0.0.1:{port}"

    actual_args, actual_stdin = prepare_args_and_stdin(
        capabilities=DEFAULT_DOCKER_CAPABILITIES,
        config_path=config_path,
        config=config,
        docker_compose=pathlib.Path("custom-docker-compose.yml"),
        port=port,
        debugger_port=debugger_port,
        debugger_base_url=debugger_graph_url,
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
    langgraph-redis:
        image: redis:6
        healthcheck:
            test: redis-cli ping
            interval: 5s
            timeout: 1s
            retries: 5
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
        depends_on:
            langgraph-postgres:
                condition: service_healthy
        ports:
            - "{debugger_port}:3968"
        environment:
            VITE_STUDIO_LOCAL_GRAPH_URL: {debugger_graph_url}
    langgraph-api:
        ports:
            - "8000:8000"
        depends_on:
            langgraph-redis:
                condition: service_healthy
            langgraph-postgres:
                condition: service_healthy
        environment:
            REDIS_URI: redis://langgraph-redis:6379
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
                RUN PYTHONDONTWRITEBYTECODE=1 pip install --no-cache-dir -c /api/constraints.txt -e /deps/*
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


def test_version_option() -> None:
    """Test the --version option of the CLI."""
    runner = CliRunner()
    result = runner.invoke(cli, ["--version"])

    # Verify that the command executed successfully
    assert result.exit_code == 0, "Expected exit code 0 for --version option"

    # Check that the output contains the correct version information
    assert (
        "LangGraph CLI, version" in result.output
    ), "Expected version information in output"


def test_dockerfile_command_basic() -> None:
    """Test the 'dockerfile' command with basic configuration."""
    runner = CliRunner()
    config_content = {
        "node_version": "20",  # Add any other necessary configuration fields
        "graphs": {"agent": "agent.py:graph"},
    }

    with temporary_config_folder(config_content) as temp_dir:
        save_path = temp_dir / "Dockerfile"

        result = runner.invoke(
            cli,
            ["dockerfile", str(save_path), "--config", str(temp_dir / "config.json")],
        )

        # Assert command was successful
        assert result.exit_code == 0, result.output
        assert "✅ Created: Dockerfile" in result.output

        # Check if Dockerfile was created
        assert save_path.exists()


def test_dockerfile_command_with_docker_compose() -> None:
    """Test the 'dockerfile' command with Docker Compose configuration."""
    runner = CliRunner()
    config_content = {
        "dependencies": ["./my_agent"],
        "graphs": {"agent": "./my_agent/agent.py:graph"},
        "env": ".env",
    }
    with temporary_config_folder(config_content) as temp_dir:
        save_path = temp_dir / "Dockerfile"
        # Add agent.py file
        agent_path = temp_dir / "my_agent" / "agent.py"
        agent_path.parent.mkdir(parents=True, exist_ok=True)
        agent_path.touch()

        result = runner.invoke(
            cli,
            [
                "dockerfile",
                str(save_path),
                "--config",
                str(temp_dir / "config.json"),
                "--add-docker-compose",
            ],
        )

        # Assert command was successful
        assert result.exit_code == 0
        assert "✅ Created: Dockerfile" in result.output
        assert "✅ Created: .dockerignore" in result.output
        assert "✅ Created: docker-compose.yml" in result.output
        assert (
            "✅ Created: .env" in result.output or "➖ Skipped: .env" in result.output
        )
        assert "🎉 Files generated successfully" in result.output

        # Check if Dockerfile, .dockerignore, docker-compose.yml, and .env were created
        assert save_path.exists()
        assert (temp_dir / ".dockerignore").exists()
        assert (temp_dir / "docker-compose.yml").exists()
        assert (temp_dir / ".env").exists() or "➖ Skipped: .env" in result.output


def test_dockerfile_command_with_bad_config() -> None:
    """Test the 'dockerfile' command with basic configuration."""
    runner = CliRunner()
    config_content = {
        "node_version": "20"  # Add any other necessary configuration fields
    }

    with temporary_config_folder(config_content) as temp_dir:
        save_path = temp_dir / "Dockerfile"

        result = runner.invoke(
            cli,
            ["dockerfile", str(save_path), "--config", str(temp_dir / "conf.json")],
        )

        # Assert command was successful
        assert result.exit_code == 2
        assert "conf.json' does not exist" in result.output
