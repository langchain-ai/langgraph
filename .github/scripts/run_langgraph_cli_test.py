#!/usr/bin/env python3
import asyncio
import pathlib
import sys
from typing import Any

import langgraph_cli
import langgraph_cli.config
import langgraph_cli.docker
from langgraph_cli.constants import DEFAULT_PORT
from langgraph_cli.exec import Runner, subp_exec
from langgraph_cli.progress import Progress


def test(
    config: pathlib.Path,
    port: int,
    tag: str,
    verbose: bool,
):
    """Test the langgraph CLI by running a Docker container with the specified configuration.

    Args:
        config: Path to the configuration file.
        port: Port number to expose the API on.
        tag: Docker image tag to use.
        verbose: Whether to enable verbose logging.
    """
    with Runner() as runner, Progress(message="Pulling...") as set:
        # Check if Docker is available and get its capabilities
        capabilities = langgraph_cli.docker.check_capabilities(runner)

        # Validate and load the configuration file
        config_json = langgraph_cli.config.validate_config_file(config)

        set("Running...")
        args = [
            "run",
            "--rm",  # Remove the container after it exits
            "-p",
            f"{port}:8000",  # Expose the API on the specified port
        ]

        # Add environment variables from the config file
        if isinstance(config_json["env"], str):
            # If env is a string, it's a path to an env file
            args.extend(
                [
                    "--env-file",
                    str(config.parent / config_json["env"]),
                ]
            )
        else:
            # Otherwise, env is a dictionary of key-value pairs
            for k, v in config_json["env"].items():
                args.extend(
                    [
                        "-e",
                        f"{k}={v}",
                    ]
                )

        # Add Docker healthcheck options based on capabilities
        if capabilities.healthcheck_start_interval:
            args.extend(
                [
                    "--health-interval",
                    "5s",
                    "--health-retries",
                    "1",
                    "--health-start-period",
                    "10s",
                    "--health-start-interval",
                    "1s",
                ]
            )
        else:
            args.extend(
                [
                    "--health-interval",
                    "5s",
                    "--health-retries",
                    "2",
                ]
            )

        _task = None

        def on_stdout(line: str) -> bool:
            """Callback function to handle stdout from the Docker container.

            Args:
                line (str): A line of output from the container.

            Returns:
                bool: True if the container is ready, False otherwise.
            """
            nonlocal _task
            if "GET /ok" in line or "Uvicorn running on" in line:
                set("")
                sys.stdout.write(
                    f"""Ready!
- API: http://localhost:{port}
"""
                )
                sys.stdout.flush()
                _task.cancel()
                return True
            return False

        async def subp_exec_task(*args: str, **kwargs: Any) -> None:
            """
            Asynchronous function to execute a subprocess and wait for it to complete.

            Args:
                *args: Positional arguments to pass to subp_exec.
                **kwargs: Keyword arguments to pass to subp_exec.
            """
            nonlocal _task
            _task = asyncio.create_task(subp_exec(*args, **kwargs))
            await _task

        try:
            runner.run(
                subp_exec_task(
                    "docker",
                    *args,
                    tag,
                    verbose=verbose,
                    on_stdout=on_stdout,
                )
            )
        except asyncio.CancelledError:
            pass


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("-t", "--tag", type=str)
    parser.add_argument("-c", "--config", type=str, default="./langgraph.json")
    parser.add_argument("-p", "--port", default=DEFAULT_PORT)
    args = parser.parse_args()
    test(pathlib.Path(args.config), args.port, args.tag, verbose=True)
