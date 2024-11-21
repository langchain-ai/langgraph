import asyncio
import json
import os
import pathlib
import sys
import langgraph_cli
import langgraph_cli.docker
import langgraph_cli.config

from langgraph_cli.exec import Runner, subp_exec
from langgraph_cli.progress import Progress
from langgraph_cli.constants import DEFAULT_PORT


def test(
    config: pathlib.Path,
    port: int,
    tag: str,
    verbose: bool,
):
    with Runner() as runner, Progress(message="Pulling...") as set:
        # check docker available
        capabilities = langgraph_cli.docker.check_capabilities(runner)
        # open config
        config_json = langgraph_cli.config.validate_config_file(config)

        set("Running...")
        args = [
            "run",
            "--rm",
            "-p",
            f"{port}:8000",
        ]
        if isinstance(config_json["env"], str):
            args.extend(
                [
                    "--env-file",
                    str(config.parent / config_json["env"]),
                ]
            )
        else:
            for k, v in config_json["env"].items():
                args.extend(
                    [
                        "-e",
                        f"{k}={v}",
                    ]
                )
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

        def on_stdout(line: str):
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

        async def subp_exec_task(*args, **kwargs):
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
