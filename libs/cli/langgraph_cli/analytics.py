import functools
import os
import pathlib
import platform
from typing import Any

import httpx
from dotenv import load_dotenv

from langgraph_cli.constants import DEFAULT_CONFIG, DEFAULT_PORT
from langgraph_cli.version import __version__

load_dotenv()


def get_anonymized_params(kwargs: dict[str, Any]):
    params = {}

    # anonymize params with values
    if "config" in kwargs:
        params["config"] = kwargs["config"] != pathlib.Path(DEFAULT_CONFIG).resolve()

    if kwargs.get("docker_compose"):
        params["docker_compose"] = True

    if "port" in kwargs:
        params["port"] = kwargs["port"] != DEFAULT_PORT

    if kwargs.get("debugger_port"):
        params["debugger_port"] = True

    if kwargs.get("postgres_uri"):
        params["postgres_uri"] = True

    # pick up exact values for boolean flags
    for boolean_param in ["recreate", "pull", "watch", "wait", "verbose"]:
        if kwargs.get(boolean_param):
            params[boolean_param] = kwargs[boolean_param]

    return params


def log_command(func):
    @functools.wraps(func)
    def decorator(*args, **kwargs):
        if os.getenv("LANGGRAPH_CLI_NO_ANALYTICS") == "1":
            return

        data = {
            "os": platform.system(),
            "os_version": platform.version(),
            "python_version": platform.python_version(),
            "cli_version": __version__,
            "cli_command": func.__name__,
            "params": get_anonymized_params(kwargs),
        }

        # TODO: update w/ LangChain supabase key
        headers = {"apikey": os.environ["SUPABASE_PUBLIC_API_KEY"]}
        supabase_url = os.environ["SUPABASE_URL"]
        try:
            httpx.post(
                f"{supabase_url}/rest/v1/logs",
                json=data,
                headers=headers,
            )
        except httpx.HTTPStatusError:
            print("Failed to log")
            pass

        return func(*args, **kwargs)

    return decorator
