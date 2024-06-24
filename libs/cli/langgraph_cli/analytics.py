import functools
import json
import os
import pathlib
import platform
import threading
import urllib.error
import urllib.request
from typing import Any, TypedDict

from langgraph_cli.constants import (
    DEFAULT_CONFIG,
    DEFAULT_PORT,
    SUPABASE_PUBLIC_API_KEY,
    SUPABASE_URL,
)
from langgraph_cli.version import __version__


class LogData(TypedDict):
    os: str
    os_version: str
    python_version: str
    cli_version: str
    cli_command: str
    params: dict[str, Any]


def get_anonymized_params(kwargs: dict[str, Any]) -> dict[str, bool]:
    params = {}

    # anonymize params with values
    if config := kwargs.get("config"):
        if config != pathlib.Path(DEFAULT_CONFIG).resolve():
            params["config"] = True

    if port := kwargs.get("port"):
        if port != DEFAULT_PORT:
            params["port"] = True

    if kwargs.get("docker_compose"):
        params["docker_compose"] = True

    if kwargs.get("debugger_port"):
        params["debugger_port"] = True

    if kwargs.get("postgres_uri"):
        params["postgres_uri"] = True

    # pick up exact values for boolean flags
    for boolean_param in ["recreate", "pull", "watch", "wait", "verbose"]:
        if kwargs.get(boolean_param):
            params[boolean_param] = kwargs[boolean_param]

    return params


def log_data(data: LogData) -> None:
    headers = {
        "Content-Type": "application/json",
        "apikey": SUPABASE_PUBLIC_API_KEY,
        "User-Agent": "Mozilla/5.0",
    }
    supabase_url = SUPABASE_URL

    req = urllib.request.Request(
        f"{supabase_url}/rest/v1/logs",
        data=json.dumps(data).encode("utf-8"),
        headers=headers,
        method="POST",
    )

    try:
        urllib.request.urlopen(req)
    except urllib.error.URLError:
        pass


def log_command(func):
    @functools.wraps(func)
    def decorator(*args, **kwargs):
        if os.getenv("LANGGRAPH_CLI_NO_ANALYTICS") == "1":
            return func(*args, **kwargs)

        data = {
            "os": platform.system(),
            "os_version": platform.version(),
            "python_version": platform.python_version(),
            "cli_version": __version__,
            "cli_command": func.__name__,
            "params": get_anonymized_params(kwargs),
        }

        background_thread = threading.Thread(target=log_data, args=(data,))
        background_thread.start()
        return func(*args, **kwargs)

    return decorator
