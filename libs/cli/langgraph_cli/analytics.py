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
    DEFAULT_TIMEOUT,
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


def get_anonymized_params(
    kwargs: dict[str, Any], *, cli_command: str
) -> dict[str, bool | str]:
    params: dict[str, bool | str] = {}

    if cli_command == "deploy" and (
        analytics_source := os.getenv("LANGGRAPH_CLI_ANALYTICS_SOURCE")
    ):
        params["source"] = analytics_source

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


def log_data(data: LogData, timeout: int) -> None:
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
        urllib.request.urlopen(req, timeout=timeout)
    except (urllib.error.URLError, TimeoutError):
        pass


def log_command(timeout=None, daemon=None):
    """Decorator to send anonymous CLI telemetry data in a background thread.

    Root cause fixed in #8074:
    Original telemetry lacked network timeout and used non-daemon threads; stalled HTTP requests
    would block CLI from exiting completely.

    Improvements:
    1. Add configurable urlopen timeout for telemetry HTTP requests, prevents infinite network hang.
    2. Expose daemon thread toggle, default daemon=True to avoid blocking CLI exit.
    3. Fully backward compatible: bare `@log_command` requires no code changes.

    Two supported usage modes:
    1. No parentheses (use default config):
        @log_command
        def my_cli_func(): ...
    2. Customize timeout / daemon flag:
        @log(timeout=5, daemon=False)
        def my_cli_func(): ...

    Args:
        timeout: Optional[int] Max seconds for telemetry urlopen HTTP request.
            Falls back to DEFAULT_TIMEOUT if unset. Must be > 0.
        daemon: Optional[bool] Whether telemetry background thread is a daemon thread.
            Defaults to True (thread exits when CLI main process exits).

    Raises:
        ValueError: If resolved timeout value is less than or equal to zero.
    """
    _default_timeout = DEFAULT_TIMEOUT
    _default_daemon = True

    # Factory to generate wrapped decorator with fixed timeout & daemon config
    def make_wrapper(tout: int, dmn: bool):
        def decorator(func):
            @functools.wraps(func)
            def wrapper(*args, **kwargs):
                # Skip telemetry collection if env opt-out flag is set
                if os.getenv("LANGGRAPH_CLI_NO_ANALYTICS") == "1":
                    return func(*args, **kwargs)

                # Build anonymized telemetry payload without sensitive user data
                data = {
                    "os": platform.system(),
                    "os_version": platform.version(),
                    "python_version": platform.python_version(),
                    "cli_version": __version__,
                    "cli_command": func.__name__,
                    "params": get_anonymized_params(kwargs, cli_command=func.__name__),
                }
                # Spawn background thread to send telemetry asynchronously
                background_thread = threading.Thread(
                    target=log_data, args=(data, tout), daemon=dmn
                )
                background_thread.start()
                # Execute original CLI logic immediately, do not wait for telemetry thread
                return func(*args, **kwargs)

            return wrapper

        return decorator

    # Case 1: Bare decorator call @log_command (timeout receives target func)
    if callable(timeout) and daemon is None:
        func = timeout
        wrapper_factory = make_wrapper(_default_timeout, _default_daemon)
        return wrapper_factory(func)

    # Case 2: Decorator with parentheses @log_command(timeout=X, daemon=Y)
    final_timeout = timeout if timeout is not None else _default_timeout
    final_daemon = daemon if daemon is not None else _default_daemon

    # Validate timeout positive integer constraint
    if final_timeout <= 0:
        raise ValueError("Analytics timeout must be greater than 0")

    return make_wrapper(final_timeout, final_daemon)
