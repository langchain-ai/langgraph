"""Regression tests for telemetry never blocking the CLI.

A black-holed/unreachable telemetry host previously hung the CLI at exit: the
POST had no timeout and ran in a non-daemon thread, so the interpreter waited
for it forever.
"""

import threading
from typing import Any
from unittest.mock import patch

import pytest

from langgraph_cli import analytics


def _sample_data() -> analytics.LogData:
    return {
        "os": "linux",
        "os_version": "1",
        "python_version": "3.11",
        "cli_version": "0.0.0",
        "cli_command": "build",
        "params": {},
    }


def test_log_data_uses_timeout() -> None:
    """The telemetry request must pass a bounded timeout."""
    with patch("langgraph_cli.analytics.urllib.request.urlopen") as mock_urlopen:
        analytics.log_data(_sample_data())

    assert mock_urlopen.call_count == 1
    _args, kwargs = mock_urlopen.call_args
    assert kwargs.get("timeout") == analytics.ANALYTICS_TIMEOUT_SECONDS
    assert isinstance(analytics.ANALYTICS_TIMEOUT_SECONDS, (int, float))
    assert analytics.ANALYTICS_TIMEOUT_SECONDS > 0


@pytest.mark.parametrize(
    "error",
    [
        TimeoutError("blackholed host"),
        OSError("network unreachable"),
        ConnectionResetError("reset by peer"),
        ValueError("unexpected response"),
    ],
)
def test_log_data_swallows_all_errors(error: Exception) -> None:
    """Telemetry is best-effort: timeouts and other errors must not surface."""
    with patch(
        "langgraph_cli.analytics.urllib.request.urlopen",
        side_effect=error,
    ):
        # Must not raise.
        analytics.log_data(_sample_data())


def test_log_command_spawns_daemon_thread() -> None:
    """The telemetry worker thread must be a daemon so a hung request never
    keeps the interpreter alive after the command finishes."""
    captured: dict[str, Any] = {}
    started = threading.Event()

    def fake_log_data(_data: analytics.LogData) -> None:
        captured["daemon"] = threading.current_thread().daemon
        started.set()

    @analytics.log_command
    def some_command(**kwargs: Any) -> str:
        return "ok"

    with patch("langgraph_cli.analytics.log_data", side_effect=fake_log_data):
        assert some_command() == "ok"
        assert started.wait(timeout=5), "telemetry thread never ran"

    assert captured["daemon"] is True
