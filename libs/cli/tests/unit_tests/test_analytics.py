import urllib.error
import urllib.request
from unittest.mock import MagicMock, patch

from langgraph_cli.analytics import (
    ANALYTICS_URLOPEN_TIMEOUT,
    LogData,
    log_command,
    log_data,
)

LOG_DATA: LogData = {
    "os": "Darwin",
    "os_version": "24.0.0",
    "python_version": "3.13.0",
    "cli_version": "0.4.31",
    "cli_command": "fake_command",
    "params": {},
}


def test_log_data_posts_payload_with_bounded_timeout():
    """Telemetry requests must be time-bounded so a stalled network cannot
    block the CLI (langchain-ai/langgraph#8074)."""
    with patch("urllib.request.urlopen") as mock_urlopen:
        log_data(LOG_DATA)

    mock_urlopen.assert_called_once()
    req = mock_urlopen.call_args.args[0]
    assert isinstance(req, urllib.request.Request)
    assert req.full_url == "https://kzrlppojinpcyyaipxnb.supabase.co/rest/v1/logs"
    assert req.method == "POST"
    assert mock_urlopen.call_args.kwargs["timeout"] == ANALYTICS_URLOPEN_TIMEOUT


def test_log_data_swallows_urlerror():
    with patch("urllib.request.urlopen", side_effect=urllib.error.URLError("boom")):
        log_data(LOG_DATA)  # must not raise


def test_log_data_swallows_timeout_error():
    with patch("urllib.request.urlopen", side_effect=TimeoutError("timed out")):
        log_data(LOG_DATA)  # must not raise


def test_log_data_swallows_unexpected_errors():
    """Telemetry is best-effort: any failure must be swallowed, never crash."""
    with patch("urllib.request.urlopen", side_effect=RuntimeError("unexpected")):
        log_data(LOG_DATA)  # must not raise


def test_log_command_spawns_daemon_thread_and_returns_result():
    """The telemetry worker must be a daemon thread so it cannot keep the
    process alive after the command body has returned."""
    mock_thread = MagicMock()
    with patch("threading.Thread", mock_thread):

        @log_command
        def fake_command(param="x"):
            return "ok"

        result = fake_command(param="x")

    assert result == "ok"
    mock_thread.assert_called_once()
    assert mock_thread.call_args.kwargs["daemon"] is True
    mock_thread.return_value.start.assert_called_once()


def test_log_command_opt_out_env_skips_telemetry(monkeypatch):
    """LANGGRAPH_CLI_NO_ANALYTICS=1 must fully opt out of telemetry."""
    monkeypatch.setenv("LANGGRAPH_CLI_NO_ANALYTICS", "1")
    mock_thread = MagicMock()
    with patch("threading.Thread", mock_thread):

        @log_command
        def fake_command(param="x"):
            return "ok"

        result = fake_command(param="x")

    assert result == "ok"
    mock_thread.assert_not_called()


def test_log_command_calls_func_when_env_unset(monkeypatch):
    """Without the opt-out env var, telemetry runs in the background and the
    command still executes."""
    monkeypatch.delenv("LANGGRAPH_CLI_NO_ANALYTICS", raising=False)
    mock_thread = MagicMock()
    with patch("threading.Thread", mock_thread):

        @log_command
        def fake_command(param="x"):
            return "ok"

        result = fake_command(param="x")

    assert result == "ok"
    mock_thread.assert_called_once()
