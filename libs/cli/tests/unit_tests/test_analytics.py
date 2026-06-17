from unittest.mock import MagicMock, patch

from langgraph_cli.analytics import _ANALYTICS_TIMEOUT_SECONDS, log_data


@patch("urllib.request.urlopen")
def test_log_data_uses_bounded_timeout(mock_urlopen: MagicMock) -> None:
    log_data(
        {
            "os": "Windows",
            "os_version": "test",
            "python_version": "3.12",
            "cli_version": "0.0.0",
            "cli_command": "test",
            "params": {},
        }
    )

    mock_urlopen.assert_called_once()
    assert mock_urlopen.call_args.kwargs["timeout"] == _ANALYTICS_TIMEOUT_SECONDS
