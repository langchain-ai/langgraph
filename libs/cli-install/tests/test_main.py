"""Tests for the main module."""

import sys
from unittest.mock import MagicMock, patch

from langgraph_cli_install.main import get_latest_python_version, main


def test_get_latest_python_version():
    """Test that the get_latest_python_version function returns a string."""
    with patch("subprocess.run") as mock_run:
        mock_result = MagicMock()
        mock_result.stdout = "python3.12"
        mock_run.return_value = mock_result

        with patch("uv.find_uv_bin", return_value="/path/to/uv"):
            version = get_latest_python_version()
            assert isinstance(version, str)
            assert "python" in version


def test_get_latest_python_version_fallback():
    """Test fallback to current version when 3.12 is not available."""
    with patch("subprocess.run") as mock_run:
        mock_result = MagicMock()
        mock_result.stdout = "python3.8"  # No 3.12 here
        mock_run.return_value = mock_result

        with patch("uv.find_uv_bin", return_value="/path/to/uv"):
            # Mock sys.version_info
            old_version_info = sys.version_info
            sys.version_info = MagicMock()
            sys.version_info.major = 3
            sys.version_info.minor = 9

            try:
                version = get_latest_python_version()
                assert isinstance(version, str)
                assert "python3.9" in version
            finally:
                # Restore original version_info
                sys.version_info = old_version_info


def test_main_exception():
    """Test main function handles exceptions."""
    with patch("uv.find_uv_bin", side_effect=Exception("Test error")):
        with patch("sys.exit") as mock_exit:
            main()
            mock_exit.assert_called_once_with(1)
