import time
import threading
from unittest.mock import patch, MagicMock
from langgraph_cli.analytics import log_command
from langgraph_cli.constants import DEFAULT_TIMEOUT
import urllib.request



def test_log_command_default_daemon_and_timeout():
    """Test the default daemon and timeout parameters of the log_command decorator
    to ensure backward compatibility with existing code"""
    mock_thread = MagicMock()
    with patch("threading.Thread", mock_thread):
        @log_command()
        def test_func():
            return "ok"

        test_func()

        # Verify default thread parameters: daemon=True, timeout=DEFAULT_TIMEOUT
        mock_thread.assert_called_once()
        _, thread_kwargs = mock_thread.call_args
        assert thread_kwargs["daemon"] is True

        # Verify timeout parameter passed to log_data
        log_data_args = thread_kwargs["args"]
        assert log_data_args[1] == DEFAULT_TIMEOUT


def test_log_command_custom_daemon_and_timeout():
    """Test custom daemon and timeout parameters of the log_command decorator
    to ensure parameters are passed and take effect correctly"""
    mock_thread = MagicMock()
    with patch("threading.Thread", mock_thread):
        @log_command(timeout=8, daemon=False)
        def test_func():
            return "ok"

        test_func()

        # Verify custom thread parameters
        _, thread_kwargs = mock_thread.call_args
        assert thread_kwargs["daemon"] is False

        # Verify custom timeout parameter
        log_data_args = thread_kwargs["args"]
        assert log_data_args[1] == 8


def test_log_command_daemon_thread_exit():
    """Verify that daemon threads do not block process exit"""

    def slow_urlopen(req, **kwargs):
        time.sleep(5)  # Longer sleep to verify daemon threads don't block process exit
        raise TimeoutError("simulated telemetry stall")

    @log_command(daemon=True)
    def command(**kwargs):
        return "done"

    start = time.monotonic()
    with patch.object(urllib.request, "urlopen", slow_urlopen):
        assert command(config=None) == "done"

    # Verify process exits immediately (daemon thread runs in background without blocking)
    elapsed = time.monotonic() - start
    assert elapsed < 1


def test_log_command_urlopen_timeout_no_block():
    """Reproduce and verify #8074: Main thread blocks when urlopen times out
    Issue #8074: CLI analytics can keep commands alive because urlopen has no timeout.

    Set daemon=False to disable daemon threads and actively wait for thread completion.
    Since urllib.error.URLError and TimeoutError exceptions are handled silently,
    no exceptions will be thrown here and the program can continue to run normally."""
    blocking_time = 3
    thread_ref = None
    # Wrap Thread to capture the instance
    original_thread = threading.Thread

    def slow_urlopen(req, **kwargs):
        time.sleep(blocking_time)  # Simulate long-running urlopen call
        raise TimeoutError("simulated telemetry stall")

    def capture_thread(*args, **kwargs):
        nonlocal thread_ref
        t = original_thread(*args, **kwargs)
        thread_ref = t
        return t

    with patch("threading.Thread", capture_thread):
        # Keep original logic unchanged
        @log_command(timeout=blocking_time, daemon=False)
        def command(**kwargs):
            return "done"

        start = time.monotonic()
        with patch.object(urllib.request, "urlopen", slow_urlopen):
            result = command(config=None)
            assert result == "done"
        elapsed = time.monotonic() - start
        assert elapsed < 1
        print(f"elapsed in main thread: {elapsed:.3f}s")
        # Manual join to force waiting for 3 seconds
        thread_ref.join()
        total_time = time.monotonic() - start
        assert total_time >= blocking_time
        print(f"times in main thread: {total_time:.3f}s")