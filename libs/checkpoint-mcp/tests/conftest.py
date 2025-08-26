"""
Common test fixtures for MCP store tests
"""

import gc
import os
import socket
import subprocess
import sys
import time
from pathlib import Path
from unittest.mock import Mock

import pytest

# Verbosity level from environment or default to 0 (quiet)
VERBOSE = int(os.environ.get("TEST_VERBOSE", "0"))

# Track which test module is currently running
_current_test_module = None
_server_process = None


def vprint(message, level=1):
    """Print message only if verbosity level is high enough"""
    if VERBOSE >= level:
        print(message)


def _start_mcp_server():
    """Start the MCP server and return the process."""
    global _server_process

    print("Starting MCP server...")

    # Kill any existing server process first
    subprocess.run(["pkill", "-f", "mcp_server.py"], capture_output=True, text=True)
    time.sleep(1)

    # Use fixed port 8000
    port = 8000

    # Start the server
    server_script = Path(__file__).parent / "mcp_server.py"
    _server_process = subprocess.Popen([sys.executable, str(server_script)])

    # Wait for server to be ready with socket check
    server_ready = False
    max_attempts = 100  # 10 seconds total
    for attempt in range(max_attempts):
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(0.1)
            result = sock.connect_ex(("localhost", port))
            sock.close()
            if result == 0:
                server_ready = True
                break
        except Exception:
            pass
        time.sleep(0.1)

    if not server_ready:
        _server_process.terminate()
        _server_process.wait()
        raise RuntimeError(
            f"Server failed to start on port {port} after {max_attempts / 10} seconds"
        )

    print(f"MCP server started on port {port}")
    return _server_process


def _stop_mcp_server():
    """Stop the MCP server."""
    global _server_process

    if _server_process:
        print("Shutting down MCP server...")
        _server_process.terminate()
        try:
            _server_process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            _server_process.kill()
            _server_process.wait()
        _server_process = None
        print("MCP server cleanup completed")

        # Enhanced cleanup after server restart
        time.sleep(1.0)
        gc.collect()


@pytest.fixture(scope="function", autouse=True)
def mcp_server_per_module(request):
    """Restart MCP server when switching between test modules."""
    global _current_test_module, _server_process

    # Get the current test module name
    current_module = request.module.__name__

    # If we're switching to a different test module, restart the server
    if _current_test_module != current_module:
        print(f"\nSwitching from {_current_test_module} to {current_module}")

        # Stop existing server
        if _server_process:
            _stop_mcp_server()

        # Start fresh server for new module
        _start_mcp_server()
        _current_test_module = current_module

        print(f"Server restarted for module: {current_module}")

    # Return the port (always 8000)
    yield 8000

    # Cleanup between tests within same module
    print("Performing cleanup between tests...")
    time.sleep(0.3)
    gc.collect()


@pytest.fixture(scope="session", autouse=True)
def cleanup_on_exit():
    """Ensure server is stopped when test session ends."""
    yield
    _stop_mcp_server()


@pytest.fixture
def mcp_server(mcp_server_per_module):
    """Compatibility fixture for existing tests."""
    return mcp_server_per_module


@pytest.fixture
def mock_embeddings():
    """Create mock embeddings for testing"""
    embeddings = Mock()

    # IMPORTANT: Make the mock callable to work with EmbeddingsLambda
    # EmbeddingsLambda will call the mock directly as func(texts)
    def mock_embedding_func(texts):
        # Return embeddings based on number of texts
        return [[0.1, 0.2, 0.3, 0.4, 0.5] for _ in texts]

    # Make the mock callable by setting the return value for direct calls
    embeddings.side_effect = mock_embedding_func

    # Configure the mock to return proper iterables for method calls
    embeddings.embed_documents.return_value = [
        [0.1, 0.2, 0.3, 0.4, 0.5],  # First document embedding
        [0.6, 0.7, 0.8, 0.9, 1.0],  # Second document embedding
        [0.2, 0.3, 0.4, 0.5, 0.6],  # Third document embedding
    ]
    embeddings.embed_query.return_value = [0.1, 0.2, 0.3, 0.4, 0.5]  # Query embedding

    # Add any other properties that might be needed
    embeddings.dimension = 5

    return embeddings
