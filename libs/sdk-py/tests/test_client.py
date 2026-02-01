import os

import pytest

from langgraph_sdk import get_client


def test_url_none_outside_server_raises_clear_error():
    """Test that url=None outside server raises helpful error."""
    if "LANGGRAPH_AUTO_FALLBACK" in os.environ:
        del os.environ["LANGGRAPH_AUTO_FALLBACK"]

    with pytest.raises(RuntimeError) as exc_info:
        get_client(url=None)
    error_message = str(exc_info.value)
    assert "Cannot use in-process connection" in error_message
    assert "url=None" in error_message
    assert "get_client(url='http://localhost:2024')" in error_message


def test_url_none_with_auto_fallback():
    """Test that auto-fallback works when enabled."""
    os.environ["LANGGRAPH_AUTO_FALLBACK"] = "true"

    try:
        with pytest.warns(UserWarning, match="Auto-falling back"):
            client = get_client(url=None)

        assert client.http.client.base_url == "http://localhost:2024"
    finally:
        del os.environ["LANGGRAPH_AUTO_FALLBACK"]


def test_explicit_url_works():
    """Test that explicit URL works correctly."""
    client = get_client(url="http://localhost:2024")
    assert client.http.client.base_url == "http://localhost:2024"
