"""Test that the version is defined."""

import langgraph_cli_install


def test_version():
    """Test that the version is defined."""
    assert langgraph_cli_install.__version__ is not None
