import os
from unittest.mock import patch

import pytest


@pytest.fixture(autouse=True)
def disable_analytics_env() -> None:
    """Disable analytics for unit tests LANGGRAPH_CLI_NO_ANALYTICS."""
    # First check if the environment variable is already set, if so, log a warning prior
    # to overriding it.
    if "LANGGRAPH_CLI_NO_ANALYTICS" in os.environ:
        print("⚠️ LANGGRAPH_CLI_NO_ANALYTICS is set. Overriding it for the test.")

    with patch.dict(os.environ, {"LANGGRAPH_CLI_NO_ANALYTICS": "0"}):
        yield
