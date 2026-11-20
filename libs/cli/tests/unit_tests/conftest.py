import json
import os
from functools import partial
from pathlib import Path
from unittest.mock import patch

import httpx
import pytest

import langgraph_cli.deploy as deploy_module
from langgraph_cli.host_backend import LangSmithApiClient


@pytest.fixture
def deployment_metrics_response():
    return json.loads(Path(__file__).with_name("deployment_metrics.json").read_text())


@pytest.fixture(autouse=True)
def disable_analytics_env() -> None:
    """Disable analytics for unit tests LANGGRAPH_CLI_NO_ANALYTICS."""
    # First check if the environment variable is already set, if so, log a warning prior
    # to overriding it.
    if "LANGGRAPH_CLI_NO_ANALYTICS" in os.environ:
        print("⚠️ LANGGRAPH_CLI_NO_ANALYTICS is set. Overriding it for the test.")

    with patch.dict(os.environ, {"LANGGRAPH_CLI_NO_ANALYTICS": "1"}):
        yield


@pytest.fixture(autouse=True)
def no_discovered_workspaces(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep unit tests off the LangSmith API: discovery finds no workspaces."""
    transport = httpx.MockTransport(lambda request: httpx.Response(200, json=[]))
    monkeypatch.setattr(
        deploy_module,
        "LangSmithApiClient",
        partial(LangSmithApiClient, transport=transport),
    )
