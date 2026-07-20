"""Regression tests for #8383 and #8378."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from langgraph_sdk import get_sync_client
from langgraph_sdk._sync.runs import SyncRunsClient


def test_sync_wait_raises_on_error_envelope_when_raise_error_true():
    response = {
        "__error__": {
            "error": "RuntimeError",
            "message": "boom",
        }
    }
    http = MagicMock()
    http.request_reconnect.return_value = response
    client = SyncRunsClient(http)

    with pytest.raises(Exception, match="RuntimeError: boom"):
        client.wait(
            None,
            "agent",
            input={"messages": []},
            raise_error=True,
        )

    # raise_error must not be forwarded as a request body field
    kwargs = http.request_reconnect.call_args.kwargs
    assert "raise_error" not in (kwargs.get("json") or {})


def test_sync_wait_returns_error_envelope_when_raise_error_false():
    response = {
        "__error__": {
            "error": "RuntimeError",
            "message": "boom",
        }
    }
    http = MagicMock()
    http.request_reconnect.return_value = response
    client = SyncRunsClient(http)

    result = client.wait(
        None,
        "agent",
        input={"messages": []},
        raise_error=False,
    )
    assert result == response


def test_reserved_x_api_key_header_rejected_case_insensitively():
    with pytest.raises(ValueError, match="reserved header"):
        get_sync_client(
            url="http://localhost:8123",
            api_key="configured",
            headers={"X-API-Key": "custom"},
        )
