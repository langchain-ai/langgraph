"""Tests that the reserved-header guard is case-insensitive."""

import pytest

from langgraph_sdk import get_client, get_sync_client


class TestReservedHeaderGuard:
    """``x-api-key`` must be rejected as a custom header regardless of casing."""

    @pytest.mark.parametrize("header_name", ["x-api-key", "X-API-Key", "X-Api-Key"])
    def test_get_client_rejects_reserved_header(self, header_name):
        with pytest.raises(ValueError, match="reserved header"):
            get_client(url="http://localhost:8123", headers={header_name: "custom"})

    @pytest.mark.parametrize("header_name", ["x-api-key", "X-API-Key", "X-Api-Key"])
    def test_get_sync_client_rejects_reserved_header(self, header_name):
        with pytest.raises(ValueError, match="reserved header"):
            get_sync_client(
                url="http://localhost:8123", headers={header_name: "custom"}
            )

    def test_get_sync_client_allows_non_reserved_header(self):
        client = get_sync_client(
            url="http://localhost:8123", headers={"X-Custom": "value"}
        )
        assert client.http.client.headers["X-Custom"] == "value"
        client.close()
