"""Tests for the crons client."""

from __future__ import annotations

import json
from datetime import datetime, timezone

import httpx
import pytest

from langgraph_sdk._shared.utilities import (
    _parse_cron_field,
    _validate_cron_schedule,
)
from langgraph_sdk.client import (
    CronClient,
    HttpClient,
    SyncCronClient,
    SyncHttpClient,
)


def _cron_payload() -> dict[str, object]:
    """Return a mock cron response payload."""
    return {
        "run_id": "run_123",
        "thread_id": "thread_123",
        "assistant_id": "asst_123",
        "created_at": "2024-01-01T00:00:00Z",
        "updated_at": "2024-01-02T00:00:00Z",
        "status": "success",
        "metadata": {},
        "multitask_strategy": "reject",
    }


@pytest.mark.asyncio
async def test_async_create_for_thread():
    """Test that CronClient.create_for_thread works without end_time."""
    cron = _cron_payload()

    async def handler(request: httpx.Request) -> httpx.Response:
        assert request.method == "POST"
        assert request.url.path == "/threads/thread_123/runs/crons"

        # Parse the request body
        body = json.loads(request.content)
        assert body["schedule"] == "0 0 * * *"
        assert body["assistant_id"] == "asst_123"
        assert "end_time" not in body  # Should be filtered out by the None check

        return httpx.Response(200, json=cron)

    transport = httpx.MockTransport(handler)
    async with httpx.AsyncClient(
        transport=transport, base_url="https://example.com"
    ) as client:
        http_client = HttpClient(client)
        cron_client = CronClient(http_client)
        result = await cron_client.create_for_thread(
            thread_id="thread_123",
            assistant_id="asst_123",
            schedule="0 0 * * *",
        )

    assert result == cron


@pytest.mark.asyncio
async def test_async_create_for_thread_with_end_time():
    """Test that CronClient.create_for_thread includes end_time in the payload."""
    cron = _cron_payload()
    end_time = datetime(2025, 12, 31, 23, 59, 59, tzinfo=timezone.utc)

    async def handler(request: httpx.Request) -> httpx.Response:
        assert request.method == "POST"
        assert request.url.path == "/threads/thread_123/runs/crons"

        # Parse the request body
        body = json.loads(request.content)
        assert body["schedule"] == "0 0 * * *"
        assert body["assistant_id"] == "asst_123"
        assert body["end_time"] == "2025-12-31T23:59:59+00:00"

        return httpx.Response(200, json=cron)

    transport = httpx.MockTransport(handler)
    async with httpx.AsyncClient(
        transport=transport, base_url="https://example.com"
    ) as client:
        http_client = HttpClient(client)
        cron_client = CronClient(http_client)
        result = await cron_client.create_for_thread(
            thread_id="thread_123",
            assistant_id="asst_123",
            schedule="0 0 * * *",
            end_time=end_time,
        )

    assert result == cron


@pytest.mark.asyncio
async def test_async_create():
    """Test that CronClient.create works without end_time."""
    cron = _cron_payload()

    async def handler(request: httpx.Request) -> httpx.Response:
        assert request.method == "POST"
        assert request.url.path == "/runs/crons"

        # Parse the request body
        body = json.loads(request.content)
        assert body["schedule"] == "0 12 * * *"
        assert body["assistant_id"] == "asst_456"
        assert "end_time" not in body  # Should be filtered out by the None check

        return httpx.Response(200, json=cron)

    transport = httpx.MockTransport(handler)
    async with httpx.AsyncClient(
        transport=transport, base_url="https://example.com"
    ) as client:
        http_client = HttpClient(client)
        cron_client = CronClient(http_client)
        result = await cron_client.create(
            assistant_id="asst_456",
            schedule="0 12 * * *",
        )

    assert result == cron


@pytest.mark.asyncio
async def test_async_create_with_end_time():
    """Test that CronClient.create includes end_time in the payload."""
    cron = _cron_payload()
    end_time = datetime(2025, 6, 15, 12, 0, 0, tzinfo=timezone.utc)

    async def handler(request: httpx.Request) -> httpx.Response:
        assert request.method == "POST"
        assert request.url.path == "/runs/crons"

        # Parse the request body
        body = json.loads(request.content)
        assert body["schedule"] == "0 12 * * *"
        assert body["assistant_id"] == "asst_456"
        assert body["end_time"] == "2025-06-15T12:00:00+00:00"

        return httpx.Response(200, json=cron)

    transport = httpx.MockTransport(handler)
    async with httpx.AsyncClient(
        transport=transport, base_url="https://example.com"
    ) as client:
        http_client = HttpClient(client)
        cron_client = CronClient(http_client)
        result = await cron_client.create(
            assistant_id="asst_456",
            schedule="0 12 * * *",
            end_time=end_time,
        )

    assert result == cron


def test_sync_create_for_thread():
    """Test that SyncCronClient.create_for_thread works without end_time."""
    cron = _cron_payload()

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.method == "POST"
        assert request.url.path == "/threads/thread_123/runs/crons"

        # Parse the request body
        body = json.loads(request.content)
        assert body["schedule"] == "0 0 * * *"
        assert body["assistant_id"] == "asst_123"
        assert "end_time" not in body  # Should be filtered out by the None check

        return httpx.Response(200, json=cron)

    transport = httpx.MockTransport(handler)
    with httpx.Client(transport=transport, base_url="https://example.com") as client:
        http_client = SyncHttpClient(client)
        cron_client = SyncCronClient(http_client)
        result = cron_client.create_for_thread(
            thread_id="thread_123",
            assistant_id="asst_123",
            schedule="0 0 * * *",
        )

    assert result == cron


def test_sync_create_for_thread_with_end_time():
    """Test that SyncCronClient.create_for_thread includes end_time in the payload."""
    cron = _cron_payload()
    end_time = datetime(2025, 12, 31, 23, 59, 59, tzinfo=timezone.utc)

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.method == "POST"
        assert request.url.path == "/threads/thread_123/runs/crons"

        # Parse the request body
        body = json.loads(request.content)
        assert body["schedule"] == "0 0 * * *"
        assert body["assistant_id"] == "asst_123"
        assert body["end_time"] == "2025-12-31T23:59:59+00:00"

        return httpx.Response(200, json=cron)

    transport = httpx.MockTransport(handler)
    with httpx.Client(transport=transport, base_url="https://example.com") as client:
        http_client = SyncHttpClient(client)
        cron_client = SyncCronClient(http_client)
        result = cron_client.create_for_thread(
            thread_id="thread_123",
            assistant_id="asst_123",
            schedule="0 0 * * *",
            end_time=end_time,
        )

    assert result == cron


def test_sync_create():
    """Test that SyncCronClient.create works without end_time."""
    cron = _cron_payload()

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.method == "POST"
        assert request.url.path == "/runs/crons"

        # Parse the request body
        body = json.loads(request.content)
        assert body["schedule"] == "0 12 * * *"
        assert body["assistant_id"] == "asst_456"
        assert "end_time" not in body  # Should be filtered out by the None check

        return httpx.Response(200, json=cron)

    transport = httpx.MockTransport(handler)
    with httpx.Client(transport=transport, base_url="https://example.com") as client:
        http_client = SyncHttpClient(client)
        cron_client = SyncCronClient(http_client)
        result = cron_client.create(
            assistant_id="asst_456",
            schedule="0 12 * * *",
        )

    assert result == cron


def test_sync_create_with_end_time():
    """Test that SyncCronClient.create includes end_time in the payload."""
    cron = _cron_payload()
    end_time = datetime(2025, 6, 15, 12, 0, 0, tzinfo=timezone.utc)

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.method == "POST"
        assert request.url.path == "/runs/crons"

        # Parse the request body
        body = json.loads(request.content)
        assert body["schedule"] == "0 12 * * *"
        assert body["assistant_id"] == "asst_456"
        assert body["end_time"] == "2025-06-15T12:00:00+00:00"

        return httpx.Response(200, json=cron)

    transport = httpx.MockTransport(handler)
    with httpx.Client(transport=transport, base_url="https://example.com") as client:
        http_client = SyncHttpClient(client)
        cron_client = SyncCronClient(http_client)
        result = cron_client.create(
            assistant_id="asst_456",
            schedule="0 12 * * *",
            end_time=end_time,
        )

    assert result == cron


@pytest.mark.parametrize(
    "enabled_value",
    [True, False],
    ids=["enabled", "disabled"],
)
def test_sync_create_with_enabled_parameter(enabled_value):
    """Test that SyncCronClient.create includes enabled parameter in the payload."""
    cron = _cron_payload()

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.method == "POST"
        assert request.url.path == "/runs/crons"

        body = json.loads(request.content)
        assert body["schedule"] == "0 12 * * *"
        assert body["assistant_id"] == "asst_456"
        assert body["enabled"] == enabled_value

        return httpx.Response(200, json=cron)

    transport = httpx.MockTransport(handler)
    with httpx.Client(transport=transport, base_url="https://example.com") as client:
        http_client = SyncHttpClient(client)
        cron_client = SyncCronClient(http_client)
        result = cron_client.create(
            assistant_id="asst_456",
            schedule="0 12 * * *",
            enabled=enabled_value,
        )

    assert result == cron


def _cron_response() -> dict[str, object]:
    """Return a mock Cron object response."""
    return {
        "cron_id": "cron_123",
        "assistant_id": "asst_123",
        "thread_id": "thread_123",
        "on_run_completed": None,
        "end_time": "2025-12-31T23:59:59+00:00",
        "schedule": "0 10 * * *",
        "created_at": "2024-01-01T00:00:00Z",
        "updated_at": "2024-01-02T00:00:00Z",
        "payload": {},
        "user_id": None,
        "next_run_date": "2024-01-03T10:00:00Z",
        "metadata": {},
        "enabled": True,
    }


@pytest.mark.asyncio
async def test_async_update():
    """Test that CronClient.update works with schedule and enabled parameters."""
    cron = _cron_response()

    async def handler(request: httpx.Request) -> httpx.Response:
        assert request.method == "PATCH"
        assert request.url.path == "/runs/crons/cron_123"

        # Parse the request body
        body = json.loads(request.content)
        assert body["schedule"] == "0 10 * * *"
        assert body["enabled"] is False
        assert "end_time" not in body  # Should be filtered out by the None check

        return httpx.Response(200, json=cron)

    transport = httpx.MockTransport(handler)
    async with httpx.AsyncClient(
        transport=transport, base_url="https://example.com"
    ) as client:
        http_client = HttpClient(client)
        cron_client = CronClient(http_client)
        result = await cron_client.update(
            cron_id="cron_123",
            schedule="0 10 * * *",
            enabled=False,
        )

    assert result == cron


@pytest.mark.asyncio
async def test_async_update_with_end_time():
    """Test that CronClient.update includes end_time in the payload."""
    cron = _cron_response()
    end_time = datetime(2025, 12, 31, 23, 59, 59, tzinfo=timezone.utc)

    async def handler(request: httpx.Request) -> httpx.Response:
        assert request.method == "PATCH"
        assert request.url.path == "/runs/crons/cron_123"

        # Parse the request body
        body = json.loads(request.content)
        assert body["schedule"] == "0 10 * * *"
        assert body["end_time"] == "2025-12-31T23:59:59+00:00"
        assert body["enabled"] is True

        return httpx.Response(200, json=cron)

    transport = httpx.MockTransport(handler)
    async with httpx.AsyncClient(
        transport=transport, base_url="https://example.com"
    ) as client:
        http_client = HttpClient(client)
        cron_client = CronClient(http_client)
        result = await cron_client.update(
            cron_id="cron_123",
            schedule="0 10 * * *",
            end_time=end_time,
            enabled=True,
        )

    assert result == cron


def test_sync_update():
    """Test that SyncCronClient.update works with schedule and enabled parameters."""
    cron = _cron_response()

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.method == "PATCH"
        assert request.url.path == "/runs/crons/cron_123"

        # Parse the request body
        body = json.loads(request.content)
        assert body["schedule"] == "0 10 * * *"
        assert body["enabled"] is False
        assert "end_time" not in body  # Should be filtered out by the None check

        return httpx.Response(200, json=cron)

    transport = httpx.MockTransport(handler)
    with httpx.Client(transport=transport, base_url="https://example.com") as client:
        http_client = SyncHttpClient(client)
        cron_client = SyncCronClient(http_client)
        result = cron_client.update(
            cron_id="cron_123",
            schedule="0 10 * * *",
            enabled=False,
        )

    assert result == cron


def test_sync_update_with_end_time():
    """Test that SyncCronClient.update includes end_time in the payload."""
    cron = _cron_response()
    end_time = datetime(2025, 12, 31, 23, 59, 59, tzinfo=timezone.utc)

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.method == "PATCH"
        assert request.url.path == "/runs/crons/cron_123"

        # Parse the request body
        body = json.loads(request.content)
        assert body["schedule"] == "0 10 * * *"
        assert body["end_time"] == "2025-12-31T23:59:59+00:00"
        assert body["enabled"] is True

        return httpx.Response(200, json=cron)

    transport = httpx.MockTransport(handler)
    with httpx.Client(transport=transport, base_url="https://example.com") as client:
        http_client = SyncHttpClient(client)
        cron_client = SyncCronClient(http_client)
        result = cron_client.update(
            cron_id="cron_123",
            schedule="0 10 * * *",
            end_time=end_time,
            enabled=True,
        )

    assert result == cron


@pytest.mark.parametrize(
    "enabled_value",
    [True, False],
    ids=["enabled", "disabled"],
)
def test_sync_update_with_enabled_parameter(enabled_value):
    """Test that SyncCronClient.update includes enabled parameter in the payload."""
    cron = _cron_response()

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.method == "PATCH"
        assert request.url.path == "/runs/crons/cron_456"

        body = json.loads(request.content)
        assert body["enabled"] == enabled_value
        assert "schedule" not in body  # Only enabled is set

        return httpx.Response(200, json=cron)

    transport = httpx.MockTransport(handler)
    with httpx.Client(transport=transport, base_url="https://example.com") as client:
        http_client = SyncHttpClient(client)
        cron_client = SyncCronClient(http_client)
        result = cron_client.update(
            cron_id="cron_456",
            enabled=enabled_value,
        )

    assert result == cron


# ---------------------------------------------------------------------------
# Schedule validation tests
# ---------------------------------------------------------------------------


class TestParseCronField:
    """Tests for _parse_cron_field helper."""

    def test_wildcard_returns_none(self):
        assert _parse_cron_field("*", 1, 12) is None

    def test_single_value(self):
        assert _parse_cron_field("5", 1, 31) == [5]

    def test_comma_list(self):
        assert _parse_cron_field("1,15,28", 1, 31) == [1, 15, 28]

    def test_range(self):
        assert _parse_cron_field("3-6", 1, 12) == [3, 4, 5, 6]

    def test_step(self):
        assert _parse_cron_field("*/3", 1, 12) == [1, 4, 7, 10]

    def test_range_with_step(self):
        assert _parse_cron_field("1-10/3", 1, 31) == [1, 4, 7, 10]

    def test_out_of_range_filtered(self):
        assert _parse_cron_field("30,31", 1, 12) == []


class TestValidateCronSchedule:
    """Tests for _validate_cron_schedule."""

    # --- impossible schedules that should raise ValueError ---

    def test_feb_31(self):
        with pytest.raises(ValueError, match="never exist"):
            _validate_cron_schedule("0 23 31 2 *")

    def test_feb_30(self):
        with pytest.raises(ValueError, match="never exist"):
            _validate_cron_schedule("0 0 30 2 *")

    def test_feb_30_and_31(self):
        with pytest.raises(ValueError, match="never exist"):
            _validate_cron_schedule("0 0 30,31 2 *")

    def test_apr_31(self):
        with pytest.raises(ValueError, match="never exist"):
            _validate_cron_schedule("0 0 31 4 *")

    def test_jun_31(self):
        with pytest.raises(ValueError, match="never exist"):
            _validate_cron_schedule("0 0 31 6 *")

    def test_sep_31(self):
        with pytest.raises(ValueError, match="never exist"):
            _validate_cron_schedule("0 0 31 9 *")

    def test_nov_31(self):
        with pytest.raises(ValueError, match="never exist"):
            _validate_cron_schedule("0 0 31 11 *")

    def test_31st_in_all_30day_months(self):
        """Day 31 in months 4,6,9,11 — none of these months have 31 days."""
        with pytest.raises(ValueError, match="never exist"):
            _validate_cron_schedule("0 0 31 4,6,9,11 *")

    # --- valid schedules that should NOT raise ---

    def test_valid_every_minute(self):
        _validate_cron_schedule("* * * * *")  # should not raise

    def test_valid_daily(self):
        _validate_cron_schedule("0 0 * * *")

    def test_valid_monthly_15th(self):
        _validate_cron_schedule("0 0 15 * *")

    def test_valid_jan_31(self):
        _validate_cron_schedule("0 0 31 1 *")  # Jan has 31 days

    def test_valid_mar_31(self):
        _validate_cron_schedule("0 0 31 3 *")  # Mar has 31 days

    def test_valid_feb_29(self):
        """Feb 29 is valid — it occurs in leap years."""
        _validate_cron_schedule("0 0 29 2 *")

    def test_valid_feb_28(self):
        _validate_cron_schedule("0 0 28 2 *")

    def test_wildcard_day(self):
        _validate_cron_schedule("0 0 * 2 *")

    def test_wildcard_month(self):
        _validate_cron_schedule("0 0 31 * *")

    def test_day_range_with_some_valid(self):
        """Day 28-31 in Feb: 28 and 29 are valid, so this should pass."""
        _validate_cron_schedule("0 0 28-31 2 *")

    def test_multiple_months_some_valid(self):
        """Day 31 in months 1,2: Jan 31 is valid even though Feb 31 is not."""
        _validate_cron_schedule("0 0 31 1,2 *")

    def test_non_standard_format_ignored(self):
        """Non 5-field expressions are passed through to server."""
        _validate_cron_schedule("@daily")  # should not raise


@pytest.mark.asyncio
async def test_async_create_rejects_impossible_schedule():
    """Async create should raise ValueError for impossible schedules."""
    async with httpx.AsyncClient(
        transport=httpx.MockTransport(lambda r: httpx.Response(200, json={})),
        base_url="https://example.com",
    ) as client:
        cron_client = CronClient(HttpClient(client))
        with pytest.raises(ValueError, match="never exist"):
            await cron_client.create(
                assistant_id="asst_123",
                schedule="0 23 31 2 *",
            )


@pytest.mark.asyncio
async def test_async_create_for_thread_rejects_impossible_schedule():
    """Async create_for_thread should raise ValueError for impossible schedules."""
    async with httpx.AsyncClient(
        transport=httpx.MockTransport(lambda r: httpx.Response(200, json={})),
        base_url="https://example.com",
    ) as client:
        cron_client = CronClient(HttpClient(client))
        with pytest.raises(ValueError, match="never exist"):
            await cron_client.create_for_thread(
                thread_id="thread_123",
                assistant_id="asst_123",
                schedule="0 23 31 2 *",
            )


@pytest.mark.asyncio
async def test_async_update_rejects_impossible_schedule():
    """Async update should raise ValueError for impossible schedules."""
    async with httpx.AsyncClient(
        transport=httpx.MockTransport(lambda r: httpx.Response(200, json={})),
        base_url="https://example.com",
    ) as client:
        cron_client = CronClient(HttpClient(client))
        with pytest.raises(ValueError, match="never exist"):
            await cron_client.update(
                cron_id="cron_123",
                schedule="0 0 31 4 *",
            )


def test_sync_create_rejects_impossible_schedule():
    """Sync create should raise ValueError for impossible schedules."""
    with httpx.Client(
        transport=httpx.MockTransport(lambda r: httpx.Response(200, json={})),
        base_url="https://example.com",
    ) as client:
        cron_client = SyncCronClient(SyncHttpClient(client))
        with pytest.raises(ValueError, match="never exist"):
            cron_client.create(
                assistant_id="asst_456",
                schedule="0 0 30 2 *",
            )


def test_sync_create_for_thread_rejects_impossible_schedule():
    """Sync create_for_thread should raise ValueError for impossible schedules."""
    with httpx.Client(
        transport=httpx.MockTransport(lambda r: httpx.Response(200, json={})),
        base_url="https://example.com",
    ) as client:
        cron_client = SyncCronClient(SyncHttpClient(client))
        with pytest.raises(ValueError, match="never exist"):
            cron_client.create_for_thread(
                thread_id="thread_123",
                assistant_id="asst_123",
                schedule="0 0 30 2 *",
            )


def test_sync_update_rejects_impossible_schedule():
    """Sync update should raise ValueError for impossible schedules."""
    with httpx.Client(
        transport=httpx.MockTransport(lambda r: httpx.Response(200, json={})),
        base_url="https://example.com",
    ) as client:
        cron_client = SyncCronClient(SyncHttpClient(client))
        with pytest.raises(ValueError, match="never exist"):
            cron_client.update(
                cron_id="cron_456",
                schedule="0 0 31 6 *",
            )


def test_sync_update_no_schedule_skips_validation():
    """Sync update without schedule should not trigger validation."""
    cron = _cron_response()

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=cron)

    with httpx.Client(
        transport=httpx.MockTransport(handler),
        base_url="https://example.com",
    ) as client:
        cron_client = SyncCronClient(SyncHttpClient(client))
        result = cron_client.update(cron_id="cron_456", enabled=False)

    assert result == cron
