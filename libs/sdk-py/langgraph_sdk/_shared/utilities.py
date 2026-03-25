"""Shared utility functions for async and sync clients."""

from __future__ import annotations

import calendar
import functools
import os
import re
from collections.abc import Mapping
from datetime import tzinfo
from typing import TYPE_CHECKING, Any, cast

import httpx

import langgraph_sdk
from langgraph_sdk.schema import RunCreateMetadata

if TYPE_CHECKING:
    from zoneinfo import ZoneInfo

RESERVED_HEADERS = ("x-api-key",)

NOT_PROVIDED = cast(None, object())


def _get_api_key(api_key: str | None = NOT_PROVIDED) -> str | None:
    """Get the API key from the environment.
    Precedence:
        1. explicit string argument
        2. LANGGRAPH_API_KEY (if api_key not provided)
        3. LANGSMITH_API_KEY (if api_key not provided)
        4. LANGCHAIN_API_KEY (if api_key not provided)

    Args:
        api_key: The API key to use. Can be:
            - A string: use this exact API key
            - None: explicitly skip loading from environment
            - NOT_PROVIDED (default): auto-load from environment variables
    """
    if isinstance(api_key, str):
        return api_key
    if api_key is NOT_PROVIDED:
        # api_key is not explicitly provided, try to load from environment
        for prefix in ["LANGGRAPH", "LANGSMITH", "LANGCHAIN"]:
            if env := os.getenv(f"{prefix}_API_KEY"):
                return env.strip().strip('"').strip("'")
    # api_key is explicitly None, don't load from environment
    return None


def _get_headers(
    api_key: str | None,
    custom_headers: Mapping[str, str] | None,
) -> dict[str, str]:
    """Combine api_key and custom user-provided headers."""
    custom_headers = custom_headers or {}
    for header in RESERVED_HEADERS:
        if header in custom_headers:
            raise ValueError(f"Cannot set reserved header '{header}'")

    headers = {
        "User-Agent": f"langgraph-sdk-py/{langgraph_sdk.__version__}",
        **custom_headers,
    }
    resolved_api_key = _get_api_key(api_key)
    if resolved_api_key:
        headers["x-api-key"] = resolved_api_key

    return headers


def _orjson_default(obj: Any) -> Any:
    is_class = isinstance(obj, type)
    if hasattr(obj, "model_dump") and callable(obj.model_dump):
        if is_class:
            raise TypeError(
                f"Cannot JSON-serialize type object: {obj!r}. Did you mean to pass an instance of the object instead?"
                f"\nReceived type: {obj!r}"
            )
        return obj.model_dump()
    elif hasattr(obj, "dict") and callable(obj.dict):
        if is_class:
            raise TypeError(
                f"Cannot JSON-serialize type object: {obj!r}. Did you mean to pass an instance of the object instead?"
                f"\nReceived type: {obj!r}"
            )
        return obj.dict()
    elif isinstance(obj, (set, frozenset)):
        return list(obj)
    else:
        raise TypeError(f"Object of type {type(obj)} is not JSON serializable")


# Compiled regex pattern for extracting run metadata from Content-Location header
_RUN_METADATA_PATTERN = re.compile(
    r"(\/threads\/(?P<thread_id>.+))?\/runs\/(?P<run_id>.+)"
)


def _get_run_metadata_from_response(
    response: httpx.Response,
) -> RunCreateMetadata | None:
    """Extract run metadata from the response headers."""
    if (content_location := response.headers.get("Content-Location")) and (
        match := _RUN_METADATA_PATTERN.search(content_location)
    ):
        return RunCreateMetadata(
            run_id=match.group("run_id"),
            thread_id=match.group("thread_id") or None,
        )

    return None


def _sse_to_v2_dict(event: str, data: Any) -> dict[str, Any] | None:
    """Convert an SSE event+data pair into a v2 stream part dict.

    Returns None for ``end`` events (signals end of stream).
    """
    if event == "end":
        return None
    parts = event.split("|")
    event_type = parts[0]
    ns = parts[1:] if len(parts) > 1 else []
    result: dict[str, Any] = {"type": event_type, "ns": ns, "data": data}
    if event_type == "values" and isinstance(data, dict):
        result["interrupts"] = data.pop("__interrupt__", [])
    else:
        result["interrupts"] = []
    return result


def _resolve_timezone(tz: str | tzinfo | ZoneInfo | None) -> str | None:
    """Convert a timezone argument to an IANA timezone string.

    Accepts:
        - A string (returned as-is, assumed to be an IANA timezone name)
        - A ``datetime.tzinfo`` instance (e.g. ``zoneinfo.ZoneInfo("America/New_York")``,
          ``datetime.timezone.utc``). The ``key`` attribute is used if available,
          otherwise ``tzname(None)`` is used.
        - ``None`` (returned as ``None``)
    """
    if tz is None or isinstance(tz, str):
        return tz
    if isinstance(tz, tzinfo):
        # ZoneInfo objects have a .key attribute with the IANA name
        if hasattr(tz, "key"):
            return tz.key  # type: ignore[union-attr]
        # Fall back to tzname for fixed-offset timezones like datetime.timezone.utc
        name = tz.tzname(None)
        if name is not None:
            return name
        raise ValueError(
            f"Cannot determine timezone name from {tz!r}. "
            "Use a zoneinfo.ZoneInfo instance or pass a string like 'America/New_York'."
        )
    raise TypeError(
        f"Expected str, datetime.tzinfo, or None for timezone, got {type(tz).__name__}"
    )


def _provided_vals(d: Mapping[str, Any]) -> dict[str, Any]:
    return {k: v for k, v in d.items() if v is not None}


_registered_transports: list[httpx.ASGITransport] = []


# Do not move; this is used in the server.
def configure_loopback_transports(app: Any) -> None:
    for transport in _registered_transports:
        transport.app = app


@functools.lru_cache(maxsize=1)
def get_asgi_transport() -> type[httpx.ASGITransport]:
    try:
        from langgraph_api import asgi_transport  # type: ignore[unresolved-import]

        return asgi_transport.ASGITransport
    except ImportError:
        # Older versions of the server
        return httpx.ASGITransport


def _parse_cron_field(field: str, min_val: int, max_val: int) -> list[int] | None:
    """Parse a single cron field into a sorted list of integer values.

    Returns None if the field is '*' (meaning all values).
    """
    if field == "*":
        return None

    values: set[int] = set()
    for part in field.split(","):
        if "/" in part:
            range_part, step_str = part.split("/", 1)
            step = int(step_str)
            if range_part == "*":
                start, end = min_val, max_val
            elif "-" in range_part:
                start, end = (int(x) for x in range_part.split("-", 1))
            else:
                start = int(range_part)
                end = max_val
            values.update(range(start, end + 1, step))
        elif "-" in part:
            start, end = (int(x) for x in part.split("-", 1))
            values.update(range(start, end + 1))
        else:
            values.add(int(part))

    return sorted(v for v in values if min_val <= v <= max_val)


def _validate_cron_schedule(schedule: str) -> None:
    """Validate that a cron schedule can produce at least one valid trigger time.

    Raises ValueError for cron expressions that specify month/day combinations
    which never exist (e.g., ``0 23 31 2 *`` for February 31st).

    The Go cron library used by the LangGraph server silently returns a zero-value
    time (``0001-01-01T00:00:00``) for such schedules, which causes the
    scheduler's ``WHERE next_run_date <= NOW()`` condition to always be true,
    triggering the cron job in an infinite loop every few seconds.
    """
    parts = schedule.strip().split()
    if len(parts) != 5:
        return  # non-standard format, let the server validate

    _, _, day_field, month_field, _ = parts

    # Only validate when both day-of-month and month are constrained
    if day_field == "*" or month_field == "*":
        return

    try:
        days = _parse_cron_field(day_field, 1, 31)
        months = _parse_cron_field(month_field, 1, 12)
    except (ValueError, IndexError):
        return  # malformed, let the server validate

    if not days or not months:
        return

    # Check across a full leap-year cycle (4 years) to account for Feb 29
    for month in months:
        max_day_in_month = max(
            calendar.monthrange(year, month)[1] for year in range(2024, 2028)
        )
        for day in days:
            if day <= max_day_in_month:
                return  # at least one valid (month, day) combination exists

    raise ValueError(
        f"Cron schedule '{schedule}' specifies date(s) that never exist "
        f"(day {day_field} in month {month_field}). "
        f"The server's Go cron library would return a zero-value time for this "
        f"schedule, causing the scheduler to trigger continuously in an infinite "
        f"loop. See: https://github.com/langchain-ai/langgraph/issues/XXXX"
    )
