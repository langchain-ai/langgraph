"""HTTP client for async operations."""

from __future__ import annotations

import asyncio
import base64
import hashlib
import logging
import re
import sys
import time
import warnings
from collections.abc import AsyncIterator, Callable, Mapping
from typing import Any, cast

import httpx
import orjson

from langgraph_sdk._shared.utilities import (
    _orjson_default,
    _validate_reconnect_location,
)
from langgraph_sdk.errors import _araise_for_status_typed
from langgraph_sdk.schema import QueryParamTypes, StreamPart
from langgraph_sdk.sse import SSEDecoder, aiter_lines_raw

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Security constants
# ---------------------------------------------------------------------------

# Maximum allowed response body size (10 MB) to enforce output data minimisation
_MAX_RESPONSE_BODY_BYTES = 10 * 1024 * 1024

# Maximum allowed request payload size (5 MB)
_MAX_REQUEST_BODY_BYTES = 5 * 1024 * 1024

# Patterns that indicate potentially malicious prompt injection or shell commands
_MALICIOUS_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"(?i)(ignore\s+(previous|prior|above)\s+instructions?)"),
    re.compile(r"(?i)(system\s*prompt|you\s+are\s+now|act\s+as\s+)"),
    re.compile(r"(?i)(rm\s+-rf|del\s+/[sqf]|format\s+c:|mkfs\b|dd\s+if=)"),
    re.compile(r"(?i)(exec\s*\(|eval\s*\(|__import__\s*\()"),
    re.compile(r"(?i)(base64\s*,\s*[A-Za-z0-9+/]{20,}={0,2})"),
    re.compile(r"(?i)(curl\s+|wget\s+|nc\s+-|ncat\s+|bash\s+-[ic])"),
    re.compile(r"(?i)(powershell\s+-|cmd\.exe|/bin/sh|/bin/bash)"),
]

# Allowed URL schemes for outbound requests
_ALLOWED_SCHEMES = {"http", "https"}


def _check_malicious_content(value: Any, path: str = "root") -> None:
    """Recursively scan a value for malicious content patterns."""
    if isinstance(value, str):
        for pattern in _MALICIOUS_PATTERNS:
            if pattern.search(value):
                raise ValueError(
                    f"Potentially malicious content detected in request payload at '{path}': "
                    f"matched pattern {pattern.pattern!r}"
                )
        # Check for suspicious base64-encoded content
        if len(value) > 40:
            try:
                decoded = base64.b64decode(value + "==", validate=False)
                decoded_str = decoded.decode("utf-8", errors="ignore")
                for pattern in _MALICIOUS_PATTERNS:
                    if pattern.search(decoded_str):
                        raise ValueError(
                            f"Potentially malicious base64-encoded content detected at '{path}'"
                        )
            except Exception as exc:
                if "malicious" in str(exc):
                    raise
    elif isinstance(value, dict):
        for k, v in value.items():
            _check_malicious_content(v, path=f"{path}.{k}")
    elif isinstance(value, list):
        for i, item in enumerate(value):
            _check_malicious_content(item, path=f"{path}[{i}]")


def _validate_input_payload(json: Any) -> None:
    """Validate and sanitize input payload before sending to the AI model endpoint."""
    if json is None:
        return
    # Check payload size
    try:
        serialized = orjson.dumps(json)
        if len(serialized) > _MAX_REQUEST_BODY_BYTES:
            raise ValueError(
                f"Request payload exceeds maximum allowed size of {_MAX_REQUEST_BODY_BYTES} bytes"
            )
    except (TypeError, ValueError) as exc:
        if "maximum allowed size" in str(exc):
            raise
        # orjson serialization error — let it propagate naturally later
    # Scan for malicious content
    _check_malicious_content(json)


def _validate_url_path(base_url: httpx.URL, path: str) -> None:
    """Validate that a URL path/location is within the allowed scheme set."""
    # If path looks like an absolute URL, validate its scheme
    if path.startswith("http://") or path.startswith("https://"):
        try:
            parsed = httpx.URL(path)
        except Exception:
            raise ValueError(f"Invalid URL: {path!r}")
        if parsed.scheme not in _ALLOWED_SCHEMES:
            raise ValueError(
                f"URL scheme {parsed.scheme!r} is not allowed. "
                f"Allowed schemes: {_ALLOWED_SCHEMES}"
            )
    # Relative paths are resolved against the base_url which is already trusted


def _sanitize_response(data: Any, path: str = "root") -> Any:
    """Sanitize and validate response data from the server."""
    if data is None:
        return data
    if isinstance(data, str):
        # Truncate excessively long string values
        if len(data) > 1_000_000:
            logger.warning(
                "Response field at '%s' truncated: value exceeded 1,000,000 characters",
                path,
            )
            data = data[:1_000_000]
        return data
    if isinstance(data, dict):
        return {k: _sanitize_response(v, path=f"{path}.{k}") for k, v in data.items()}
    if isinstance(data, list):
        return [_sanitize_response(item, path=f"{path}[{i}]") for i, item in enumerate(data)]
    return data


def _compute_hash(data: Any) -> str:
    """Compute a SHA-256 hash of the serialized data for audit purposes."""
    try:
        serialized = orjson.dumps(data, default=_orjson_default)
        return hashlib.sha256(serialized).hexdigest()
    except Exception:
        return "<unhashable>"


def _log_audit(
    action: str,
    method: str,
    path: str,
    status_code: int | None = None,
    input_hash: str | None = None,
    output_hash: str | None = None,
) -> None:
    """Emit a structured audit log entry for forensic readiness."""
    logger.info(
        "AUDIT action=%s method=%s path=%s status=%s input_hash=%s output_hash=%s timestamp=%s",
        action,
        method,
        path,
        status_code if status_code is not None else "N/A",
        input_hash or "N/A",
        output_hash or "N/A",
        time.time(),
    )


class HttpClient:
    """Handle async requests to the LangGraph API.

    Adds additional error messaging & content handling above the
    provided httpx client.

    Attributes:
        client (httpx.AsyncClient): Underlying HTTPX async client.
    """

    def __init__(self, client: httpx.AsyncClient) -> None:
        self.client = client

    async def get(
        self,
        path: str,
        *,
        params: QueryParamTypes | None = None,
        headers: Mapping[str, str] | None = None,
        on_response: Callable[[httpx.Response], None] | None = None,
    ) -> Any:
        """Send a `GET` request."""
        _validate_url_path(self.client.base_url, path)
        logger.debug("HTTP request: method=GET path=%s params=%s", path, params)
        _log_audit("request", "GET", path)
        r = await self.client.get(path, params=params, headers=headers)
        logger.debug(
            "HTTP response: method=GET path=%s status=%s", path, r.status_code
        )
        if on_response:
            on_response(r)
        await _araise_for_status_typed(r)
        result = await _adecode_json(r)
        result = _sanitize_response(result)
        _log_audit(
            "response",
            "GET",
            path,
            status_code=r.status_code,
            output_hash=_compute_hash(result),
        )
        return result

    async def post(
        self,
        path: str,
        *,
        json: dict[str, Any] | list | None,
        params: QueryParamTypes | None = None,
        headers: Mapping[str, str] | None = None,
        on_response: Callable[[httpx.Response], None] | None = None,
    ) -> Any:
        """Send a `POST` request."""
        _validate_url_path(self.client.base_url, path)
        _validate_input_payload(json)
        input_hash = _compute_hash(json)
        logger.debug(
            "HTTP request: method=POST path=%s params=%s input_hash=%s",
            path,
            params,
            input_hash,
        )
        _log_audit("request", "POST", path, input_hash=input_hash)
        if json is not None:
            request_headers, content = await _aencode_json(json)
        else:
            request_headers, content = {}, b""
        # Merge headers, with runtime headers taking precedence
        if headers:
            request_headers.update(headers)
        r = await self.client.post(
            path, headers=request_headers, content=content, params=params
        )
        logger.debug(
            "HTTP response: method=POST path=%s status=%s", path, r.status_code
        )
        if on_response:
            on_response(r)
        await _araise_for_status_typed(r)
        result = await _adecode_json(r)
        result = _sanitize_response(result)
        _log_audit(
            "response",
            "POST",
            path,
            status_code=r.status_code,
            input_hash=input_hash,
            output_hash=_compute_hash(result),
        )
        return result

    async def put(
        self,
        path: str,
        *,
        json: dict,
        params: QueryParamTypes | None = None,
        headers: Mapping[str, str] | None = None,
        on_response: Callable[[httpx.Response], None] | None = None,
    ) -> Any:
        """Send a `PUT` request."""
        _validate_url_path(self.client.base_url, path)
        _validate_input_payload(json)
        input_hash = _compute_hash(json)
        logger.debug(
            "HTTP request: method=PUT path=%s params=%s input_hash=%s",
            path,
            params,
            input_hash,
        )
        _log_audit("request", "PUT", path, input_hash=input_hash)
        request_headers, content = await _aencode_json(json)
        if headers:
            request_headers.update(headers)
        r = await self.client.put(
            path, headers=request_headers, content=content, params=params
        )
        logger.debug(
            "HTTP response: method=PUT path=%s status=%s", path, r.status_code
        )
        if on_response:
            on_response(r)
        await _araise_for_status_typed(r)
        result = await _adecode_json(r)
        result = _sanitize_response(result)
        _log_audit(
            "response",
            "PUT",
            path,
            status_code=r.status_code,
            input_hash=input_hash,
            output_hash=_compute_hash(result),
        )
        return result

    async def patch(
        self,
        path: str,
        *,
        json: dict,
        params: QueryParamTypes | None = None,
        headers: Mapping[str, str] | None = None,
        on_response: Callable[[httpx.Response], None] | None = None,
    ) -> Any:
        """Send a `PATCH` request."""
        _validate_url_path(self.client.base_url, path)
        _validate_input_payload(json)
        input_hash = _compute_hash(json)
        logger.debug(
            "HTTP request: method=PATCH path=%s params=%s input_hash=%s",
            path,
            params,
            input_hash,
        )
        _log_audit("request", "PATCH", path, input_hash=input_hash)
        request_headers, content = await _aencode_json(json)
        if headers:
            request_headers.update(headers)
        r = await self.client.patch(
            path, headers=request_headers, content=content, params=params
        )
        logger.debug(
            "HTTP response: method=PATCH path=%s status=%s", path, r.status_code
        )
        if on_response:
            on_response(r)
        await _araise_for_status_typed(r)
        result = await _adecode_json(r)
        result = _sanitize_response(result)
        _log_audit(
            "response",
            "PATCH",
            path,
            status_code=r.status_code,
            input_hash=input_hash,
            output_hash=_compute_hash(result),
        )
        return result

    async def delete(
        self,
        path: str,
        *,
        json: Any | None = None,
        params: QueryParamTypes | None = None,
        headers: Mapping[str, str] | None = None,
        on_response: Callable[[httpx.Response], None] | None = None,
        _hitl_confirmed: bool = False,
    ) -> None:
        """Send a `DELETE` request.

        Requires Human-in-the-Loop (HITL) approval before executing.
        Pass ``_hitl_confirmed=True`` only after obtaining explicit human approval.
        """
        _validate_url_path(self.client.base_url, path)
        if not _hitl_confirmed:
            raise PermissionError(
                f"DELETE request to '{path}' requires Human-in-the-Loop (HITL) approval. "
                "Obtain explicit human confirmation and pass _hitl_confirmed=True to proceed."
            )
        logger.warning(
            "HITL-approved DELETE request: path=%s params=%s", path, params
        )
        _log_audit("request", "DELETE", path)
        r = await self.client.request(
            "DELETE", path, json=json, params=params, headers=headers
        )
        logger.debug(
            "HTTP response: method=DELETE path=%s status=%s", path, r.status_code
        )
        if on_response:
            on_response(r)
        _log_audit("response", "DELETE", path, status_code=r.status_code)
        await _araise_for_status_typed(r)

    async def request_reconnect(
        self,
        path: str,
        method: str,
        *,
        json: dict[str, Any] | None = None,
        params: QueryParamTypes | None = None,
        headers: Mapping[str, str] | None = None,
        on_response: Callable[[httpx.Response], None] | None = None,
        reconnect_limit: int = 5,
    ) -> Any:
        """Send a request that automatically reconnects to Location header."""
        _validate_url_path(self.client.base_url, path)
        _validate_input_payload(json)
        input_hash = _compute_hash(json)
        logger.debug(
            "HTTP request: method=%s path=%s params=%s input_hash=%s",
            method,
            path,
            params,
            input_hash,
        )
        _log_audit("request", method, path, input_hash=input_hash)
        request_headers, content = await _aencode_json(json)
        if headers:
            request_headers.update(headers)
        async with self.client.stream(
            method, path, headers=request_headers, content=content, params=params
        ) as r:
            if on_response:
                on_response(r)
            try:
                r.raise_for_status()
            except httpx.HTTPStatusError as e:
                body = (await r.aread()).decode()
                if sys.version_info >= (3, 11):
                    e.add_note(body)
                else:
                    logger.error(f"Error from langgraph-api: {body}", exc_info=e)
                raise e
            loc = r.headers.get("location")
            if reconnect_limit <= 0 or not loc:
                result = await _adecode_json(r)
                result = _sanitize_response(result)
                _log_audit(
                    "response",
                    method,
                    path,
                    status_code=r.status_code,
                    input_hash=input_hash,
                    output_hash=_compute_hash(result),
                )
                return result
            _validate_reconnect_location(self.client.base_url, loc)
            _validate_url_path(self.client.base_url, loc)
            try:
                result = await _adecode_json(r)
                result = _sanitize_response(result)
                _log_audit(
                    "response",
                    method,
                    path,
                    status_code=r.status_code,
                    input_hash=input_hash,
                    output_hash=_compute_hash(result),
                )
                return result
            except httpx.HTTPError:
                warnings.warn(
                    f"Request failed, attempting reconnect to Location: {loc}",
                    stacklevel=2,
                )
                await r.aclose()
                return await self.request_reconnect(
                    loc,
                    "GET",
                    headers=request_headers,
                    # don't pass on_response so it's only called once
                    reconnect_limit=reconnect_limit - 1,
                )

    async def stream(
        self,
        path: str,
        method: str,
        *,
        json: dict[str, Any] | None = None,
        params: QueryParamTypes | None = None,
        headers: Mapping[str, str] | None = None,
        on_response: Callable[[httpx.Response], None] | None = None,
    ) -> AsyncIterator[StreamPart]:
        """Stream results using SSE."""
        _validate_url_path(self.client.base_url, path)
        _validate_input_payload(json)
        input_hash = _compute_hash(json)
        logger.debug(
            "HTTP stream request: method=%s path=%s params=%s input_hash=%s",
            method,
            path,
            params,
            input_hash,
        )
        _log_audit("request", method, path, input_hash=input_hash)
        request_headers, content = await _aencode_json(json)
        request_headers["Accept"] = "text/event-stream"
        request_headers["Cache-Control"] = "no-store"
        # Add runtime headers with precedence
        if headers:
            request_headers.update(headers)

        reconnect_headers = {
            key: value
            for key, value in request_headers.items()
            if key.lower() not in {"content-length", "content-type"}
        }

        last_event_id: str | None = None
        reconnect_path: str | None = None
        reconnect_attempts = 0
        max_reconnect_attempts = 5
        event_count = 0

        while True:
            current_headers = dict(
                request_headers if reconnect_path is None else reconnect_headers
            )
            if last_event_id is not None:
                current_headers["Last-Event-ID"] = last_event_id

            current_method = method if reconnect_path is None else "GET"
            current_content = content if reconnect_path is None else None
            current_params = params if reconnect_path is None else None

            retry = False
            async with self.client.stream(
                current_method,
                reconnect_path or path,
                headers=current_headers,
                content=current_content,
                params=current_params,
            ) as res:
                if reconnect_path is None and on_response:
                    on_response(res)
                logger.debug(
                    "HTTP stream response: method=%s path=%s status=%s",
                    current_method,
                    reconnect_path or path,
                    res.status_code,
                )
                _log_audit(
                    "stream_response",
                    current_method,
                    reconnect_path or path,
                    status_code=res.status_code,
                    input_hash=input_hash,
                )
                # check status
                await _araise_for_status_typed(res)
                # check content type
                content_type = res.headers.get("content-type", "").partition(";")[0]
                if "text/event-stream" not in content_type:
                    raise httpx.TransportError(
                        "Expected response header Content-Type to contain 'text/event-stream', "
                        f"got {content_type!r}"
                    )

                reconnect_location = res.headers.get("location")
                if reconnect_location:
                    _validate_reconnect_location(
                        self.client.base_url, reconnect_location
                    )
                    _validate_url_path(self.client.base_url, reconnect_location)
                    reconnect_path = reconnect_location

                # parse SSE
                decoder = SSEDecoder()
                try:
                    async for line in aiter_lines_raw(res):
                        sse = decoder.decode(line=cast("bytes", line).rstrip(b"\n"))
                        if sse is not None:
                            if decoder.last_event_id is not None:
                                last_event_id = decoder.last_event_id
                            if sse.event or sse.data is not None:
                                event_count += 1
                                logger.debug(
                                    "SSE event received: event=%s event_count=%d",
                                    sse.event,
                                    event_count,
                                )
                                yield sse
                except httpx.HTTPError:
                    # httpx.TransportError inherits from HTTPError, so transient
                    # disconnects during streaming land here.
                    if reconnect_path is None:
                        raise
                    retry = True
                else:
                    if sse := decoder.decode(b""):
                        if decoder.last_event_id is not None:
                            last_event_id = decoder.last_event_id
                        if sse.event or sse.data is not None:
                            # decoder.decode(b"") flushes the in-flight event and may
                            # return an empty placeholder when there is no pending
                            # message. Skip these no-op events so the stream doesn't
                            # emit a trailing blank item after reconnects.
                            event_count += 1
                            logger.debug(
                                "SSE flush event received: event=%s event_count=%d",
                                sse.event,
                                event_count,
                            )
                            yield sse
            if retry:
                reconnect_attempts += 1
                if reconnect_attempts > max_reconnect_attempts:
                    raise httpx.TransportError(
                        "Exceeded maximum SSE reconnection attempts"
                    )
                continue
            break

        logger.debug(
            "HTTP stream completed: method=%s path=%s total_events=%d",
            method,
            path,
            event_count,
        )
        _log_audit("stream_complete", method, path, input_hash=input_hash)


async def _aencode_json(json: Any) -> tuple[dict[str, str], bytes | None]:
    if json is None:
        return {}, None
    body = await asyncio.get_running_loop().run_in_executor(
        None,
        orjson.dumps,
        json,
        _orjson_default,
        orjson.OPT_SERIALIZE_NUMPY | orjson.OPT_NON_STR_KEYS,
    )
    content_length = str(len(body))
    content_type = "application/json"
    headers = {"Content-Length": content_length, "Content-Type": content_type}
    return headers, body


async def _adecode_json(r: httpx.Response) -> Any:
    body = await r.aread()
    if not body:
        return None
    if len(body) > _MAX_RESPONSE_BODY_BYTES:
        raise ValueError(
            f"Response body size {len(body)} bytes exceeds maximum allowed "
            f"{_MAX_RESPONSE_BODY_BYTES} bytes"
        )
    return await asyncio.get_running_loop().run_in_executor(None, orjson.loads, body)