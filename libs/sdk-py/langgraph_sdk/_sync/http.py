"""Synchronous HTTP client for LangGraph API."""

from __future__ import annotations

import logging
import os
import sys
import warnings
from collections.abc import Callable, Iterator, Mapping
from typing import Any, cast

import httpx
import orjson

from langgraph_sdk._shared.utilities import (
    _orjson_default,
    _validate_reconnect_location,
)
from langgraph_sdk.errors import _raise_for_status_typed
from langgraph_sdk.schema import QueryParamTypes, StreamPart
from langgraph_sdk.sse import SSEDecoder, iter_lines_raw

logger = logging.getLogger(__name__)

_ALLOWED_HOSTS: list[str] | None = None


def _get_allowed_hosts() -> list[str] | None:
    """Return the list of allowed hosts from environment, or None if not set."""
    global _ALLOWED_HOSTS
    env_val = os.environ.get("LANGGRAPH_ALLOWED_HOSTS", "")
    if env_val:
        return [h.strip() for h in env_val.split(",") if h.strip()]
    return None


def _validate_url_allowlist(client: httpx.Client, path: str) -> None:
    """Validate that the request path/URL is within the allowed hosts."""
    allowed_hosts = _get_allowed_hosts()
    if allowed_hosts is None:
        # No allowlist configured; fall back to base_url host only
        return
    # Resolve the full URL
    try:
        if path.startswith("http://") or path.startswith("https://"):
            url = httpx.URL(path)
        else:
            url = client.base_url.copy_with(path=path)
        host = url.host
    except Exception:
        raise ValueError(f"Invalid URL path: {path!r}")
    if host not in allowed_hosts:
        raise ValueError(
            f"Request to host {host!r} is not allowed. "
            f"Allowed hosts: {allowed_hosts}"
        )


def _validate_server_certificate(client: httpx.Client) -> None:
    """Validate that the underlying httpx client has SSL verification enabled."""
    # httpx.Client uses verify=True by default; warn if it has been disabled.
    transport = getattr(client, "_transport", None)
    if transport is not None:
        ssl_context = getattr(transport, "_ssl_context", None)
        if ssl_context is not None:
            verify = getattr(ssl_context, "verify_mode", None)
            import ssl
            if verify is not None and verify == ssl.CERT_NONE:
                raise ValueError(
                    "SSL certificate verification is disabled. "
                    "The MCP server cannot be authenticated."
                )
    # Also check the verify attribute on the client if present
    verify_attr = getattr(client, "_verify", None)
    if verify_attr is False:
        raise ValueError(
            "SSL certificate verification is disabled. "
            "The MCP server cannot be authenticated."
        )


def _sanitize_response(data: Any) -> Any:
    """Sanitize and validate response data from the server."""
    if data is None:
        return data
    if isinstance(data, (str, int, float, bool)):
        return data
    if isinstance(data, list):
        return [_sanitize_response(item) for item in data]
    if isinstance(data, dict):
        return {str(k): _sanitize_response(v) for k, v in data.items()}
    # For any other type, convert to string representation for safety
    return str(data)


def _require_hitl_approval(path: str, json: Any) -> None:
    """Human-in-the-Loop approval gate for DELETE operations."""
    print(
        f"\n[HITL] A DELETE operation has been requested.\n"
        f"  Path: {path}\n"
        f"  Payload: {json}\n"
        "Do you approve this DELETE operation? [yes/no]: ",
        end="",
        flush=True,
    )
    try:
        answer = input().strip().lower()
    except (EOFError, OSError):
        answer = ""
    if answer not in ("yes", "y"):
        raise PermissionError(
            f"DELETE operation to {path!r} was not approved by the user."
        )


class SyncHttpClient:
    """Handle synchronous requests to the LangGraph API.

    Provides error messaging and content handling enhancements above the
    underlying httpx client, mirroring the interface of [HttpClient](#HttpClient)
    but for sync usage.

    Attributes:
        client (httpx.Client): Underlying HTTPX sync client.
    """

    def __init__(self, client: httpx.Client) -> None:
        _validate_server_certificate(client)
        self.client = client

    def get(
        self,
        path: str,
        *,
        params: QueryParamTypes | None = None,
        headers: Mapping[str, str] | None = None,
        on_response: Callable[[httpx.Response], None] | None = None,
    ) -> Any:
        """Send a `GET` request."""
        _validate_url_allowlist(self.client, path)
        logger.debug("GET request: path=%s params=%s", path, params)
        r = self.client.get(path, params=params, headers=headers)
        logger.debug(
            "GET response: path=%s status=%s", path, r.status_code
        )
        if on_response:
            on_response(r)
        _raise_for_status_typed(r)
        result = _decode_json(r)
        return _sanitize_response(result)

    def post(
        self,
        path: str,
        *,
        json: dict[str, Any] | list | None,
        params: QueryParamTypes | None = None,
        headers: Mapping[str, str] | None = None,
        on_response: Callable[[httpx.Response], None] | None = None,
    ) -> Any:
        """Send a `POST` request."""
        _validate_url_allowlist(self.client, path)
        if json is not None:
            request_headers, content = _encode_json(json)
        else:
            request_headers, content = {}, b""
        if headers:
            request_headers.update(headers)
        logger.debug("POST request: path=%s params=%s", path, params)
        r = self.client.post(
            path, headers=request_headers, content=content, params=params
        )
        logger.debug(
            "POST response: path=%s status=%s", path, r.status_code
        )
        if on_response:
            on_response(r)
        _raise_for_status_typed(r)
        result = _decode_json(r)
        return _sanitize_response(result)

    def put(
        self,
        path: str,
        *,
        json: dict,
        params: QueryParamTypes | None = None,
        headers: Mapping[str, str] | None = None,
        on_response: Callable[[httpx.Response], None] | None = None,
    ) -> Any:
        """Send a `PUT` request."""
        _validate_url_allowlist(self.client, path)
        request_headers, content = _encode_json(json)
        if headers:
            request_headers.update(headers)
        logger.debug("PUT request: path=%s params=%s", path, params)
        r = self.client.put(
            path, headers=request_headers, content=content, params=params
        )
        logger.debug(
            "PUT response: path=%s status=%s", path, r.status_code
        )
        if on_response:
            on_response(r)
        _raise_for_status_typed(r)
        result = _decode_json(r)
        return _sanitize_response(result)

    def patch(
        self,
        path: str,
        *,
        json: dict,
        params: QueryParamTypes | None = None,
        headers: Mapping[str, str] | None = None,
        on_response: Callable[[httpx.Response], None] | None = None,
    ) -> Any:
        """Send a `PATCH` request."""
        _validate_url_allowlist(self.client, path)
        request_headers, content = _encode_json(json)
        if headers:
            request_headers.update(headers)
        logger.debug("PATCH request: path=%s params=%s", path, params)
        r = self.client.patch(
            path, headers=request_headers, content=content, params=params
        )
        logger.debug(
            "PATCH response: path=%s status=%s", path, r.status_code
        )
        if on_response:
            on_response(r)
        _raise_for_status_typed(r)
        result = _decode_json(r)
        return _sanitize_response(result)

    def delete(
        self,
        path: str,
        *,
        json: Any | None = None,
        params: QueryParamTypes | None = None,
        headers: Mapping[str, str] | None = None,
        on_response: Callable[[httpx.Response], None] | None = None,
    ) -> None:
        """Send a `DELETE` request."""
        _validate_url_allowlist(self.client, path)
        _require_hitl_approval(path, json)
        logger.debug("DELETE request: path=%s params=%s", path, params)
        r = self.client.request(
            "DELETE", path, json=json, params=params, headers=headers
        )
        logger.debug(
            "DELETE response: path=%s status=%s", path, r.status_code
        )
        if on_response:
            on_response(r)
        _raise_for_status_typed(r)

    def request_reconnect(
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
        _validate_url_allowlist(self.client, path)
        request_headers, content = _encode_json(json)
        if headers:
            request_headers.update(headers)
        logger.debug(
            "request_reconnect: method=%s path=%s params=%s", method, path, params
        )
        with self.client.stream(
            method, path, headers=request_headers, content=content, params=params
        ) as r:
            logger.debug(
                "request_reconnect response: method=%s path=%s status=%s",
                method,
                path,
                r.status_code,
            )
            if on_response:
                on_response(r)
            try:
                r.raise_for_status()
            except httpx.HTTPStatusError as e:
                body = r.read().decode()
                if sys.version_info >= (3, 11):
                    e.add_note(body)
                else:
                    logger.error(f"Error from langgraph-api: {body}", exc_info=e)
                raise e
            loc = r.headers.get("location")
            if reconnect_limit <= 0 or not loc:
                result = _decode_json(r)
                return _sanitize_response(result)
            _validate_reconnect_location(self.client.base_url, loc)
            try:
                result = _decode_json(r)
                return _sanitize_response(result)
            except httpx.HTTPError:
                warnings.warn(
                    f"Request failed, attempting reconnect to Location: {loc}",
                    stacklevel=2,
                )
                r.close()
                return self.request_reconnect(
                    loc,
                    "GET",
                    headers=request_headers,
                    # don't pass on_response so it's only called once
                    reconnect_limit=reconnect_limit - 1,
                )

    def stream(
        self,
        path: str,
        method: str,
        *,
        json: dict[str, Any] | None = None,
        params: QueryParamTypes | None = None,
        headers: Mapping[str, str] | None = None,
        on_response: Callable[[httpx.Response], None] | None = None,
    ) -> Iterator[StreamPart]:
        """Stream the results of a request using SSE."""
        _validate_url_allowlist(self.client, path)
        if json is not None:
            request_headers, content = _encode_json(json)
        else:
            request_headers, content = {}, None
        request_headers["Accept"] = "text/event-stream"
        request_headers["Cache-Control"] = "no-store"
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

        logger.debug("stream request: method=%s path=%s params=%s", method, path, params)

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
            with self.client.stream(
                current_method,
                reconnect_path or path,
                headers=current_headers,
                content=current_content,
                params=current_params,
            ) as res:
                logger.debug(
                    "stream response: method=%s path=%s status=%s",
                    current_method,
                    reconnect_path or path,
                    res.status_code,
                )
                if reconnect_path is None and on_response:
                    on_response(res)
                # check status
                _raise_for_status_typed(res)
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
                    reconnect_path = reconnect_location

                decoder = SSEDecoder()
                try:
                    for line in iter_lines_raw(res):
                        sse = decoder.decode(cast(bytes, line).rstrip(b"\n"))
                        if sse is not None:
                            if decoder.last_event_id is not None:
                                last_event_id = decoder.last_event_id
                            if sse.event or sse.data is not None:
                                sanitized_sse = _sanitize_sse_part(sse)
                                logger.debug(
                                    "stream SSE event: event=%s", sanitized_sse.event
                                )
                                yield sanitized_sse
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
                            # See async stream implementation for rationale on
                            # skipping empty flush events.
                            sanitized_sse = _sanitize_sse_part(sse)
                            logger.debug(
                                "stream SSE flush event: event=%s", sanitized_sse.event
                            )
                            yield sanitized_sse
            if retry:
                reconnect_attempts += 1
                if reconnect_attempts > max_reconnect_attempts:
                    raise httpx.TransportError(
                        "Exceeded maximum SSE reconnection attempts"
                    )
                continue
            break


def _sanitize_sse_part(sse: StreamPart) -> StreamPart:
    """Sanitize an SSE StreamPart received from the server."""
    sanitized_data = _sanitize_response(sse.data) if sse.data is not None else sse.data
    sanitized_event = str(sse.event) if sse.event is not None else sse.event
    return StreamPart(event=sanitized_event, data=sanitized_data)


def _encode_json(json: Any) -> tuple[dict[str, str], bytes]:
    body = orjson.dumps(
        json,
        _orjson_default,
        orjson.OPT_SERIALIZE_NUMPY | orjson.OPT_NON_STR_KEYS,
    )
    content_length = str(len(body))
    content_type = "application/json"
    headers = {"Content-Length": content_length, "Content-Type": content_type}
    return headers, body


def _decode_json(r: httpx.Response) -> Any:
    body = r.read()
    return orjson.loads(body) if body else None