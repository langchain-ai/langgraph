"""HTTP client for async operations."""

from __future__ import annotations

import asyncio
import logging
import sys
import warnings
from collections.abc import AsyncIterator, Callable, Mapping
from typing import Any, cast

import httpx
import orjson

from langgraph_sdk._shared.utilities import _orjson_default
from langgraph_sdk.errors import _araise_for_status_typed
from langgraph_sdk.schema import QueryParamTypes, StreamPart
from langgraph_sdk.sse import SSEDecoder, aiter_lines_raw

logger = logging.getLogger(__name__)


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
        r = await self.client.get(path, params=params, headers=headers)
        if on_response:
            on_response(r)
        await _araise_for_status_typed(r)
        return await _adecode_json(r)

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
        if on_response:
            on_response(r)
        await _araise_for_status_typed(r)
        return await _adecode_json(r)

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
        request_headers, content = await _aencode_json(json)
        if headers:
            request_headers.update(headers)
        r = await self.client.put(
            path, headers=request_headers, content=content, params=params
        )
        if on_response:
            on_response(r)
        await _araise_for_status_typed(r)
        return await _adecode_json(r)

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
        request_headers, content = await _aencode_json(json)
        if headers:
            request_headers.update(headers)
        r = await self.client.patch(
            path, headers=request_headers, content=content, params=params
        )
        if on_response:
            on_response(r)
        await _araise_for_status_typed(r)
        return await _adecode_json(r)

    async def delete(
        self,
        path: str,
        *,
        json: Any | None = None,
        params: QueryParamTypes | None = None,
        headers: Mapping[str, str] | None = None,
        on_response: Callable[[httpx.Response], None] | None = None,
    ) -> None:
        """Send a `DELETE` request."""
        r = await self.client.request(
            "DELETE", path, json=json, params=params, headers=headers
        )
        if on_response:
            on_response(r)
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
                return await _adecode_json(r)
            try:
                return await _adecode_json(r)
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
                            yield sse
            if retry:
                reconnect_attempts += 1
                if reconnect_attempts > max_reconnect_attempts:
                    raise httpx.TransportError(
                        "Exceeded maximum SSE reconnection attempts"
                    )
                continue
            break


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
    return (
        await asyncio.get_running_loop().run_in_executor(None, orjson.loads, body)
        if body
        else None
    )
