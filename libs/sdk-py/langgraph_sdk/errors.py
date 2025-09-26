from __future__ import annotations

import logging
import sys
from typing import Any, Literal, cast

import httpx
import orjson

logger = logging.getLogger(__name__)


class LangGraphError(Exception):
    pass


class APIError(httpx.HTTPStatusError, LangGraphError):
    message: str
    request: httpx.Request

    body: object | None
    code: str | None
    param: str | None
    type: str | None

    def __init__(
        self, message: str, response: httpx.Response, *, body: object | None
    ) -> None:
        httpx.HTTPStatusError.__init__(
            self, message, request=response.request, response=response
        )
        LangGraphError.__init__(self)

        self.request = response.request
        self.message = message
        self.body = body

        if isinstance(body, dict):
            b = cast(dict[str, Any], body)
            # Best-effort extraction of common fields if present
            code_val = b.get("code")
            self.code = code_val if isinstance(code_val, str) else None
            param_val = b.get("param")
            self.param = param_val if isinstance(param_val, str) else None
            t = b.get("type")
            self.type = t if isinstance(t, str) else None
        else:
            self.code = None
            self.param = None
            self.type = None


class APIResponseValidationError(APIError):
    response: httpx.Response
    status_code: int

    def __init__(
        self,
        response: httpx.Response,
        body: object | None,
        *,
        message: str | None = None,
    ) -> None:
        super().__init__(
            message or "Data returned by API invalid for expected schema.",
            response,
            body=body,
        )
        self.response = response
        self.status_code = response.status_code


class APIStatusError(APIError):
    response: httpx.Response
    status_code: int
    request_id: str | None

    def __init__(
        self, message: str, *, response: httpx.Response, body: object | None
    ) -> None:
        super().__init__(message, response, body=body)
        self.response = response
        self.status_code = response.status_code
        self.request_id = response.headers.get("x-request-id")


class APIConnectionError(APIError):
    def __init__(
        self, *, message: str = "Connection error.", request: httpx.Request
    ) -> None:
        super().__init__(message, request, body=None)


class APITimeoutError(APIConnectionError):
    def __init__(self, request: httpx.Request) -> None:
        super().__init__(message="Request timed out.", request=request)


class BadRequestError(APIStatusError):
    status_code: Literal[400] = 400


class AuthenticationError(APIStatusError):
    status_code: Literal[401] = 401


class PermissionDeniedError(APIStatusError):
    status_code: Literal[403] = 403


class NotFoundError(APIStatusError):
    status_code: Literal[404] = 404


class ConflictError(APIStatusError):
    status_code: Literal[409] = 409


class UnprocessableEntityError(APIStatusError):
    status_code: Literal[422] = 422


class RateLimitError(APIStatusError):
    status_code: Literal[429] = 429


class InternalServerError(APIStatusError):
    pass


def _extract_error_message(body: object | None, fallback: str) -> str:
    if isinstance(body, dict):
        b = cast(dict[str, Any], body)
        for key in ("message", "detail", "error"):
            val = b.get(key)
            if isinstance(val, str) and val:
                return val
        # Sometimes errors are structured like {"error": {"message": "..."}}
        err = b.get("error")
        if isinstance(err, dict):
            e = cast(dict[str, Any], err)
            for key in ("message", "detail"):
                val = e.get(key)
                if isinstance(val, str) and val:
                    return val
    return fallback


async def _adecode_error_body(r: httpx.Response) -> object | None:
    try:
        data = await r.aread()
    except Exception:
        return None
    if not data:
        return None
    try:
        return orjson.loads(data)
    except Exception:
        try:
            return data.decode()
        except Exception:
            return None


def _decode_error_body(r: httpx.Response) -> object | None:
    try:
        data = r.read()
    except Exception:
        return None
    if not data:
        return None
    try:
        return orjson.loads(data)
    except Exception:
        try:
            return data.decode()
        except Exception:
            return None


def _map_status_error(response: httpx.Response, body: object | None) -> APIStatusError:
    status = response.status_code
    reason = response.reason_phrase or "HTTP Error"
    message = _extract_error_message(body, f"{status} {reason}")
    if status == 400:
        return BadRequestError(message, response=response, body=body)
    if status == 401:
        return AuthenticationError(message, response=response, body=body)
    if status == 403:
        return PermissionDeniedError(message, response=response, body=body)
    if status == 404:
        return NotFoundError(message, response=response, body=body)
    if status == 409:
        return ConflictError(message, response=response, body=body)
    if status == 422:
        return UnprocessableEntityError(message, response=response, body=body)
    if status == 429:
        return RateLimitError(message, response=response, body=body)
    if 500 <= status:
        return InternalServerError(message, response=response, body=body)
    return APIStatusError(message, response=response, body=body)


async def _araise_for_status_typed(r: httpx.Response) -> None:
    if r.status_code < 400:
        return
    body = await _adecode_error_body(r)
    err = _map_status_error(r, body)
    # Log for older Python versions without Exception notes
    if not (sys.version_info >= (3, 11)):
        logger.error(f"Error from langgraph-api: {getattr(err, 'message', '')}")
    raise err


def _raise_for_status_typed(r: httpx.Response) -> None:
    if r.status_code < 400:
        return
    body = _decode_error_body(r)
    err = _map_status_error(r, body)
    if not (sys.version_info >= (3, 11)):
        logger.error(f"Error from langgraph-api: {getattr(err, 'message', '')}")
    raise err
