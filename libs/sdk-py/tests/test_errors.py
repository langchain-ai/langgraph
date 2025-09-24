from __future__ import annotations

from typing import cast

import httpx
import orjson
import pytest

from langgraph_sdk.errors import (
    APIStatusError,
    AuthenticationError,
    BadRequestError,
    ConflictError,
    InternalServerError,
    NotFoundError,
    PermissionDeniedError,
    RateLimitError,
    UnprocessableEntityError,
    _raise_for_status_typed,
)


def make_response(
    status: int,
    *,
    json_body: dict | None = None,
    text_body: str | None = None,
    headers: dict[str, str] | None = None,
) -> httpx.Response:
    request = httpx.Request("GET", "https://example.com/test")
    content: bytes | None
    if json_body is not None:
        content = orjson.dumps(json_body)
    elif text_body is not None:
        content = text_body.encode()
    else:
        content = b""
    return httpx.Response(
        status, headers=headers or {}, content=content, request=request
    )


@pytest.mark.parametrize(
    "status,exc_type",
    [
        (400, BadRequestError),
        (401, AuthenticationError),
        (403, PermissionDeniedError),
        (404, NotFoundError),
        (409, ConflictError),
        (422, UnprocessableEntityError),
        (429, RateLimitError),
        (500, InternalServerError),
        (503, InternalServerError),  # any 5xx
        (418, APIStatusError),  # unmapped 4xx falls back to base type
    ],
)
def test_raise_for_status_typed_maps_exceptions_and_sets_status_code(
    status: int, exc_type: type[APIStatusError]
) -> None:
    r = make_response(
        status, json_body={"message": "boom", "code": "abc", "param": "p", "type": "t"}
    )

    with pytest.raises(exc_type) as ei:
        _raise_for_status_typed(r)

    err = cast(APIStatusError, ei.value)
    assert err.status_code == status
    # response attribute should be present and match
    assert err.response.status_code == status


def test_request_id_is_extracted_when_present() -> None:
    r = make_response(
        404, json_body={"detail": "missing"}, headers={"x-request-id": "req-123"}
    )
    with pytest.raises(NotFoundError) as ei:
        _raise_for_status_typed(r)
    err = cast(APIStatusError, ei.value)
    # request_id only exists on APIStatusError subclasses
    assert err.request_id == "req-123"


def test_non_json_body_does_not_break_mapping() -> None:
    r = make_response(429, text_body="Too many requests")
    with pytest.raises(RateLimitError) as ei:
        _raise_for_status_typed(r)
    err = cast(APIStatusError, ei.value)
    assert err.status_code == 429


def test_field_extraction_from_json_body() -> None:
    r = make_response(
        400,
        json_body={
            "message": "Invalid parameter",
            "code": "invalid_param",
            "param": "limit",
            "type": "invalid_request_error",
        },
    )
    with pytest.raises(BadRequestError) as ei:
        _raise_for_status_typed(r)
    err = cast(APIStatusError, ei.value)
    assert err.code == "invalid_param"
    assert err.param == "limit"
    assert err.type == "invalid_request_error"
