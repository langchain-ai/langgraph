"""Tests that the SDK HttpClient wraps httpx transport-level errors.

When a network-level error occurs (httpx.ReadError, ConnectError, timeouts,
protocol errors — all subclasses of httpx.TransportError), the SDK should raise
an APIConnectionError (or its APITimeoutError subclass) instead of letting the
raw httpx exception bubble up, so callers can catch all SDK errors uniformly via
`except LangGraphError`. The original httpx exception is preserved on
`__cause__` for callers that need the specifics.

These error classes already exist in langgraph_sdk.errors but were never wired
into the HttpClient request path.

Regression test for https://github.com/langchain-ai/langgraph/issues/5819
"""

from __future__ import annotations

from collections.abc import Callable

import httpx
import pytest

from langgraph_sdk.client import HttpClient, SyncHttpClient
from langgraph_sdk.errors import (
    APIConnectionError,
    APITimeoutError,
    LangGraphError,
    _map_transport_error,
)

# (id, factory, expected SDK exception type).
# - TimeoutException subclasses (ConnectTimeout/ReadTimeout) -> APITimeoutError
# - other TransportError (ReadError/ConnectError/RemoteProtocolError) -> APIConnectionError
TRANSPORT_ERROR_CASES: list[
    tuple[
        str, Callable[[httpx.Request], httpx.TransportError], type[APIConnectionError]
    ]
] = [
    (
        "read_error",
        lambda req: httpx.ReadError("read failed", request=req),
        APIConnectionError,
    ),
    (
        "connect_error",
        lambda req: httpx.ConnectError("connect failed", request=req),
        APIConnectionError,
    ),
    (
        "connect_timeout",
        lambda req: httpx.ConnectTimeout("connect timeout", request=req),
        APITimeoutError,
    ),
    (
        "read_timeout",
        lambda req: httpx.ReadTimeout("read timeout", request=req),
        APITimeoutError,
    ),
    (
        "remote_protocol_error",
        lambda req: httpx.RemoteProtocolError("remote protocol error", request=req),
        APIConnectionError,
    ),
]


def _transport_that_raises(
    factory: Callable[[httpx.Request], httpx.TransportError],
) -> httpx.MockTransport:
    def handler(request: httpx.Request) -> httpx.Response:
        raise factory(request)

    return httpx.MockTransport(handler)


def _read_error(request: httpx.Request) -> httpx.ReadError:
    return httpx.ReadError("simulated read failure", request=request)


# --------------------------------- async ---------------------------------


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("factory", "expected"),
    [(c[1], c[2]) for c in TRANSPORT_ERROR_CASES],
    ids=[c[0] for c in TRANSPORT_ERROR_CASES],
)
async def test_async_get_wraps_transport_errors(
    factory: Callable[[httpx.Request], httpx.TransportError],
    expected: type[APIConnectionError],
) -> None:
    transport = _transport_that_raises(factory)
    async with httpx.AsyncClient(
        transport=transport, base_url="https://example.com"
    ) as client:
        http_client = HttpClient(client)
        with pytest.raises(expected) as ei:
            await http_client.get("/foo")

    # Catchable uniformly via the LangGraphError hierarchy.
    assert isinstance(ei.value, LangGraphError)
    # APITimeoutError is a subclass of APIConnectionError.
    assert isinstance(ei.value, APIConnectionError)
    # Original httpx exception preserved for callers that need the specifics.
    assert isinstance(ei.value.__cause__, httpx.TransportError)


@pytest.mark.asyncio
async def test_async_post_wraps_transport_error() -> None:
    transport = _transport_that_raises(_read_error)
    async with httpx.AsyncClient(
        transport=transport, base_url="https://example.com"
    ) as client:
        http_client = HttpClient(client)
        with pytest.raises(APIConnectionError) as ei:
            await http_client.post("/foo", json={"a": 1})

    assert isinstance(ei.value.__cause__, httpx.TransportError)


# --------------------------------- sync ----------------------------------


@pytest.mark.parametrize(
    ("factory", "expected"),
    [(c[1], c[2]) for c in TRANSPORT_ERROR_CASES],
    ids=[c[0] for c in TRANSPORT_ERROR_CASES],
)
def test_sync_get_wraps_transport_errors(
    factory: Callable[[httpx.Request], httpx.TransportError],
    expected: type[APIConnectionError],
) -> None:
    transport = _transport_that_raises(factory)
    with httpx.Client(transport=transport, base_url="https://example.com") as client:
        http_client = SyncHttpClient(client)
        with pytest.raises(expected) as ei:
            http_client.get("/foo")

    assert isinstance(ei.value, LangGraphError)
    assert isinstance(ei.value, APIConnectionError)
    assert isinstance(ei.value.__cause__, httpx.TransportError)


def test_sync_post_wraps_transport_error() -> None:
    transport = _transport_that_raises(_read_error)
    with httpx.Client(transport=transport, base_url="https://example.com") as client:
        http_client = SyncHttpClient(client)
        with pytest.raises(APIConnectionError) as ei:
            http_client.post("/foo", json={"a": 1})

    assert isinstance(ei.value.__cause__, httpx.TransportError)


# --------------------- mapper unit + full method coverage --------------------


def test_map_transport_error_classifies_timeouts() -> None:
    # Timeouts map to APITimeoutError; other transport errors to APIConnectionError.
    request = httpx.Request("GET", "https://example.com/foo")
    err = _map_transport_error(httpx.ReadTimeout("read timeout", request=request))
    assert isinstance(err, APITimeoutError)
    assert isinstance(err, APIConnectionError)

    err2 = _map_transport_error(httpx.ReadError("read failed", request=request))
    assert isinstance(err2, APIConnectionError)
    assert not isinstance(err2, APITimeoutError)


def test_map_transport_error_without_request_does_not_raise() -> None:
    # A transport error whose .request was never attached must not crash the
    # mapper. httpx raises RuntimeError (not AttributeError) when accessing
    # .request on an exception constructed without one; the mapper must guard
    # so it never leaks while handling a transport error.
    exc = httpx.ReadError("no request attached")  # constructed without request=
    err = _map_transport_error(exc)  # must NOT raise RuntimeError
    assert isinstance(err, APIConnectionError)


def _body_kwargs(method: str) -> dict[str, object]:
    # get/delete take no json body; post/put/patch require one.
    if method in {"post", "put", "patch"}:
        return {"json": {"a": 1}}
    return {}


@pytest.mark.asyncio
@pytest.mark.parametrize("method", ["get", "post", "put", "patch", "delete"])
async def test_async_all_methods_wrap_transport_error(method: str) -> None:
    transport = _transport_that_raises(_read_error)
    async with httpx.AsyncClient(
        transport=transport, base_url="https://example.com"
    ) as client:
        http_client = HttpClient(client)
        with pytest.raises(APIConnectionError) as ei:
            await getattr(http_client, method)("/foo", **_body_kwargs(method))

    assert isinstance(ei.value.__cause__, httpx.TransportError)


@pytest.mark.parametrize("method", ["get", "post", "put", "patch", "delete"])
def test_sync_all_methods_wrap_transport_error(method: str) -> None:
    transport = _transport_that_raises(_read_error)
    with httpx.Client(transport=transport, base_url="https://example.com") as client:
        http_client = SyncHttpClient(client)
        with pytest.raises(APIConnectionError) as ei:
            getattr(http_client, method)("/foo", **_body_kwargs(method))

    assert isinstance(ei.value.__cause__, httpx.TransportError)
