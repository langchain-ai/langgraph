"""Resend idempotent requests after a transient gateway failure."""

from __future__ import annotations

import asyncio
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass

import httpx

from langgraph_sdk._shared.utilities import _reconnect_delay


@dataclass(frozen=True)
class RetryPolicy:
    """Which failed requests to resend, how many times, and how long to wait first."""

    max_retries: int = 1
    methods: frozenset[str] = frozenset({"GET", "HEAD", "OPTIONS", "PUT", "DELETE"})
    status_codes: frozenset[int] = frozenset({502, 503, 504})

    def can_resend(self, request: httpx.Request) -> bool:
        """Checked before sending, because the first attempt consumes a streamed body."""
        return request.method in self.methods and isinstance(
            request.stream, httpx.ByteStream
        )

    def should_retry(self, response: httpx.Response, retries: int) -> bool:
        return retries < self.max_retries and response.status_code in self.status_codes

    def delay(self, retries: int) -> float:
        return _reconnect_delay(retries, base=1.0, cap=5.0)


_POLICY = RetryPolicy()


class RetryTransport(httpx.BaseTransport):
    """Resend requests through a synchronous transport as the retry policy allows."""

    def __init__(
        self,
        transport: httpx.BaseTransport,
        *,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self._transport = transport
        self._sleep = sleep

    def handle_request(self, request: httpx.Request) -> httpx.Response:
        resendable = _POLICY.can_resend(request)
        retries = 0
        while True:
            response = self._transport.handle_request(request)
            if not (resendable and _POLICY.should_retry(response, retries)):
                return response
            response.close()
            self._sleep(_POLICY.delay(retries))
            retries += 1

    def close(self) -> None:
        self._transport.close()


class AsyncRetryTransport(httpx.AsyncBaseTransport):
    """Resend requests through an asynchronous transport as the retry policy allows."""

    def __init__(
        self,
        transport: httpx.AsyncBaseTransport,
        *,
        sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
    ) -> None:
        self._transport = transport
        self._sleep = sleep

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        resendable = _POLICY.can_resend(request)
        retries = 0
        while True:
            response = await self._transport.handle_async_request(request)
            if not (resendable and _POLICY.should_retry(response, retries)):
                return response
            await response.aclose()
            await self._sleep(_POLICY.delay(retries))
            retries += 1

    async def aclose(self) -> None:
        await self._transport.aclose()
