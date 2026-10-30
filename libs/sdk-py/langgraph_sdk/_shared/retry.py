"""Bounded retries for replayable, idempotent network requests."""

from __future__ import annotations

import asyncio
import random
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass

import httpx


@dataclass(frozen=True)
class RetryPolicy:
    """Retry a transient gateway response once, with roughly one second of jittered delay."""

    max_retries: int = 1
    methods: frozenset[str] = frozenset({"GET", "HEAD", "OPTIONS", "PUT", "DELETE"})
    status_codes: frozenset[int] = frozenset({502, 503, 504})

    def should_retry(
        self, request: httpx.Request, response: httpx.Response, retries: int
    ) -> bool:
        return (
            retries < self.max_retries
            and request.method in self.methods
            and response.status_code in self.status_codes
        )

    def delay(self) -> float:
        return random.uniform(0.5, 1.5)


_DEFAULT_RETRY_POLICY = RetryPolicy()


class RetryTransport(httpx.BaseTransport):
    """Apply the shared retry policy to a synchronous network transport."""

    def __init__(
        self,
        transport: httpx.BaseTransport,
        *,
        policy: RetryPolicy = _DEFAULT_RETRY_POLICY,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self._transport = transport
        self._policy = policy
        self._sleep = sleep

    def handle_request(self, request: httpx.Request) -> httpx.Response:
        replayable = isinstance(request.stream, httpx.ByteStream)
        retries = 0
        while True:
            response = self._transport.handle_request(request)
            if not replayable or not self._policy.should_retry(
                request, response, retries
            ):
                return response
            response.close()
            self._sleep(self._policy.delay())
            retries += 1

    def close(self) -> None:
        self._transport.close()


class AsyncRetryTransport(httpx.AsyncBaseTransport):
    """Apply the shared retry policy to an asynchronous network transport."""

    def __init__(
        self,
        transport: httpx.AsyncBaseTransport,
        *,
        policy: RetryPolicy = _DEFAULT_RETRY_POLICY,
        sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
    ) -> None:
        self._transport = transport
        self._policy = policy
        self._sleep = sleep

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        replayable = isinstance(request.stream, httpx.ByteStream)
        retries = 0
        while True:
            response = await self._transport.handle_async_request(request)
            if not replayable or not self._policy.should_retry(
                request, response, retries
            ):
                return response
            await response.aclose()
            await self._sleep(self._policy.delay())
            retries += 1

    async def aclose(self) -> None:
        await self._transport.aclose()
