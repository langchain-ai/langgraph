import asyncio
import time

import pytest


@pytest.fixture
def no_sleep(monkeypatch):
    """Record backoff sleeps without actually waiting."""
    sleeps: list[float] = []

    async def fake_async_sleep(d):
        sleeps.append(d)

    monkeypatch.setattr(asyncio, "sleep", fake_async_sleep)
    monkeypatch.setattr(time, "sleep", lambda d: sleeps.append(d))
    return sleeps
