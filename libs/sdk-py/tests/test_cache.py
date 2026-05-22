from datetime import timedelta
from unittest.mock import AsyncMock

import pytest

import langgraph_sdk.cache as cache_module


@pytest.mark.asyncio
async def test_swr_proxies_to_server_impl(monkeypatch):
    loader = AsyncMock(return_value={"ok": True})
    forwarded = {}

    async def fake_swr(key, inner_loader, *, fresh_for, max_age, model):
        forwarded["key"] = key
        forwarded["fresh_for"] = fresh_for
        forwarded["max_age"] = max_age
        forwarded["model"] = model
        return await inner_loader()

    monkeypatch.setattr(cache_module, "_api_swr", fake_swr)

    result = await cache_module.swr(
        "test-key",
        loader,
        fresh_for=timedelta(seconds=1),
        max_age=timedelta(seconds=10),
    )

    assert result == {"ok": True}
    assert forwarded == {
        "key": "test-key",
        "fresh_for": timedelta(seconds=1),
        "max_age": timedelta(seconds=10),
        "model": None,
    }
    loader.assert_awaited_once()


@pytest.mark.asyncio
async def test_swr_requires_server_runtime(monkeypatch):
    monkeypatch.setattr(cache_module, "_api_swr", None)

    with pytest.raises(RuntimeError, match="Cache is only available server-side"):
        await cache_module.swr(
            "test-key",
            AsyncMock(return_value="unused"),
            fresh_for=timedelta(seconds=1),
            max_age=timedelta(seconds=10),
        )


@pytest.mark.asyncio
async def test_swr_defaults(monkeypatch):
    """fresh_for defaults to 0, max_age defaults to 1 day."""
    loader = AsyncMock(return_value="val")
    forwarded = {}

    async def fake_swr(_key, inner_loader, *, fresh_for, max_age, model):  # noqa: ARG001
        forwarded["fresh_for"] = fresh_for
        forwarded["max_age"] = max_age
        return await inner_loader()

    monkeypatch.setattr(cache_module, "_api_swr", fake_swr)

    await cache_module.swr("k", loader)

    assert forwarded["fresh_for"] == timedelta(0)
    assert forwarded["max_age"] == timedelta(days=1)
