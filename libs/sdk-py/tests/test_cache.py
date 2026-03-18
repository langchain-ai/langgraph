from datetime import timedelta
from unittest.mock import AsyncMock

import pytest

import langgraph_sdk.cache as cache_module
from langgraph_sdk.cache import swr_cached


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


# -- swr_cached decorator tests --


@pytest.mark.asyncio
async def test_swr_cached_no_parens(monkeypatch):
    """@swr_cached without parentheses uses qualname as key."""
    forwarded = {}

    async def fake_swr(key, loader, *, fresh_for, max_age, model):  # noqa: ARG001
        forwarded["key"] = key
        forwarded["model"] = model
        return await loader()

    monkeypatch.setattr(cache_module, "_api_swr", fake_swr)

    @swr_cached
    async def fetch_config():
        return {"debug": True}

    result = await fetch_config()

    assert result == {"debug": True}
    assert forwarded["key"] == "test_swr_cached_no_parens.<locals>.fetch_config"
    assert forwarded["model"] is None


@pytest.mark.asyncio
async def test_swr_cached_with_options(monkeypatch):
    """@swr_cached(...) with keyword options."""
    forwarded = {}

    async def fake_swr(key, loader, *, fresh_for, max_age, model):  # noqa: ARG001
        forwarded["key"] = key
        forwarded["fresh_for"] = fresh_for
        forwarded["max_age"] = max_age
        return await loader()

    monkeypatch.setattr(cache_module, "_api_swr", fake_swr)

    @swr_cached(fresh_for=timedelta(minutes=5), max_age=timedelta(hours=1))
    async def fetch_config():
        return "ok"

    await fetch_config()

    assert forwarded["fresh_for"] == timedelta(minutes=5)
    assert forwarded["max_age"] == timedelta(hours=1)


@pytest.mark.asyncio
async def test_swr_cached_key_includes_args(monkeypatch):
    """Arguments are appended to the auto-derived cache key."""
    forwarded = {}

    async def fake_swr(key, loader, *, fresh_for, max_age, model):  # noqa: ARG001
        forwarded["key"] = key
        return await loader()

    monkeypatch.setattr(cache_module, "_api_swr", fake_swr)

    @swr_cached
    async def fetch_profile(user_id: str):
        return {"id": user_id}

    await fetch_profile("abc123")

    expected_prefix = "test_swr_cached_key_includes_args.<locals>.fetch_profile"
    assert forwarded["key"] == f"{expected_prefix}:abc123"


@pytest.mark.asyncio
async def test_swr_cached_explicit_key_string(monkeypatch):
    """Explicit string key overrides auto-derivation."""
    forwarded = {}

    async def fake_swr(key, loader, *, fresh_for, max_age, model):  # noqa: ARG001
        forwarded["key"] = key
        return await loader()

    monkeypatch.setattr(cache_module, "_api_swr", fake_swr)

    @swr_cached(key="my-static-key")
    async def fetch_stuff():
        return 42

    await fetch_stuff()
    assert forwarded["key"] == "my-static-key"


@pytest.mark.asyncio
async def test_swr_cached_explicit_key_callable(monkeypatch):
    """Explicit callable key receives the function's arguments."""
    forwarded = {}

    async def fake_swr(key, loader, *, fresh_for, max_age, model):  # noqa: ARG001
        forwarded["key"] = key
        return await loader()

    monkeypatch.setattr(cache_module, "_api_swr", fake_swr)

    @swr_cached(key=lambda org, repo: f"repo:{org}/{repo}")
    async def fetch_repo(org: str, repo: str):
        return {"full_name": f"{org}/{repo}"}

    await fetch_repo("langchain-ai", "langgraph")
    assert forwarded["key"] == "repo:langchain-ai/langgraph"


@pytest.mark.asyncio
async def test_swr_cached_preserves_function_metadata(monkeypatch):
    """functools.wraps preserves __name__ and __doc__."""

    async def fake_swr(_key, loader, **_kw):
        return await loader()

    monkeypatch.setattr(cache_module, "_api_swr", fake_swr)

    @swr_cached
    async def my_loader():
        """My docstring."""
        return 1

    assert my_loader.__name__ == "my_loader"
    assert my_loader.__doc__ == "My docstring."


@pytest.mark.asyncio
async def test_swr_cached_infers_pydantic_model(monkeypatch):
    """Model is auto-detected from return type annotation."""
    pytest.importorskip("pydantic")
    from pydantic import BaseModel

    forwarded = {}

    async def fake_swr(key, loader, *, fresh_for, max_age, model):  # noqa: ARG001
        forwarded["model"] = model
        return await loader()

    monkeypatch.setattr(cache_module, "_api_swr", fake_swr)

    class Profile(BaseModel):
        name: str

    @swr_cached
    async def fetch_profile() -> Profile:
        return Profile(name="Alice")

    await fetch_profile()
    assert forwarded["model"] is Profile
