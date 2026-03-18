import inspect
from datetime import timedelta
from unittest.mock import AsyncMock

import orjson
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
    """`@swr_cached` without parentheses uses a structured auto-derived key."""
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
    cache_key = orjson.loads(forwarded["key"])

    assert result == {"debug": True}
    assert cache_key == {
        "args": [],
        "module": fetch_config.__module__,
        "qualname": fetch_config.__qualname__,
    }
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
    """Arguments are serialized structurally into the auto-derived cache key."""
    forwarded = {}

    async def fake_swr(key, loader, *, fresh_for, max_age, model):  # noqa: ARG001
        forwarded["key"] = key
        return await loader()

    monkeypatch.setattr(cache_module, "_api_swr", fake_swr)

    @swr_cached
    async def fetch_profile(user_id: str):
        return {"id": user_id}

    await fetch_profile("abc123")
    cache_key = orjson.loads(forwarded["key"])
    assert cache_key["args"] == [{"name": "user_id", "value": "abc123"}]


def test_build_cache_key_distinguishes_modules():
    async def fetch_profile(user_id: str): ...

    sig = inspect.signature(fetch_profile)
    key_one = cache_module._build_cache_key(
        "alpha.module", "fetch_profile", sig, ("1",), {}
    )
    key_two = cache_module._build_cache_key(
        "beta.module", "fetch_profile", sig, ("1",), {}
    )

    assert key_one != key_two


def test_build_cache_key_distinguishes_argument_boundaries():
    async def fetch_profile(left: str, right: str): ...

    sig = inspect.signature(fetch_profile)
    key_one = cache_module._build_cache_key(
        "alpha.module",
        "fetch_profile",
        sig,
        ("a:b", "c"),
        {},
    )
    key_two = cache_module._build_cache_key(
        "alpha.module",
        "fetch_profile",
        sig,
        ("a", "b:c"),
        {},
    )

    assert key_one != key_two


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
async def test_swr_cached_method_key_uses_class_identity(monkeypatch):
    forwarded = []

    async def fake_swr(key, loader, *, fresh_for, max_age, model):  # noqa: ARG001
        forwarded.append(key)
        return await loader()

    monkeypatch.setattr(cache_module, "_api_swr", fake_swr)

    class Client:
        @swr_cached
        async def fetch_repo(self, repo: str):
            return {"repo": repo}

    await Client().fetch_repo("langgraph")
    await Client().fetch_repo("langgraph")

    assert forwarded[0] == forwarded[1]
    cache_key = orjson.loads(forwarded[0])
    assert cache_key["args"][0] == {
        "name": "self",
        "value": {
            "class": f"{Client.__module__}.{Client.__qualname__}",
            "kind": "self",
        },
    }


@pytest.mark.asyncio
async def test_swr_cached_rejects_unstable_object_args():
    class Unstable:
        pass

    @swr_cached
    async def fetch_profile(user: Unstable):
        return {"user_type": type(user).__name__}

    with pytest.raises(
        TypeError,
        match="Cannot auto-generate a stable cache key for `user`",
    ):
        await fetch_profile(Unstable())


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
