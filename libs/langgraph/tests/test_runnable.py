from __future__ import annotations

from typing import Any, Optional

import pytest
from langchain_core.runnables.config import RunnableConfig
from langgraph.store.base import BaseStore

from langgraph._internal._runnable import RunnableCallable
from langgraph.runtime import Runtime
from langgraph.types import StreamWriter

pytestmark = pytest.mark.anyio


def test_runnable_callable_func_accepts():
    def sync_func(x: Any) -> str:
        return f"{x}"

    async def async_func(x: Any) -> str:
        return f"{x}"

    def func_with_store(x: Any, store: BaseStore) -> str:
        return f"{x}"

    def func_with_writer(x: Any, writer: StreamWriter) -> str:
        return f"{x}"

    async def afunc_with_store(x: Any, store: BaseStore) -> str:
        return f"{x}"

    async def afunc_with_writer(x: Any, writer: StreamWriter) -> str:
        return f"{x}"

    runnables = {
        "sync": RunnableCallable(sync_func),
        "async": RunnableCallable(func=None, afunc=async_func),
        "with_store": RunnableCallable(func_with_store),
        "with_writer": RunnableCallable(func_with_writer),
        "awith_store": RunnableCallable(afunc_with_store),
        "awith_writer": RunnableCallable(afunc_with_writer),
    }

    expected_store = {"with_store": True, "awith_store": True}
    expected_writer = {"with_writer": True, "awith_writer": True}

    for name, runnable in runnables.items():
        if expected_writer.get(name, False):
            assert "writer" in runnable.func_accepts
        else:
            assert "writer" not in runnable.func_accepts

        if expected_store.get(name, False):
            assert "store" in runnable.func_accepts
        else:
            assert "store" not in runnable.func_accepts


async def test_runnable_callable_basic():
    def sync_func(x: Any) -> str:
        return f"{x}"

    async def async_func(x: Any) -> str:
        return f"{x}"

    runnable_sync = RunnableCallable(sync_func)
    runnable_async = RunnableCallable(func=None, afunc=async_func)

    result_sync = runnable_sync.invoke("test")
    assert result_sync == "test"

    # Test asynchronous ainvoke
    result_async = await runnable_async.ainvoke("test")
    assert result_async == "test"


def test_runnable_callable_injectable_arguments() -> None:
    """Test injectable arguments for RunnableCallable.

    This test verifies that injectable arguments like BaseStore work correctly.
    It tests:
    - Optional store injection
    - Required store injection
    - Store injection via config
    - Store injection override behavior
    - Store value injection and validation
    """

    # Test Optional[BaseStore] annotation.
    def func_optional_store(inputs: Any, store: Optional[BaseStore]) -> str:  # noqa: UP045
        """Test function that accepts an optional store parameter."""
        assert store is None
        return "success"

    assert (
        RunnableCallable(func_optional_store).invoke(
            {"x": "1"},
            config={
                "configurable": {
                    "__pregel_runtime": Runtime(
                        store=None,
                        context=None,
                        stream_writer=lambda _: None,
                        previous=None,
                    )
                }
            },
        )
        == "success"
    )

    # Test BaseStore annotation
    def func_required_store(inputs: Any, store: BaseStore) -> str:
        """Test function that requires a store parameter."""
        assert store is None
        return "success"

    with pytest.raises(ValueError):
        # Should fail b/c store is not Optional and config is not populated with store.
        assert RunnableCallable(func_required_store).invoke({}) == "success"

    # Manually provide store
    assert RunnableCallable(func_required_store).invoke({}, store=None) == "success"

    # Specify a value for store in the config
    assert (
        RunnableCallable(func_required_store).invoke(
            {},
            config={
                "configurable": {
                    "__pregel_runtime": Runtime(
                        store=None,
                        context=None,
                        stream_writer=lambda _: None,
                        previous=None,
                    )
                }
            },
        )
        == "success"
    )

    # Specify a value for store in config, but override with None
    assert (
        RunnableCallable(func_optional_store).invoke(
            {"x": "1"},
            store=None,
            config={
                "configurable": {
                    "__pregel_runtime": Runtime(
                        store="foobar",  # type: ignore[assignment]
                        context=None,
                        stream_writer=lambda _: None,
                        previous=None,
                    )
                }
            },
        )
        == "success"
    )

    # Set of tests where we verify that 'foobar' is injected as the store value.
    def func_required_store_v2(inputs: Any, store: BaseStore) -> str:
        """Test function that requires a store parameter and validates its value.

        The store value is expected to be 'foobar' when injected.
        """
        assert store == "foobar"
        return "success"

    assert (
        RunnableCallable(func_required_store_v2).invoke(
            {},
            config={
                "configurable": {
                    "__pregel_runtime": Runtime(
                        store="foobar",  # type: ignore[assignment]
                        context=None,
                        stream_writer=lambda _: None,
                        previous=None,
                    )
                }
            },
        )
        == "success"
    )

    assert RunnableCallable(func_required_store_v2).invoke(
        # And manual override takes precedence.
        {},
        store="foobar",
        config={
            "configurable": {
                "__pregel_runtime": Runtime(
                    store="foobar",  # type: ignore[assignment]
                    context=None,
                    stream_writer=lambda _: None,
                    previous=None,
                )
            }
        },
    )


async def test_runnable_callable_injectable_arguments_async() -> None:
    """Test injectable arguments for async RunnableCallable.

    This test verifies that injectable arguments like BaseStore work correctly
    in the async context. It tests:
    - Optional store injection
    - Required store injection
    - Store injection via config
    - Store injection override behavior
    """

    # Test Optional[BaseStore] annotation.
    def func_optional_store(inputs: Any, store: Optional[BaseStore]) -> str:  # noqa: UP045
        """Test function that accepts an optional store parameter."""
        assert store is None
        return "success"

    async def afunc_optional_store(inputs: Any, store: BaseStore | None) -> str:
        """Async version of func_optional_store."""
        assert store is None
        return "success"

    assert (
        await RunnableCallable(
            func=func_optional_store, afunc=afunc_optional_store
        ).ainvoke({"x": "1"})
        == "success"
    )

    # Test BaseStore annotation
    def func_required_store(inputs: Any, store: BaseStore) -> str:
        """Test function that requires a store parameter."""
        assert store is None
        return "success"

    async def afunc_required_store(inputs: Any, store: BaseStore) -> str:
        """Async version of func_required_store."""
        assert store is None
        return "success"

    with pytest.raises(ValueError):
        # Should fail b/c store is not Optional and config is not populated with store.
        assert (
            await RunnableCallable(
                func=func_required_store, afunc=afunc_required_store
            ).ainvoke(
                {},
            )
            == "success"
        )

    # Manually provide store
    assert (
        await RunnableCallable(
            func=func_required_store, afunc=afunc_required_store
        ).ainvoke(
            {},
            store=None,
            config={
                "configurable": {
                    "__pregel_runtime": Runtime(
                        store=None,
                        context=None,
                        stream_writer=lambda _: None,
                        previous=None,
                    )
                }
            },
        )
        == "success"
    )

    # Specify a value for store in the config
    assert (
        await RunnableCallable(
            func=func_required_store, afunc=afunc_required_store
        ).ainvoke(
            {},
            config={
                "configurable": {
                    "__pregel_runtime": Runtime(
                        store=None,
                        context=None,
                        stream_writer=lambda _: None,
                        previous=None,
                    )
                }
            },
        )
        == "success"
    )

    # Specify a value for store in config, but override with None
    assert (
        await RunnableCallable(
            func=func_optional_store, afunc=afunc_optional_store
        ).ainvoke(
            {"x": "1"},
            store=None,
            config={
                "configurable": {
                    "__pregel_runtime": Runtime(
                        store="foobar",
                        context=None,
                        stream_writer=lambda _: None,
                        previous=None,
                    )
                }
            },
        )
        == "success"
    )

    # Set of tests where we verify that 'foobar' is injected as the store value.
    def func_required_store_v2(inputs: Any, store: BaseStore) -> str:
        """Test function that requires a store parameter with specific value.

        The store parameter is expected to be 'foobar' when injected.
        """
        assert store == "foobar"
        return "success"

    async def afunc_required_store_v2(inputs: Any, store: BaseStore) -> str:
        """Async version of func_required_store_v2.

        The store parameter is expected to be 'foobar' when injected.
        """
        assert store == "foobar"
        return "success"

    assert (
        await RunnableCallable(
            func=func_required_store_v2, afunc=afunc_required_store_v2
        ).ainvoke(
            {},
            config={
                "configurable": {
                    "__pregel_runtime": Runtime(
                        store="foobar",
                        context=None,
                        stream_writer=lambda _: None,
                        previous=None,
                    )
                }
            },
        )
        == "success"
    )

    assert (
        await RunnableCallable(
            func=func_required_store_v2, afunc=afunc_required_store_v2
        ).ainvoke(
            # And manual override takes precedence.
            {},
            store="foobar",
            config={
                "configurable": {
                    "__pregel_runtime": Runtime(
                        store="foobar",
                        context=None,
                        stream_writer=lambda _: None,
                        previous=None,
                    )
                }
            },
        )
        == "success"
    )


def test_config_injection() -> None:
    def func(x: Any, config: RunnableConfig) -> list[str]:
        return config.get("tags", [])

    assert RunnableCallable(func).invoke(
        "test", config={"tags": ["test"], "configurable": {}}
    ) == ["test"]

    def func_optional(x: Any, config: Optional[RunnableConfig]) -> list[str]:  # noqa: UP045
        return config.get("tags", []) if config else []

    assert RunnableCallable(func_optional).invoke(
        "test", config={"tags": ["test"], "configurable": {}}
    ) == ["test"]

    def func_untyped(x: Any, config) -> list[str]:
        return config.get("tags", [])

    assert RunnableCallable(func_untyped).invoke(
        "test", config={"tags": ["test"], "configurable": {}}
    ) == ["test"]


def test_config_ensured() -> None:
    def func(input: str, config: RunnableConfig) -> None:
        assert input == "test"
        assert config is not None
        assert config.get("configurable") is not None

    RunnableCallable(func).invoke("test")


async def test_config_ensured_async() -> None:
    async def func(input: str, config: RunnableConfig) -> None:
        assert input == "test"
        assert config is not None
        assert config.get("configurable") is not None

    await RunnableCallable(func).ainvoke("test")
