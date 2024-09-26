from __future__ import annotations

from typing import Any

from langgraph.store.base import BaseStore
from langgraph.types import StreamWriter
from langgraph.utils.runnable import RunnableCallable


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
        assert runnable.func_accepts["writer"] == expected_writer.get(name, False)
        assert runnable.func_accepts["store"] == expected_store.get(name, False)
