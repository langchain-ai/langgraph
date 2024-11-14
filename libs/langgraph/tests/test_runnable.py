from __future__ import annotations

from typing import Any

import pytest

from langgraph.store.base import BaseStore
from langgraph.types import StreamWriter
from langgraph.utils.runnable import RunnableCallable, RunnableSeq

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
        assert runnable.func_accepts["writer"] == expected_writer.get(name, False)
        assert runnable.func_accepts["store"] == expected_store.get(name, False)


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


def test_runnable_seq_operator_overload():
    def step1(x: Any) -> str:
        return f"{x}-1"

    def step2(x: Any) -> str:
        return f"{x}-2"

    def step3(x: Any) -> str:
        return f"{x}-3"

    seq1 = RunnableSeq(step1, step2)
    seq2 = RunnableSeq(step2, step3)

    combined = seq1 | step3
    result = combined.invoke("test")
    assert result == "test-1-2-3"

    combined2 = seq1 | seq2
    result2 = combined2.invoke("test")
    assert result2 == "test-1-2-2-3"


def test_runnable_seq_stream_error():
    def step1(x: Any) -> str:
        return f"{x}-1"

    def step2(x: Any) -> str:
        raise ValueError("Test error")

    seq = RunnableSeq(step1, step2)
    with pytest.raises(ValueError, match="Test error"):
        list(seq.stream("test"))


async def test_runnable_seq_astream():
    async def astep1(x: Any) -> str:
        return f"{x}-1"

    async def astep2(x: Any) -> str:
        return f"{x}-2"

    seq = RunnableSeq(astep1, astep2)
    chunks = []
    async for chunk in seq.astream("test"):
        chunks.append(chunk)
    assert chunks == ["test-1-2"]


def test_runnable_seq_validation():
    def step1(x: Any) -> str:
        return f"{x}"

    with pytest.raises(ValueError, match="RunnableSeq must have at least 2 steps"):
        RunnableSeq(step1)


def test_runnable_seq_stream():
    def step1(x: Any) -> str:
        return f"{x}-1"

    def step2(x: Any) -> str:
        return f"{x}-2"

    seq = RunnableSeq(step1, step2)
    chunks = list(seq.stream("test"))
    assert chunks == ["test-1-2"]


def test_runnable_seq_basic():
    def step1(x: Any) -> str:
        return f"{x}-1"

    def step2(x: Any) -> str:
        return f"{x}-2"

    seq = RunnableSeq(step1, step2)
    result = seq.invoke("test")
    assert result == "test-1-2"


def test_runnable_callable_recursive():
    def outer_func(x: Any) -> RunnableCallable:
        def inner_func(y: Any) -> str:
            return f"{y}-inner"

        return RunnableCallable(inner_func)

    runnable = RunnableCallable(outer_func, recurse=True)
    result = runnable.invoke("test")
    assert result == "test-inner"


def test_runnable_callable_no_funcs():
    with pytest.raises(
        ValueError, match="At least one of func or afunc must be provided."
    ):
        RunnableCallable(func=None, afunc=None)


async def test_runnable_callable_no_trace():
    def sync_func(x: Any) -> str:
        return f"{x}"

    runnable = RunnableCallable(sync_func, trace=False)
    result = runnable.invoke("test")
    assert result == "test"


async def test_runnable_seq_ainvoke_no_config():
    async def astep1(x: Any) -> str:
        return f"{x}-1"

    async def astep2(x: Any) -> str:
        return f"{x}-2"

    seq = RunnableSeq(astep1, astep2)
    result = await seq.ainvoke("test")
    assert result == "test-1-2"


async def test_runnable_seq_ainvoke_error():
    async def astep1(x: Any) -> str:
        return f"{x}-1"

    async def astep2(x: Any) -> str:
        raise ValueError("Test error")

    seq = RunnableSeq(astep1, astep2)
    with pytest.raises(ValueError, match="Test error"):
        await seq.ainvoke("test")


def test_runnable_seq_ror_with_callable():
    def step1(x: Any) -> str:
        return f"{x}-1"

    def step2(x: Any) -> str:
        return f"{x}-2"

    def step3(x: Any) -> str:
        return f"{x}-3"

    seq = RunnableSeq(step2, step3)
    combined = step1 | seq
    result = combined.invoke("test")
    assert result == "test-1-2-3"


def test_runnable_callable_repr():
    def test_func(x: Any) -> str:
        return f"{x}"

    runnable = RunnableCallable(test_func, tags=["test"])
    repr_str = repr(runnable)
    assert "test_func" in repr_str
    assert "tags=['test']" in repr_str


async def test_runnable_seq_astream_error():
    async def astep1(x: Any) -> str:
        return f"{x}-1"

    async def astep2(x: Any) -> str:
        raise ValueError("Test error")

    seq = RunnableSeq(astep1, astep2)
    with pytest.raises(ValueError, match="Test error"):
        async for _ in seq.astream("test"):
            pass


def test_runnable_callable_sync_error():
    async def async_func(x: Any) -> str:
        return f"{x}"

    runnable = RunnableCallable(func=None, afunc=async_func)
    with pytest.raises(TypeError, match="No synchronous function provided"):
        runnable.invoke("test")
