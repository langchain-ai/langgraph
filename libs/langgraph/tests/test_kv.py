import asyncio
from typing import Any

from pytest_mock import MockerFixture

from langgraph.kv.base import BaseKV
from langgraph.kv.batch import AsyncBatchedKV


async def test_kv_async_batch(mocker: MockerFixture) -> None:
    aget = mocker.stub()
    alist = mocker.stub()

    class MockKV(BaseKV):
        async def aget(
            self, pairs: list[tuple[str, str]]
        ) -> dict[tuple[str, str], dict[str, Any] | None]:
            aget(pairs)
            return {pair: 1 for pair in pairs}

        async def alist(self, prefixes: list[str]) -> dict[str, dict[str, Any]]:
            alist(prefixes)
            return {prefix: {prefix: 1} for prefix in prefixes}

    store = AsyncBatchedKV(MockKV())

    # concurrent calls are batched
    results = await asyncio.gather(
        store.aget([("a", "b")]),
        store.aget([("c", "d")]),
    )
    assert results == [
        {("a", "b"): 1},
        {("c", "d"): 1},
    ]
    assert [c.args for c in aget.call_args_list] == [
        ([("a", "b"), ("c", "d")],),
    ]

    results = await asyncio.gather(
        store.alist(["a", "b"]),
        store.alist(["c", "d"]),
    )
    assert results == [{"a": {"a": 1}, "b": {"b": 1}}, {"c": {"c": 1}, "d": {"d": 1}}]
    assert [c.args for c in alist.call_args_list] == [
        (["a", "b", "c", "d"],),
    ]
