import asyncio
from typing import Any, List

from pytest_mock import MockerFixture

from langgraph.kv.base import BaseKV, KeyValueStore


async def test_kv_queue(mocker: MockerFixture) -> None:
    aget = mocker.stub()

    class MockKV(BaseKV):
        async def aget(
            self, pairs: List[tuple[str, str]]
        ) -> dict[tuple[str, str], dict[str, Any] | None]:
            aget(pairs)
            return {pair: {0: pair[0], 1: pair[1]} for pair in pairs}

    store = KeyValueStore(MockKV())

    # concurrent calls are batched
    results = await asyncio.gather(
        store.aget([("a", "b")]),
        store.aget([("c", "d")]),
    )
    assert results == [
        {("a", "b"): {0: "a", 1: "b"}},
        {("c", "d"): {0: "c", 1: "d"}},
    ]
    assert [c.args for c in aget.call_args_list] == [
        ([("a", "b"), ("c", "d")],),
    ]
