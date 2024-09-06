import asyncio
from typing import Any, Optional

import pytest
from pytest_mock import MockerFixture

from langgraph.store.base import BaseStore
from langgraph.store.batch import AsyncBatchedStore

pytestmark = pytest.mark.anyio


async def test_async_batch_store(mocker: MockerFixture) -> None:
    aget = mocker.stub()
    alist = mocker.stub()

    class MockStore(BaseStore):
        async def aget(
            self, pairs: list[tuple[str, str]]
        ) -> dict[tuple[str, str], Optional[dict[str, Any]]]:
            aget(pairs)
            return {pair: 1 for pair in pairs}

        async def alist(self, prefixes: list[str]) -> dict[str, dict[str, Any]]:
            alist(prefixes)
            return {prefix: {prefix: 1} for prefix in prefixes}

    store = AsyncBatchedStore(MockStore())

    # concurrent calls are batched
    results = await asyncio.gather(
        store.alist(["a", "b"]),
        store.alist(["c", "d"]),
    )
    assert results == [{"a": {"a": 1}, "b": {"b": 1}}, {"c": {"c": 1}, "d": {"d": 1}}]
    assert [c.args for c in alist.call_args_list] == [
        (["a", "b", "c", "d"],),
    ]
