import asyncio
from datetime import datetime
from typing import Optional

import pytest
from pytest_mock import MockerFixture

from langgraph.store.base import BaseStore, GetOp, Item
from langgraph.store.batch import AsyncBatchedStore

pytestmark = pytest.mark.anyio


async def test_async_batch_store(mocker: MockerFixture) -> None:
    aget = mocker.stub()

    class MockStore(BaseStore):
        async def aget(self, ops: list[GetOp]) -> list[Optional[Item]]:
            aget(ops)
            return [
                Item(
                    value={},
                    scores={},
                    id=op.id,
                    namespace=op.namespace,
                    created_at=datetime(2024, 9, 24, 17, 29, 10, 128397),
                    updated_at=datetime(2024, 9, 24, 17, 29, 10, 128397),
                    last_accessed_at=datetime(2024, 9, 24, 17, 29, 10, 128397),
                )
                for op in ops
            ]

    store = AsyncBatchedStore(MockStore())

    # concurrent calls are batched
    results = await asyncio.gather(
        store.aget([GetOp(("a",), "b")]),
        store.aget([GetOp(("c",), "d")]),
    )
    assert results == [
        [
            Item(
                {},
                {},
                "b",
                ("a",),
                datetime(2024, 9, 24, 17, 29, 10, 128397),
                datetime(2024, 9, 24, 17, 29, 10, 128397),
                datetime(2024, 9, 24, 17, 29, 10, 128397),
            )
        ],
        [
            Item(
                {},
                {},
                "d",
                ("c",),
                datetime(2024, 9, 24, 17, 29, 10, 128397),
                datetime(2024, 9, 24, 17, 29, 10, 128397),
                datetime(2024, 9, 24, 17, 29, 10, 128397),
            )
        ],
    ]
    assert [c.args for c in aget.call_args_list] == [
        (
            [
                GetOp(("a",), "b"),
                GetOp(("c",), "d"),
            ],
        ),
    ]
