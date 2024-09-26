import asyncio
from datetime import datetime
from typing import Iterable

from pytest_mock import MockerFixture

from langgraph.store.base import GetOp, Item, Op, Result
from langgraph.store.batch import AsyncBatchedBaseStore


async def test_async_batch_store(mocker: MockerFixture) -> None:
    abatch = mocker.stub()

    class MockStore(AsyncBatchedBaseStore):
        def batch(self, ops: Iterable[Op]) -> list[Result]:
            raise NotImplementedError

        async def abatch(self, ops: Iterable[Op]) -> list[Result]:
            assert all(isinstance(op, GetOp) for op in ops)
            abatch(ops)
            return [
                Item(
                    value={},
                    scores={},
                    id=getattr(op, "id", ""),
                    namespace=getattr(op, "namespace", ()),
                    created_at=datetime(2024, 9, 24, 17, 29, 10, 128397),
                    updated_at=datetime(2024, 9, 24, 17, 29, 10, 128397),
                    last_accessed_at=datetime(2024, 9, 24, 17, 29, 10, 128397),
                )
                for op in ops
            ]

    store = MockStore()

    # concurrent calls are batched
    results = await asyncio.gather(
        store.aget(namespace=("a",), id="b"),
        store.aget(namespace=("c",), id="d"),
    )
    assert results == [
        Item(
            {},
            {},
            "b",
            ("a",),
            datetime(2024, 9, 24, 17, 29, 10, 128397),
            datetime(2024, 9, 24, 17, 29, 10, 128397),
            datetime(2024, 9, 24, 17, 29, 10, 128397),
        ),
        Item(
            {},
            {},
            "d",
            ("c",),
            datetime(2024, 9, 24, 17, 29, 10, 128397),
            datetime(2024, 9, 24, 17, 29, 10, 128397),
            datetime(2024, 9, 24, 17, 29, 10, 128397),
        ),
    ]
    assert abatch.call_count == 1
    assert [tuple(c.args[0]) for c in abatch.call_args_list] == [
        (
            GetOp(("a",), "b"),
            GetOp(("c",), "d"),
        ),
    ]
