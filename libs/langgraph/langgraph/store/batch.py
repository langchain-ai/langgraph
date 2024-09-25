import asyncio
from typing import Optional, Union, cast

from langgraph.store.base import BaseStore, GetOp, Item, PutOp, SearchOp

Ops = Union[list[GetOp], list[SearchOp], list[PutOp]]


class AsyncBatchedStore(BaseStore):
    def __init__(self, store: BaseStore) -> None:
        self._store = store
        self._loop = asyncio.get_running_loop()
        self._aqueue: dict[asyncio.Future, Ops] = {}
        self._task = self._loop.create_task(_run(self._aqueue, self._store))

    def __del__(self) -> None:
        self._task.cancel()

    async def aget(self, ops: list[GetOp]) -> list[Optional[Item]]:
        fut = self._loop.create_future()
        self._aqueue[fut] = ops
        return await fut

    async def asearch(self, ops: list[SearchOp]) -> list[list[Item]]:
        fut = self._loop.create_future()
        self._aqueue[fut] = ops
        return await fut

    async def aput(self, ops: list[PutOp]) -> None:
        fut = self._loop.create_future()
        self._aqueue[fut] = ops
        return await fut


async def _run(aqueue: dict[asyncio.Future, Ops], store: BaseStore) -> None:
    while True:
        await asyncio.sleep(0)
        if not aqueue:
            continue
        # get the operations to run
        taken = aqueue.copy()
        # action each operation
        if gets := {
            f: cast(list[GetOp], ops)
            for f, ops in taken.items()
            if isinstance(ops[0], GetOp)
        }:
            try:
                results = await store.aget([g for ops in gets.values() for g in ops])
                for fut, ops in gets.items():
                    fut.set_result(results[: len(ops)])
                    results = results[len(ops) :]
            except Exception as e:
                for fut in gets:
                    fut.set_exception(e)
        if searches := {
            f: cast(list[SearchOp], ops)
            for f, ops in taken.items()
            if isinstance(ops[0], SearchOp)
        }:
            try:
                results = await store.asearch(
                    [s for ops in searches.values() for s in ops]
                )
                for fut, ops in searches.items():
                    fut.set_result(results[: len(ops)])
                    results = results[len(ops) :]
            except Exception as e:
                for fut in searches:
                    fut.set_exception(e)
        if puts := {
            f: cast(list[PutOp], ops)
            for f, ops in taken.items()
            if isinstance(ops[0], PutOp)
        }:
            try:
                await store.aput([p for ops in puts.values() for p in ops])
                for fut in puts:
                    fut.set_result(None)
            except Exception as e:
                for fut in puts:
                    fut.set_exception(e)
        # remove the operations from the queue
        for fut in taken:
            del aqueue[fut]
