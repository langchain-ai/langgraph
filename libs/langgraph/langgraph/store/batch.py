import asyncio
from typing import NamedTuple, Optional, Union

from langgraph.store.base import BaseStore, V


class ListOp(NamedTuple):
    prefixes: list[str]


class PutOp(NamedTuple):
    writes: list[tuple[str, str, Optional[V]]]


class AsyncBatchedStore(BaseStore):
    def __init__(self, kv: BaseStore) -> None:
        self.kv = kv
        self.aqueue: dict[asyncio.Future, Union[ListOp, PutOp]] = {}
        self.task = asyncio.create_task(_run(self.aqueue, self.kv))

    def __del__(self) -> None:
        self.task.cancel()

    async def alist(self, prefixes: list[str]) -> dict[str, dict[str, V]]:
        fut = asyncio.get_running_loop().create_future()
        self.aqueue[fut] = ListOp(prefixes)
        return await fut

    async def aput(self, writes: list[tuple[str, str, Optional[V]]]) -> None:
        fut = asyncio.get_running_loop().create_future()
        self.aqueue[fut] = PutOp(writes)
        return await fut


async def _run(
    aqueue: dict[asyncio.Future, Union[ListOp, PutOp]], kv: BaseStore
) -> None:
    while True:
        await asyncio.sleep(0)
        if not aqueue:
            continue
        # this could use a lock, if we want thread safety
        taken = aqueue.copy()
        aqueue.clear()
        # action each operation
        lists = {f: o for f, o in taken.items() if isinstance(o, ListOp)}
        if lists:
            try:
                results = await kv.alist(
                    [p for op in lists.values() for p in op.prefixes]
                )
                for fut, op in lists.items():
                    fut.set_result({k: results.get(k) for k in op.prefixes})
            except Exception as e:
                for fut in lists:
                    fut.set_exception(e)
        puts = {f: o for f, o in taken.items() if isinstance(o, PutOp)}
        if puts:
            try:
                await kv.aput([w for op in puts.values() for w in op.writes])
                for fut in puts:
                    fut.set_result(None)
            except Exception as e:
                for fut in puts:
                    fut.set_exception(e)
