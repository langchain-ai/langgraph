import asyncio
from typing import Any, List, NamedTuple, Optional, Union

V = dict[str, Any]


class BaseKV:
    def get(self, pairs: List[tuple[str, str]]) -> dict[tuple[str, str], Optional[V]]:
        # list[(namespace, key)] -> dict[(namespace, key), value | none]
        raise NotImplementedError

    def list(self, prefixes: List[str]) -> dict[str, dict[str, V]]:
        # list[namespace] -> dict[namespace, list[value]]
        raise NotImplementedError

    def put(self, writes: List[tuple[str, str, Optional[V]]]) -> None:
        # list[(namespace, key, value | none)] -> None
        raise NotImplementedError

    async def aget(
        self, pairs: List[tuple[str, str]]
    ) -> dict[tuple[str, str], Optional[V]]:
        # list[(namespace, key)] -> dict[(namespace, key), value | none]
        raise NotImplementedError

    async def alist(self, prefixes: List[str]) -> dict[str, dict[str, V]]:
        # list[namespace] -> dict[namespace, list[value]]
        raise NotImplementedError

    async def aput(self, writes: List[tuple[str, str, Optional[V]]]) -> None:
        # list[(namespace, key, value | none)] -> None
        raise NotImplementedError


class GetOp(NamedTuple):
    pairs: List[tuple[str, str]]


class ListOp(NamedTuple):
    prefixes: List[str]


class PutOp(NamedTuple):
    writes: List[tuple[str, str, Optional[V]]]


class KeyValueStore(BaseKV):
    def __init__(self, kv: BaseKV) -> None:
        self.kv = kv
        self.aqueue: dict[asyncio.Future, Union[GetOp, ListOp, PutOp]] = {}
        self.task = asyncio.create_task(_run(self.aqueue, self.kv))

    def __del__(self) -> None:
        self.task.cancel()

    async def aget(
        self, pairs: List[tuple[str, str]]
    ) -> dict[tuple[str, str], Optional[V]]:
        fut = asyncio.get_running_loop().create_future()
        self.aqueue[fut] = GetOp(pairs)
        return await fut

    async def alist(self, prefixes: List[str]) -> dict[str, dict[str, V]]:
        fut = asyncio.get_running_loop().create_future()
        self.aqueue[fut] = ListOp(prefixes)
        return await fut

    async def aput(self, writes: List[tuple[str, str, Optional[V]]]) -> None:
        fut = asyncio.get_running_loop().create_future()
        self.aqueue[fut] = PutOp(writes)
        return await fut


async def _run(
    aqueue: dict[asyncio.Future, Union[GetOp, ListOp, PutOp]], kv: BaseKV
) -> None:
    while True:
        await asyncio.sleep(0)
        if not aqueue:
            continue
        # this could use a lock, if we want thread safety
        taken = aqueue.copy()
        aqueue.clear()
        # action each operation
        gets = {f: o for f, o in taken.items() if isinstance(o, GetOp)}
        if gets:
            try:
                results = await kv.aget([p for op in gets.values() for p in op.pairs])
                for fut, op in gets.items():
                    fut.set_result({k: results.get(k) for k in op.pairs})
            except Exception as e:
                for fut in gets:
                    fut.set_exception(e)
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
