import asyncio
import weakref
from typing import Any, Optional

from langgraph.store.base import (
    BaseStore,
    GetOp,
    Item,
    Op,
    PutOp,
    SearchOp,
    _validate_namespace,
)


class AsyncBatchedBaseStore(BaseStore):
    """Efficiently batch operations in a background task."""

    __slots__ = ("_loop", "_aqueue", "_task")

    def __init__(self) -> None:
        self._loop = asyncio.get_running_loop()
        self._aqueue: dict[asyncio.Future, Op] = {}
        self._task = self._loop.create_task(_run(self._aqueue, weakref.ref(self)))

    def __del__(self) -> None:
        self._task.cancel()

    async def aget(
        self,
        namespace: tuple[str, ...],
        key: str,
    ) -> Optional[Item]:
        fut = self._loop.create_future()
        self._aqueue[fut] = GetOp(namespace, key)
        return await fut

    async def asearch(
        self,
        namespace_prefix: tuple[str, ...],
        /,
        *,
        filter: Optional[dict[str, Any]] = None,
        limit: int = 10,
        offset: int = 0,
    ) -> list[Item]:
        fut = self._loop.create_future()
        self._aqueue[fut] = SearchOp(namespace_prefix, filter, limit, offset)
        return await fut

    async def aput(
        self,
        namespace: tuple[str, ...],
        key: str,
        value: dict[str, Any],
    ) -> None:
        _validate_namespace(namespace)
        fut = self._loop.create_future()
        self._aqueue[fut] = PutOp(namespace, key, value)
        return await fut

    async def adelete(
        self,
        namespace: tuple[str, ...],
        key: str,
    ) -> None:
        fut = self._loop.create_future()
        self._aqueue[fut] = PutOp(namespace, key, None)
        return await fut


async def _run(
    aqueue: dict[asyncio.Future, Op], store: weakref.ReferenceType[BaseStore]
) -> None:
    while True:
        await asyncio.sleep(0)
        if not aqueue:
            continue
        if s := store():
            # get the operations to run
            taken = aqueue.copy()
            # action each operation
            try:
                results = await s.abatch(taken.values())
                # set the results of each operation
                for fut, result in zip(taken, results):
                    fut.set_result(result)
            except Exception as e:
                for fut in taken:
                    fut.set_exception(e)
            # remove the operations from the queue
            for fut in taken:
                del aqueue[fut]
        else:
            break
        # remove strong ref to store
        del s
