import asyncio
import threading
import time
import weakref
from concurrent.futures import Future
from typing import Any, Iterable, Literal, Optional, Union

from langgraph.store.base import (
    BaseStore,
    GetOp,
    Item,
    ListNamespacesOp,
    MatchCondition,
    NamespacePath,
    Op,
    PutOp,
    Result,
    SearchItem,
    SearchOp,
    _validate_namespace,
)


class AsyncBatchedBaseStoreMixin:
    """Efficiently batch operations in a background task."""

    _loop: asyncio.AbstractEventLoop
    _aqueue: dict[asyncio.Future, Op]
    _task: asyncio.Task

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
        query: Optional[str] = None,
        filter: Optional[dict[str, Any]] = None,
        limit: int = 10,
        offset: int = 0,
    ) -> list[SearchItem]:
        fut = self._loop.create_future()
        self._aqueue[fut] = SearchOp(namespace_prefix, filter, limit, offset, query)
        return await fut

    async def aput(
        self,
        namespace: tuple[str, ...],
        key: str,
        value: dict[str, Any],
        index: Optional[Union[Literal[False], list[str]]] = None,
    ) -> None:
        _validate_namespace(namespace)
        fut = self._loop.create_future()
        self._aqueue[fut] = PutOp(namespace, key, value, index)
        return await fut

    async def adelete(
        self,
        namespace: tuple[str, ...],
        key: str,
    ) -> None:
        fut = self._loop.create_future()
        self._aqueue[fut] = PutOp(namespace, key, None)
        return await fut

    async def alist_namespaces(
        self,
        *,
        prefix: Optional[NamespacePath] = None,
        suffix: Optional[NamespacePath] = None,
        max_depth: Optional[int] = None,
        limit: int = 100,
        offset: int = 0,
    ) -> list[tuple[str, ...]]:
        fut = self._loop.create_future()
        match_conditions = []
        if prefix:
            match_conditions.append(MatchCondition(match_type="prefix", path=prefix))
        if suffix:
            match_conditions.append(MatchCondition(match_type="suffix", path=suffix))

        op = ListNamespacesOp(
            match_conditions=tuple(match_conditions),
            max_depth=max_depth,
            limit=limit,
            offset=offset,
        )
        self._aqueue[fut] = op
        return await fut


class AsyncBatchedBaseStore(AsyncBatchedBaseStoreMixin, BaseStore):
    """Efficiently batch operations in a background task."""

    __slots__ = ("_loop", "_aqueue", "_task")

    def __init__(self) -> None:
        super().__init__()
        self._loop = asyncio.get_running_loop()
        self._aqueue: dict[asyncio.Future, Op] = {}
        self._task = self._loop.create_task(_run(self._aqueue, weakref.ref(self)))

    def __del__(self) -> None:
        self._task.cancel()

    def batch(self, ops: Iterable[Op]) -> list[Result]:
        futures = []
        for op in ops:
            fut = self._loop.create_future()
            self._aqueue[fut] = op
            futures.append(fut)
        return [fut.result() for fut in asyncio.as_completed(futures)]


def _dedupe_ops(values: list[Op]) -> tuple[Optional[list[int]], list[Op]]:
    """Dedupe operations while preserving order for results.

    Args:
        values: List of operations to dedupe

    Returns:
        Tuple of (listen indices, deduped operations)
        where listen indices map deduped operation results back to original positions
    """
    if len(values) <= 1:
        return None, list(values)

    dedupped: list[Op] = []
    listen: list[int] = []
    puts: dict[tuple[tuple[str, ...], str], int] = {}

    for op in values:
        if isinstance(op, (GetOp, SearchOp, ListNamespacesOp)):
            try:
                listen.append(dedupped.index(op))
            except ValueError:
                listen.append(len(dedupped))
                dedupped.append(op)
        elif isinstance(op, PutOp):
            putkey = (op.namespace, op.key)
            if putkey in puts:
                # Overwrite previous put
                ix = puts[putkey]
                dedupped[ix] = op
                listen.append(ix)
            else:
                puts[putkey] = len(dedupped)
                listen.append(len(dedupped))
                dedupped.append(op)

        else:  # Any new ops will be treated regularly
            listen.append(len(dedupped))
            dedupped.append(op)

    return listen, dedupped


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
                values = list(taken.values())
                listen, dedupped = _dedupe_ops(values)
                results = await s.abatch(dedupped)
                if listen is not None:
                    results = [results[ix] for ix in listen]

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


class SyncBatchedBaseStoreMixin(BaseStore):
    """Efficiently batch operations in a background thread."""

    _sync_queue: dict[Future, Op]
    _sync_thread: threading.Thread

    def get(
        self,
        namespace: tuple[str, ...],
        key: str,
    ) -> Optional[Item]:
        fut: Future[Optional[Item]] = Future()
        self._sync_queue[fut] = GetOp(namespace, key)
        return fut.result()

    def search(
        self,
        namespace_prefix: tuple[str, ...],
        /,
        *,
        query: Optional[str] = None,
        filter: Optional[dict[str, Any]] = None,
        limit: int = 10,
        offset: int = 0,
    ) -> list[SearchItem]:
        fut: Future[list[SearchItem]] = Future()
        self._sync_queue[fut] = SearchOp(namespace_prefix, filter, limit, offset, query)
        return fut.result()

    def put(
        self,
        namespace: tuple[str, ...],
        key: str,
        value: dict[str, Any],
        index: Optional[Union[Literal[False], list[str]]] = None,
    ) -> None:
        _validate_namespace(namespace)
        fut: Future[None] = Future()
        self._sync_queue[fut] = PutOp(namespace, key, value, index)
        return fut.result()

    def delete(
        self,
        namespace: tuple[str, ...],
        key: str,
    ) -> None:
        fut: Future[None] = Future()
        self._sync_queue[fut] = PutOp(namespace, key, None)
        return fut.result()

    def list_namespaces(
        self,
        *,
        prefix: Optional[NamespacePath] = None,
        suffix: Optional[NamespacePath] = None,
        max_depth: Optional[int] = None,
        limit: int = 100,
        offset: int = 0,
    ) -> list[tuple[str, ...]]:
        fut: Future[list[tuple[str, ...]]] = Future()
        match_conditions = []
        if prefix:
            match_conditions.append(MatchCondition(match_type="prefix", path=prefix))
        if suffix:
            match_conditions.append(MatchCondition(match_type="suffix", path=suffix))

        op = ListNamespacesOp(
            match_conditions=tuple(match_conditions),
            max_depth=max_depth,
            limit=limit,
            offset=offset,
        )
        self._sync_queue[fut] = op
        return fut.result()


class SyncBatchedBaseStore(SyncBatchedBaseStoreMixin, BaseStore):
    """Efficiently batch operations in a background thread."""

    __slots__ = ("_sync_queue", "_sync_thread")

    def __init__(self) -> None:
        super().__init__()
        self._sync_queue: dict[Future, Op] = {}
        self._sync_thread = threading.Thread(
            target=_sync_run,
            args=(self._sync_queue, weakref.ref(self)),
            daemon=True,
        )
        self._sync_thread.start()

    def __del__(self) -> None:
        # Signal the thread to stop
        if self._sync_thread.is_alive():
            empty_future: Future = Future()
            self._sync_queue[empty_future] = None  # type: ignore
            self._sync_thread.join()

    async def abatch(self, ops: Iterable[Op]) -> list[Result]:
        futures = []
        for op in ops:
            fut: Future[Result] = Future()
            self._sync_queue[fut] = op
            futures.append(fut)
        return [fut.result() for fut in futures]


class BatchedBaseStore(
    AsyncBatchedBaseStoreMixin, SyncBatchedBaseStoreMixin, BaseStore
):
    __slots__ = (
        "_sync_queue",
        "_sync_thread",
        "_task",
        "_loop",
        "_aqueue",
    )

    def __init__(self) -> None:
        super().__init__()
        # Setup async processing
        self._loop = asyncio.get_running_loop()
        self._aqueue: dict[asyncio.Future, Op] = {}
        self._task = self._loop.create_task(_run(self._aqueue, weakref.ref(self)))

        self._sync_queue: dict[Future, Op] = {}
        self._sync_thread = threading.Thread(
            target=_sync_run,
            args=(self._sync_queue, weakref.ref(self)),
            daemon=True,
        )
        self._sync_thread.start()

    def __del__(self) -> None:
        # Signal the thread to stop
        if self._sync_thread.is_alive():
            empty_future: Future[None] = Future()
            self._sync_queue[empty_future] = None  # type: ignore
            self._sync_thread.join()

        # Signal the thread to stop
        if self._task is not None:
            self._task.cancel()


def _sync_run(queue: dict[Future, Op], store: weakref.ReferenceType[BaseStore]) -> None:
    while True:
        time.sleep(0.001)  # Yield to other threads
        if not queue:
            continue
        if s := store():
            # get the operations to run
            taken = queue.copy()
            # action each operation
            try:
                values = list(taken.values())
                if None in values:  # Exit signal
                    break
                listen, dedupped = _dedupe_ops(values)
                results = s.batch(dedupped)  # Note: Using sync batch here
                if listen is not None:
                    results = [results[ix] for ix in listen]

                # set the results of each operation
                for fut, result in zip(taken, results):
                    fut.set_result(result)
            except Exception as e:
                for fut in taken:
                    fut.set_exception(e)
            # remove the operations from the queue
            for fut in taken:
                del queue[fut]
        else:
            break
        # remove strong ref to store
        del s
