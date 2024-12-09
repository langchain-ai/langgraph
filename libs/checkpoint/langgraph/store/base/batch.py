import asyncio
import functools
import weakref
from typing import Any, Callable, Iterable, Literal, Optional, TypeVar, Union

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

F = TypeVar("F", bound=Callable)


def _check_loop(func: F) -> F:
    @functools.wraps(func)
    def wrapper(store: "AsyncBatchedBaseStore", *args: Any, **kwargs: Any) -> Any:
        method_name: str = func.__name__
        try:
            current_loop = asyncio.get_running_loop()
            if current_loop is store._loop:
                replacement_str = (
                    f"Specifically, replace `store.{method_name}(...)` with `await store.a{method_name}(...)"
                    if method_name
                    else "For example, replace `store.get(...)` with `await store.aget(...)`"
                )
                raise asyncio.InvalidStateError(
                    f"Synchronous calls to {store.__class__.__name__} detected in the main event loop. "
                    "This can lead to deadlocks or performance issues. "
                    "Please use the asynchronous interface for main thread operations. "
                    f"{replacement_str} "
                )
        except RuntimeError:
            pass
        return func(store, *args, **kwargs)

    return wrapper


class AsyncBatchedBaseStore(BaseStore):
    """Efficiently batch operations in a background task."""

    __slots__ = ("_loop", "_aqueue", "_task")

    def __init__(self) -> None:
        super().__init__()
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

    @_check_loop
    def batch(self, ops: Iterable[Op]) -> list[Result]:
        return asyncio.run_coroutine_threadsafe(self.abatch(ops), self._loop).result()

    @_check_loop
    def get(
        self,
        namespace: tuple[str, ...],
        key: str,
    ) -> Optional[Item]:
        return asyncio.run_coroutine_threadsafe(
            self.aget(namespace, key=key), self._loop
        ).result()

    @_check_loop
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
        return asyncio.run_coroutine_threadsafe(
            self.asearch(
                namespace_prefix, query=query, filter=filter, limit=limit, offset=offset
            ),
            self._loop,
        ).result()

    @_check_loop
    def put(
        self,
        namespace: tuple[str, ...],
        key: str,
        value: dict[str, Any],
        index: Optional[Union[Literal[False], list[str]]] = None,
    ) -> None:
        _validate_namespace(namespace)
        asyncio.run_coroutine_threadsafe(
            self.aput(namespace, key=key, value=value, index=index), self._loop
        ).result()

    @_check_loop
    def delete(
        self,
        namespace: tuple[str, ...],
        key: str,
    ) -> None:
        asyncio.run_coroutine_threadsafe(
            self.adelete(namespace, key=key), self._loop
        ).result()

    @_check_loop
    def list_namespaces(
        self,
        *,
        prefix: Optional[NamespacePath] = None,
        suffix: Optional[NamespacePath] = None,
        max_depth: Optional[int] = None,
        limit: int = 100,
        offset: int = 0,
    ) -> list[tuple[str, ...]]:
        return asyncio.run_coroutine_threadsafe(
            self.alist_namespaces(
                prefix=prefix,
                suffix=suffix,
                max_depth=max_depth,
                limit=limit,
                offset=offset,
            ),
            self._loop,
        ).result()


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
    aqueue: dict[asyncio.Future, Op],
    store: weakref.ReferenceType[BaseStore],
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
