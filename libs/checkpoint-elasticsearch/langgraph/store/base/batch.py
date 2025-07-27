"""Utilities for batching operations in a background task."""

from __future__ import annotations

import asyncio
import functools
import weakref
from collections.abc import Iterable
from typing import Any, Callable, Literal, TypeVar

from langgraph.store.base import (
    NOT_PROVIDED,
    BaseStore,
    GetOp,
    Item,
    ListNamespacesOp,
    MatchCondition,
    NamespacePath,
    NotProvided,
    Op,
    PutOp,
    Result,
    SearchItem,
    SearchOp,
    _ensure_refresh,
    _ensure_ttl,
    _validate_namespace,
)

F = TypeVar("F", bound=Callable)


def _check_loop(func: F) -> F:
    @functools.wraps(func)
    def wrapper(store: AsyncBatchedBaseStore, *args: Any, **kwargs: Any) -> Any:
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
        self._aqueue: asyncio.Queue[tuple[asyncio.Future, Op]] = asyncio.Queue()
        self._task = self._loop.create_task(_run(self._aqueue, weakref.ref(self)))

    def __del__(self) -> None:
        try:
            self._task.cancel()
        except RuntimeError:
            pass

    async def aget(
        self,
        namespace: tuple[str, ...],
        key: str,
        *,
        refresh_ttl: bool | None = None,
    ) -> Item | None:
        assert not self._task.done()
        fut = self._loop.create_future()
        self._aqueue.put_nowait(
            (
                fut,
                GetOp(
                    namespace,
                    key,
                    refresh_ttl=_ensure_refresh(self.ttl_config, refresh_ttl),
                ),
            )
        )
        return await fut

    async def asearch(
        self,
        namespace_prefix: tuple[str, ...],
        /,
        *,
        query: str | None = None,
        filter: dict[str, Any] | None = None,
        limit: int = 10,
        offset: int = 0,
        refresh_ttl: bool | None = None,
    ) -> list[SearchItem]:
        assert not self._task.done()
        fut = self._loop.create_future()
        self._aqueue.put_nowait(
            (
                fut,
                SearchOp(
                    namespace_prefix,
                    filter,
                    limit,
                    offset,
                    query,
                    refresh_ttl=_ensure_refresh(self.ttl_config, refresh_ttl),
                ),
            )
        )
        return await fut

    async def aput(
        self,
        namespace: tuple[str, ...],
        key: str,
        value: dict[str, Any],
        index: Literal[False] | list[str] | None = None,
        *,
        ttl: float | None | NotProvided = NOT_PROVIDED,
    ) -> None:
        assert not self._task.done()
        _validate_namespace(namespace)
        fut = self._loop.create_future()
        self._aqueue.put_nowait(
            (
                fut,
                PutOp(
                    namespace, key, value, index, ttl=_ensure_ttl(self.ttl_config, ttl)
                ),
            )
        )
        return await fut

    async def adelete(
        self,
        namespace: tuple[str, ...],
        key: str,
    ) -> None:
        assert not self._task.done()
        fut = self._loop.create_future()
        self._aqueue.put_nowait((fut, PutOp(namespace, key, None)))
        return await fut

    async def alist_namespaces(
        self,
        *,
        prefix: NamespacePath | None = None,
        suffix: NamespacePath | None = None,
        max_depth: int | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> list[tuple[str, ...]]:
        assert not self._task.done()
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
        self._aqueue.put_nowait((fut, op))
        return await fut

    @_check_loop
    def batch(self, ops: Iterable[Op]) -> list[Result]:
        return asyncio.run_coroutine_threadsafe(self.abatch(ops), self._loop).result()

    @_check_loop
    def get(
        self,
        namespace: tuple[str, ...],
        key: str,
        *,
        refresh_ttl: bool | None = None,
    ) -> Item | None:
        return asyncio.run_coroutine_threadsafe(
            self.aget(namespace, key=key, refresh_ttl=refresh_ttl), self._loop
        ).result()

    @_check_loop
    def search(
        self,
        namespace_prefix: tuple[str, ...],
        /,
        *,
        query: str | None = None,
        filter: dict[str, Any] | None = None,
        limit: int = 10,
        offset: int = 0,
        refresh_ttl: bool | None = None,
    ) -> list[SearchItem]:
        return asyncio.run_coroutine_threadsafe(
            self.asearch(
                namespace_prefix,
                query=query,
                filter=filter,
                limit=limit,
                offset=offset,
                refresh_ttl=refresh_ttl,
            ),
            self._loop,
        ).result()

    @_check_loop
    def put(
        self,
        namespace: tuple[str, ...],
        key: str,
        value: dict[str, Any],
        index: Literal[False] | list[str] | None = None,
        *,
        ttl: float | None | NotProvided = NOT_PROVIDED,
    ) -> None:
        _validate_namespace(namespace)
        asyncio.run_coroutine_threadsafe(
            self.aput(
                namespace,
                key=key,
                value=value,
                index=index,
                ttl=_ensure_ttl(self.ttl_config, ttl),
            ),
            self._loop,
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
        prefix: NamespacePath | None = None,
        suffix: NamespacePath | None = None,
        max_depth: int | None = None,
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


def _dedupe_ops(values: list[Op]) -> tuple[list[int] | None, list[Op]]:
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
    aqueue: asyncio.Queue[tuple[asyncio.Future, Op]],
    store: weakref.ReferenceType[BaseStore],
) -> None:
    while item := await aqueue.get():
        # check if store is still alive
        if s := store():
            try:
                # accumulate operations scheduled in same tick
                items = [item]
                try:
                    while item := aqueue.get_nowait():
                        items.append(item)
                except asyncio.QueueEmpty:
                    pass
                # get the operations to run
                futs = [item[0] for item in items]
                values = [item[1] for item in items]
                # action each operation
                try:
                    listen, dedupped = _dedupe_ops(values)
                    results = await s.abatch(dedupped)
                    if listen is not None:
                        results = [results[ix] for ix in listen]

                    # set the results of each operation
                    for fut, result in zip(futs, results):
                        # guard against future being done (e.g. cancelled)
                        if not fut.done():
                            fut.set_result(result)
                except Exception as e:
                    for fut in futs:
                        # guard against future being done (e.g. cancelled)
                        if not fut.done():
                            fut.set_exception(e)
            finally:
                # remove strong ref to store
                del s
        else:
            break
