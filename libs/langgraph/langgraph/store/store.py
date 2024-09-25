from typing import Any, Optional
from weakref import WeakKeyDictionary

from langgraph.store.base import BaseStore, GetOp, Item, PutOp, SearchOp

CACHE: WeakKeyDictionary[BaseStore, "Store"] = WeakKeyDictionary()


def _get_store(store: BaseStore) -> "Store":
    if store not in CACHE:
        CACHE[store] = Store(store)
    return CACHE[store]


class Store:
    __slots__ = ("_store",)

    def __init__(self, store: BaseStore) -> None:
        self._store = store

    def get(
        self,
        namespace: tuple[str, ...],
        id: str,
    ) -> Optional[Item]:
        return self._store.get([GetOp(namespace, id)])[0]

    def search(
        self,
        namesapce_prefix: tuple[str, ...],
        /,
        *,
        query: Optional[str],
        filter: Optional[dict[str, Any]],
        weights: Optional[dict[str, float]],
        limit: int = 10,
        offset: int = 0,
    ) -> list[Item]:
        return self._store.search(
            [
                SearchOp(namesapce_prefix, query, filter, weights, limit, offset),
            ]
        )[0]

    def put(
        self,
        namespace: tuple[str, ...],
        id: str,
        value: dict[str, Any],
    ) -> None:
        self._store.put([PutOp(namespace, id, value)])

    def delete(
        self,
        namespace: tuple[str, ...],
        id: str,
    ) -> None:
        self._store.put([PutOp(namespace, id, None)])

    async def aget(
        self,
        namespace: tuple[str, ...],
        id: str,
    ) -> Optional[Item]:
        return (await self._store.aget([GetOp(namespace, id)]))[0]

    async def asearch(
        self,
        namesapce_prefix: tuple[str, ...],
        /,
        *,
        query: Optional[str],
        filter: Optional[dict[str, Any]],
        weights: Optional[dict[str, float]],
        limit: int = 10,
        offset: int = 0,
    ) -> list[Item]:
        return (
            await self._store.asearch(
                [
                    SearchOp(namesapce_prefix, query, filter, weights, limit, offset),
                ]
            )
        )[0]

    async def aput(
        self,
        namespace: tuple[str, ...],
        id: str,
        value: dict[str, Any],
    ) -> None:
        await self._store.aput([PutOp(namespace, id, value)])

    async def adelete(
        self,
        namespace: tuple[str, ...],
        id: str,
    ) -> None:
        await self._store.aput([PutOp(namespace, id, None)])
