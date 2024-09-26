from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Iterable, NamedTuple, Optional, Sequence, Union


@dataclass
class Item:
    value: dict[str, Any]
    # search metadata
    scores: dict[str, float]
    # item metadata
    id: str
    namespace: tuple[str, ...]
    created_at: datetime
    updated_at: datetime
    last_accessed_at: datetime


class GetOp(NamedTuple):
    namespace: tuple[str, ...]
    id: str


class SearchOp(NamedTuple):
    namespace_prefix: tuple[str, ...]
    filter: Optional[dict[str, Any]] = None
    limit: int = 10
    offset: int = 0


class PutOp(NamedTuple):
    namespace: tuple[str, ...]
    id: str
    value: Optional[dict[str, Any]]


Op = Union[GetOp, SearchOp, PutOp]
Result = Union[Item, list[Item], None]


class BaseStore(ABC):
    __slots__ = ("__weakref__",)

    # abstract methods

    @abstractmethod
    def batch(
        self,
        ops: Iterable[Op],
    ) -> Sequence[Result]: ...

    @abstractmethod
    async def abatch(
        self,
        ops: Iterable[Op],
    ) -> Sequence[Result]: ...

    # convenience methods

    def get(
        self,
        namespace: tuple[str, ...],
        id: str,
    ) -> Optional[Item]:
        return self.batch([GetOp(namespace, id)])[0]

    def search(
        self,
        namespace_prefix: tuple[str, ...],
        /,
        *,
        filter: Optional[dict[str, Any]] = None,
        limit: int = 10,
        offset: int = 0,
    ) -> list[Item]:
        return self.batch(
            [
                SearchOp(namespace_prefix, filter, limit, offset),
            ]
        )[0]

    def put(
        self,
        namespace: tuple[str, ...],
        id: str,
        value: dict[str, Any],
    ) -> None:
        self.batch([PutOp(namespace, id, value)])

    def delete(
        self,
        namespace: tuple[str, ...],
        id: str,
    ) -> None:
        self.batch([PutOp(namespace, id, None)])

    async def aget(
        self,
        namespace: tuple[str, ...],
        id: str,
    ) -> Optional[Item]:
        return (await self.abatch([GetOp(namespace, id)]))[0]

    async def asearch(
        self,
        namespace_prefix: tuple[str, ...],
        /,
        *,
        filter: Optional[dict[str, Any]] = None,
        limit: int = 10,
        offset: int = 0,
    ) -> list[Item]:
        return (
            await self.abatch(
                [
                    SearchOp(namespace_prefix, filter, limit, offset),
                ]
            )
        )[0]

    async def aput(
        self,
        namespace: tuple[str, ...],
        id: str,
        value: dict[str, Any],
    ) -> None:
        await self.abatch([PutOp(namespace, id, value)])

    async def adelete(
        self,
        namespace: tuple[str, ...],
        id: str,
    ) -> None:
        await self.abatch([PutOp(namespace, id, None)])
