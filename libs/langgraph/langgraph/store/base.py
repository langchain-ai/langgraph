from dataclasses import dataclass
from datetime import datetime
from typing import Any, NamedTuple, Optional

SCORE_RECENCY = "recency"
SCORE_RELEVANCE = "relevance"


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
    query: Optional[str] = None
    filter: Optional[dict[str, Any]] = None
    weights: Optional[dict[str, float]] = None
    limit: int = 10
    offset: int = 0


class PutOp(NamedTuple):
    namespace: tuple[str, ...]
    id: str
    value: Optional[dict[str, Any]]


class BaseStore:
    __slots__ = ("__weakref__",)

    def get(self, ops: list[GetOp]) -> list[Optional[Item]]:
        raise NotImplementedError

    def search(self, ops: list[SearchOp]) -> list[list[Item]]:
        raise NotImplementedError

    def put(self, ops: list[PutOp]) -> None:
        raise NotImplementedError

    async def aget(self, ops: list[GetOp]) -> list[Optional[Item]]:
        raise NotImplementedError

    async def asearch(self, ops: list[SearchOp]) -> list[list[Item]]:
        raise NotImplementedError

    async def aput(self, ops: list[PutOp]) -> None:
        raise NotImplementedError
