from typing import Any, List, Optional

from langgraph.checkpoint.serde.base import SerializerProtocol
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer

V = dict[str, Any]


class BaseKV:
    def __init__(self, *, serde: SerializerProtocol = JsonPlusSerializer()) -> None:
        self.serde = serde

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
