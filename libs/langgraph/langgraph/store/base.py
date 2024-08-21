from typing import Any, List, Optional

V = dict[str, Any]


class BaseStore:
    def list(self, prefixes: List[str]) -> dict[str, dict[str, V]]:
        # list[namespace] -> dict[namespace, list[value]]
        raise NotImplementedError

    def put(self, writes: List[tuple[str, str, Optional[V]]]) -> None:
        # list[(namespace, key, value | none)] -> None
        raise NotImplementedError

    async def alist(self, prefixes: List[str]) -> dict[str, dict[str, V]]:
        # list[namespace] -> dict[namespace, list[value]]
        raise NotImplementedError

    async def aput(self, writes: List[tuple[str, str, Optional[V]]]) -> None:
        # list[(namespace, key, value | none)] -> None
        raise NotImplementedError
