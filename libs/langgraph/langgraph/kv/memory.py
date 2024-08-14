from collections import defaultdict
from typing import List

from langgraph.kv.base import BaseKV, V


class MemoryKV(BaseKV):
    def __init__(self) -> None:
        self.data: dict[str, dict[str, V]] = defaultdict(dict)

    def get(self, pairs: List[tuple[str, str]]) -> dict[tuple[str, str], V | None]:
        return {pair: self.data[pair[0]].get(pair[1]) for pair in pairs}

    async def aget(
        self, pairs: List[tuple[str, str]]
    ) -> dict[tuple[str, str], V | None]:
        return self.get(pairs)

    def list(self, prefixes: List[str]) -> dict[str, dict[str, V]]:
        return {prefix: self.data[prefix] for prefix in prefixes}

    async def alist(self, prefixes: List[str]) -> dict[str, dict[str, V]]:
        return self.list(prefixes)

    def put(self, writes: List[tuple[str, str, V | None]]) -> None:
        for namespace, key, value in writes:
            if value is None:
                self.data[namespace].pop(key, None)
            else:
                self.data[namespace][key] = value

    async def aput(self, writes: List[tuple[str, str, V | None]]) -> None:
        self.put(writes)
