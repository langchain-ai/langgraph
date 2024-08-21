from collections import defaultdict
from typing import List, Optional

from langgraph.kv.base import BaseMemory, V


class MemoryKV(BaseMemory):
    def __init__(self) -> None:
        self.data: dict[str, dict[str, V]] = defaultdict(dict)

    def list(self, prefixes: List[str]) -> dict[str, dict[str, V]]:
        return {prefix: self.data[prefix] for prefix in prefixes}

    async def alist(self, prefixes: List[str]) -> dict[str, dict[str, V]]:
        return self.list(prefixes)

    def put(self, writes: List[tuple[str, str, Optional[V]]]) -> None:
        for namespace, key, value in writes:
            if value is None:
                self.data[namespace].pop(key, None)
            else:
                self.data[namespace][key] = value

    async def aput(self, writes: List[tuple[str, str, Optional[V]]]) -> None:
        return self.put(writes)
