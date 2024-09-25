from collections import defaultdict
from datetime import datetime, timezone
from typing import List, Optional

from langgraph.store.base import BaseStore, GetOp, Item, PutOp, SearchOp


class MemoryStore(BaseStore):
    def __init__(self) -> None:
        self.data: dict[tuple[str, ...], dict[str, Item]] = defaultdict(dict)

    def get(self, ops: List[GetOp]) -> List[Optional[Item]]:
        items = [self.data[op.namespace].get(op.id) for op in ops]
        for item in items:
            if item is not None:
                item.last_accessed_at = datetime.now(timezone.utc)
        return items

    def search(self, ops: List[SearchOp]) -> List[List[Item]]:
        results: list[list[Item]] = []
        for op in ops:
            candidates = [
                item
                for namespace, items in self.data.items()
                if (
                    namespace[: len(op.namespace_prefix)] == op.namespace_prefix
                    if len(namespace) >= len(op.namespace_prefix)
                    else False
                )
                for item in items.values()
            ]
            if op.query is not None:
                raise NotImplementedError("Search queries are not supported")
            if op.filter:
                candidates = [
                    item
                    for item in candidates
                    if item.value.items() >= op.filter.items()
                ]
            if op.weights:
                raise NotImplementedError("Search weights are not supported")
            results.append(candidates[op.offset : op.offset + op.limit])
        return results

    def put(self, ops: List[PutOp]) -> None:
        for op in ops:
            if op.value is None:
                self.data[op.namespace].pop(op.id, None)
            elif op.id in self.data[op.namespace]:
                self.data[op.namespace][op.id].value = op.value
                self.data[op.namespace][op.id].updated_at = datetime.now(timezone.utc)
            else:
                self.data[op.namespace][op.id] = Item(
                    value=op.value,
                    scores={},
                    id=op.id,
                    namespace=op.namespace,
                    created_at=datetime.now(timezone.utc),
                    updated_at=datetime.now(timezone.utc),
                    last_accessed_at=datetime.now(timezone.utc),
                )

    async def aget(self, ops: List[GetOp]) -> List[Optional[Item]]:
        return self.get(ops)

    async def asearch(self, ops: List[SearchOp]) -> List[List[Item]]:
        return self.search(ops)

    async def aput(self, ops: List[PutOp]) -> None:
        return self.put(ops)
