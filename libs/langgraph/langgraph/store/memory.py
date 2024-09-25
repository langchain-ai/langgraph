from collections import defaultdict
from datetime import datetime, timezone

from langgraph.store.base import BaseStore, GetOp, Item, Op, PutOp, Result, SearchOp


class MemoryStore(BaseStore):
    def __init__(self) -> None:
        self.data: dict[tuple[str, ...], dict[str, Item]] = defaultdict(dict)

    def batch(self, ops: list[Op]) -> list[Result]:
        results: list[Result] = []
        for op in ops:
            if isinstance(op, GetOp):
                item = self.data[op.namespace].get(op.id)
                if item is not None:
                    item.last_accessed_at = datetime.now(timezone.utc)
                results.append(item)
            elif isinstance(op, SearchOp):
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
            elif isinstance(op, PutOp):
                if op.value is None:
                    self.data[op.namespace].pop(op.id, None)
                elif op.id in self.data[op.namespace]:
                    self.data[op.namespace][op.id].value = op.value
                    self.data[op.namespace][op.id].updated_at = datetime.now(
                        timezone.utc
                    )
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
                results.append(None)
        return results

    async def abatch(self, ops: list[Op]) -> list[Result]:
        return self.batch(ops)
