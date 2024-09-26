from collections import defaultdict
from datetime import datetime, timezone

from langgraph.store.base import BaseStore, GetOp, Item, Op, PutOp, Result, SearchOp


class InMemoryStore(BaseStore):
    """A KV store backed by an in-memory python dictionary.

    Useful for testing/experimentation and lightweight PoC's.
    For actual persistence, use a Store backed by a proper database.
    """

    __slots__ = ("_data",)

    def __init__(self) -> None:
        self._data: dict[tuple[str, ...], dict[str, Item]] = defaultdict(dict)

    def batch(self, ops: list[Op]) -> list[Result]:
        results: list[Result] = []
        for op in ops:
            if isinstance(op, GetOp):
                item = self._data[op.namespace].get(op.id)
                if item is not None:
                    item.last_accessed_at = datetime.now(timezone.utc)
                results.append(item)
            elif isinstance(op, SearchOp):
                candidates = [
                    item
                    for namespace, items in self._data.items()
                    if (
                        namespace[: len(op.namespace_prefix)] == op.namespace_prefix
                        if len(namespace) >= len(op.namespace_prefix)
                        else False
                    )
                    for item in items.values()
                ]
                if op.filter:
                    candidates = [
                        item
                        for item in candidates
                        if item.value.items() >= op.filter.items()
                    ]
                results.append(candidates[op.offset : op.offset + op.limit])
            elif isinstance(op, PutOp):
                if op.value is None:
                    self._data[op.namespace].pop(op.id, None)
                elif op.id in self._data[op.namespace]:
                    self._data[op.namespace][op.id].value = op.value
                    self._data[op.namespace][op.id].updated_at = datetime.now(
                        timezone.utc
                    )
                else:
                    self._data[op.namespace][op.id] = Item(
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
