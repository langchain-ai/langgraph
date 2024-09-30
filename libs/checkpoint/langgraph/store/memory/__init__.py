from collections import defaultdict
from datetime import datetime, timezone
from typing import Iterable

from langgraph.store.base import (
    BaseStore,
    GetOp,
    Item,
    ListNamespacesOp,
    MatchCondition,
    Op,
    PutOp,
    Result,
    SearchOp,
)


class InMemoryStore(BaseStore):
    """A KV store backed by an in-memory python dictionary.

    Useful for testing/experimentation and lightweight PoC's.
    For actual persistence, use a Store backed by a proper database.
    """

    __slots__ = ("_data",)

    def __init__(self) -> None:
        self._data: dict[tuple[str, ...], dict[str, Item]] = defaultdict(dict)

    def batch(self, ops: Iterable[Op]) -> list[Result]:
        results: list[Result] = []
        for op in ops:
            if isinstance(op, GetOp):
                item = self._data[op.namespace].get(op.key)
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
                    self._data[op.namespace].pop(op.key, None)
                elif op.key in self._data[op.namespace]:
                    self._data[op.namespace][op.key].value = op.value
                    self._data[op.namespace][op.key].updated_at = datetime.now(
                        timezone.utc
                    )
                else:
                    self._data[op.namespace][op.key] = Item(
                        value=op.value,
                        key=op.key,
                        namespace=op.namespace,
                        created_at=datetime.now(timezone.utc),
                        updated_at=datetime.now(timezone.utc),
                    )
                results.append(None)
            elif isinstance(op, ListNamespacesOp):
                results.append(self._handle_list_namespaces(op))
        return results

    async def abatch(self, ops: Iterable[Op]) -> list[Result]:
        return self.batch(ops)

    def _handle_list_namespaces(self, op: ListNamespacesOp) -> list[tuple[str, ...]]:
        all_namespaces = list(
            self._data.keys()
        )  # Avoid collection size changing while iterating
        namespaces = all_namespaces
        if op.match_conditions:
            namespaces = [
                ns
                for ns in namespaces
                if all(_does_match(condition, ns) for condition in op.match_conditions)
            ]

        if op.max_depth is not None:
            namespaces = sorted({ns[: op.max_depth] for ns in namespaces})
        else:
            namespaces = sorted(namespaces)
        return namespaces[op.offset : op.offset + op.limit]


def _does_match(match_condition: MatchCondition, key: tuple[str, ...]) -> bool:
    match_type = match_condition.match_type
    path = match_condition.path

    if len(key) < len(path):
        return False

    if match_type == "prefix":
        for k_elem, p_elem in zip(key, path):
            if p_elem == "*":
                continue  # Wildcard matches any element
            if k_elem != p_elem:
                return False
        return True
    elif match_type == "suffix":
        for k_elem, p_elem in zip(reversed(key), reversed(path)):
            if p_elem == "*":
                continue  # Wildcard matches any element
            if k_elem != p_elem:
                return False
        return True
    else:
        raise ValueError(f"Unsupported match type: {match_type}")
