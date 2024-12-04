"""In-memory dictionary-backed store with optional vector search.

!!! example "Examples"
    Basic key-value storage:
    ```python
    from langgraph.store.memory import InMemoryStore

    store = InMemoryStore()
    store.put(("users", "123"), "prefs", {"theme": "dark"})
    item = store.get(("users", "123"), "prefs")
    ```

    Vector search using LangChain embeddings:
    ```python
    from langchain.embeddings import init_embeddings
    from langgraph.store.memory import InMemoryStore

    store = InMemoryStore(
        index={
            "dims": 1536,
            "embed": init_embeddings("openai:text-embedding-3-small")
        }
    )

    # Store documents
    store.put(("docs",), "doc1", {"text": "Python tutorial"})
    store.put(("docs",), "doc2", {"text": "TypeScript guide"})

    # Search by similarity
    results = store.search(("docs",), query="python programming")
    ```

    Vector search using OpenAI SDK directly:
    ```python
    from openai import OpenAI
    from langgraph.store.memory import InMemoryStore

    client = OpenAI()

    def embed_texts(texts: list[str]) -> list[list[float]]:
        response = client.embeddings.create(
            model="text-embedding-3-small",
            input=texts
        )
        return [e.embedding for e in response.data]

    store = InMemoryStore(
        index={
            "dims": 1536,
            "embed": embed_texts
        }
    )

    # Store documents
    store.put(("docs",), "doc1", {"text": "Python tutorial"})
    store.put(("docs",), "doc2", {"text": "TypeScript guide"})

    # Search by similarity
    results = store.search(("docs",), query="python programming")
    ```

    Async vector search using OpenAI SDK:
    ```python
    from openai import AsyncOpenAI
    from langgraph.store.memory import InMemoryStore

    client = AsyncOpenAI()

    async def aembed_texts(texts: list[str]) -> list[list[float]]:
        response = await client.embeddings.create(
            model="text-embedding-3-small",
            input=texts
        )
        return [e.embedding for e in response.data]

    store = InMemoryStore(
        index={
            "dims": 1536,
            "embed": aembed_texts
        }
    )

    # Store documents
    await store.aput(("docs",), "doc1", {"text": "Python tutorial"})
    await store.aput(("docs",), "doc2", {"text": "TypeScript guide"})

    # Search by similarity
    results = await store.asearch(("docs",), query="python programming")
    ```

Warning:
    This store keeps all data in memory. Data is lost when the process exits.
    For persistence, use a database-backed store like PostgresStore.

Tip:
    For vector search, install numpy for better performance:
    ```bash
    pip install numpy
    ```
"""

import asyncio
import concurrent.futures as cf
import functools
import logging
from collections import defaultdict
from datetime import datetime, timezone
from importlib import util
from typing import Any, Iterable, Optional

from langchain_core.embeddings import Embeddings

from langgraph.store.base import (
    BaseStore,
    GetOp,
    IndexConfig,
    Item,
    ListNamespacesOp,
    MatchCondition,
    Op,
    PutOp,
    Result,
    SearchItem,
    SearchOp,
    ensure_embeddings,
    get_text_at_path,
    tokenize_path,
)

logger = logging.getLogger(__name__)


class InMemoryStore(BaseStore):
    """In-memory dictionary-backed store with optional vector search.

    !!! example "Examples"
        Basic key-value storage:
            store = InMemoryStore()
            store.put(("users", "123"), "prefs", {"theme": "dark"})
            item = store.get(("users", "123"), "prefs")

        Vector search with embeddings:
            from langchain.embeddings import init_embeddings
            store = InMemoryStore(index={
                "dims": 1536,
                "embed": init_embeddings("openai:text-embedding-3-small"),
                "fields": ["text"],
            })

            # Store documents
            store.put(("docs",), "doc1", {"text": "Python tutorial"})
            store.put(("docs",), "doc2", {"text": "TypeScript guide"})

            # Search by similarity
            results = store.search(("docs",), query="python programming")

    Note:
        Semantic search is disabled by default. You can enable it by providing an `index` configuration
        when creating the store. Without this configuration, all `index` arguments passed to
        `put` or `aput`will have no effect.

    Warning:
        This store keeps all data in memory. Data is lost when the process exits.
        For persistence, use a database-backed store like PostgresStore.

    Tip:
        For vector search, install numpy for better performance:
        ```bash
        pip install numpy
        ```
    """

    __slots__ = (
        "_data",
        "_vectors",
        "index_config",
        "embeddings",
    )

    def __init__(self, *, index: Optional[IndexConfig] = None) -> None:
        # Both _data and _vectors are wrapped in the In-memory API
        # Do not change their names
        self._data: dict[tuple[str, ...], dict[str, Item]] = defaultdict(dict)
        # [ns][key][path]
        self._vectors: dict[tuple[str, ...], dict[str, dict[str, list[float]]]] = (
            defaultdict(lambda: defaultdict(dict))
        )
        self.index_config = index
        if self.index_config:
            self.index_config = self.index_config.copy()
            self.embeddings: Optional[Embeddings] = ensure_embeddings(
                self.index_config.get("embed"),
            )
            self.index_config["__tokenized_fields"] = [
                (p, tokenize_path(p)) if p != "$" else (p, p)
                for p in (self.index_config.get("fields") or ["$"])
            ]

        else:
            self.index_config = None
            self.embeddings = None

    def batch(self, ops: Iterable[Op]) -> list[Result]:
        # The batch/abatch methods are treated as internal.
        # Users should access via put/search/get/list_namespaces/etc.
        results, put_ops, search_ops = self._prepare_ops(ops)
        if search_ops:
            queryinmem_store = self._embed_search_queries(search_ops)
            self._batch_search(search_ops, queryinmem_store, results)

        to_embed = self._extract_texts(put_ops)
        if to_embed and self.index_config and self.embeddings:
            embeddings = self.embeddings.embed_documents(list(to_embed))
            self._insertinmem_store(to_embed, embeddings)
        self._apply_put_ops(put_ops)
        return results

    async def abatch(self, ops: Iterable[Op]) -> list[Result]:
        # The batch/abatch methods are treated as internal.
        # Users should access via put/search/get/list_namespaces/etc.
        results, put_ops, search_ops = self._prepare_ops(ops)
        if search_ops:
            queryinmem_store = await self._aembed_search_queries(search_ops)
            self._batch_search(search_ops, queryinmem_store, results)

        to_embed = self._extract_texts(put_ops)
        if to_embed and self.index_config and self.embeddings:
            embeddings = await self.embeddings.aembed_documents(list(to_embed))
            self._insertinmem_store(to_embed, embeddings)
        self._apply_put_ops(put_ops)
        return results

    # Helpers

    def _filter_items(self, op: SearchOp) -> list[tuple[Item, list[list[float]]]]:
        """Filter items by namespace and filter function, return items with their embeddings."""
        namespace_prefix = op.namespace_prefix

        def filter_func(item: Item) -> bool:
            if not op.filter:
                return True

            return all(
                _compare_values(item.value.get(key), filter_value)
                for key, filter_value in op.filter.items()
            )

        filtered = []
        for namespace in self._data:
            if not (
                namespace[: len(namespace_prefix)] == namespace_prefix
                if len(namespace) >= len(namespace_prefix)
                else False
            ):
                continue

            for key, item in self._data[namespace].items():
                if filter_func(item):
                    if op.query and (embeddings := self._vectors[namespace].get(key)):
                        filtered.append((item, list(embeddings.values())))
                    else:
                        filtered.append((item, []))
        return filtered

    def _embed_search_queries(
        self,
        search_ops: dict[int, tuple[SearchOp, list[tuple[Item, list[list[float]]]]]],
    ) -> dict[str, list[float]]:
        queryinmem_store = {}
        if self.index_config and self.embeddings and search_ops:
            queries = {op.query for (op, _) in search_ops.values() if op.query}

            if queries:
                with cf.ThreadPoolExecutor() as executor:
                    futures = {
                        q: executor.submit(self.embeddings.embed_query, q)
                        for q in list(queries)
                    }
                    for query, future in futures.items():
                        queryinmem_store[query] = future.result()

        return queryinmem_store

    async def _aembed_search_queries(
        self,
        search_ops: dict[int, tuple[SearchOp, list[tuple[Item, list[list[float]]]]]],
    ) -> dict[str, list[float]]:
        queryinmem_store = {}
        if self.index_config and self.embeddings and search_ops:
            queries = {op.query for (op, _) in search_ops.values() if op.query}

            if queries:
                coros = [self.embeddings.aembed_query(q) for q in list(queries)]
                results = await asyncio.gather(*coros)
                queryinmem_store = dict(zip(queries, results))

        return queryinmem_store

    def _batch_search(
        self,
        ops: dict[int, tuple[SearchOp, list[tuple[Item, list[list[float]]]]]],
        queryinmem_store: dict[str, list[float]],
        results: list[Result],
    ) -> None:
        """Perform batch similarity search for multiple queries."""
        for i, (op, candidates) in ops.items():
            if not candidates:
                results[i] = []
                continue
            if op.query and queryinmem_store:
                query_embedding = queryinmem_store[op.query]
                flat_items, flat_vectors = [], []
                scoreless = []
                for item, vectors in candidates:
                    for vector in vectors:
                        flat_items.append(item)
                        flat_vectors.append(vector)
                    if not vectors:
                        scoreless.append(item)

                scores = _cosine_similarity(query_embedding, flat_vectors)
                sorted_results = sorted(
                    zip(scores, flat_items), key=lambda x: x[0], reverse=True
                )
                # max pooling
                seen: set[tuple[tuple[str, ...], str]] = set()
                kept: list[tuple[Optional[float], Item]] = []
                for score, item in sorted_results:
                    key = (item.namespace, item.key)
                    if key in seen:
                        continue
                    ix = len(seen)
                    seen.add(key)
                    if ix >= op.offset + op.limit:
                        break
                    if ix < op.offset:
                        continue

                    kept.append((score, item))
                if scoreless and len(kept) < op.limit:
                    # Corner case: if we request more items than what we have embedded,
                    # fill the rest with non-scored items
                    kept.extend(
                        (None, item) for item in scoreless[: op.limit - len(kept)]
                    )

                results[i] = [
                    SearchItem(
                        namespace=item.namespace,
                        key=item.key,
                        value=item.value,
                        created_at=item.created_at,
                        updated_at=item.updated_at,
                        score=float(score) if score is not None else None,
                    )
                    for score, item in kept
                ]
            else:
                results[i] = [
                    SearchItem(
                        namespace=item.namespace,
                        key=item.key,
                        value=item.value,
                        created_at=item.created_at,
                        updated_at=item.updated_at,
                    )
                    for (item, _) in candidates[op.offset : op.offset + op.limit]
                ]

    def _prepare_ops(
        self, ops: Iterable[Op]
    ) -> tuple[
        list[Result],
        dict[tuple[tuple[str, ...], str], PutOp],
        dict[int, tuple[SearchOp, list[tuple[Item, list[list[float]]]]]],
    ]:
        results: list[Result] = []
        put_ops: dict[tuple[tuple[str, ...], str], PutOp] = {}
        search_ops: dict[
            int, tuple[SearchOp, list[tuple[Item, list[list[float]]]]]
        ] = {}
        for i, op in enumerate(ops):
            if isinstance(op, GetOp):
                item = self._data[op.namespace].get(op.key)
                results.append(item)
            elif isinstance(op, SearchOp):
                search_ops[i] = (op, self._filter_items(op))
                results.append(None)
            elif isinstance(op, ListNamespacesOp):
                results.append(self._handle_list_namespaces(op))
            elif isinstance(op, PutOp):
                put_ops[(op.namespace, op.key)] = op
                results.append(None)
            else:
                raise ValueError(f"Unknown operation type: {type(op)}")

        return results, put_ops, search_ops

    def _apply_put_ops(self, put_ops: dict[tuple[tuple[str, ...], str], PutOp]) -> None:
        for (namespace, key), op in put_ops.items():
            if op.value is None:
                self._data[namespace].pop(key, None)
                self._vectors[namespace].pop(key, None)
            else:
                self._data[namespace][key] = Item(
                    value=op.value,
                    key=key,
                    namespace=namespace,
                    created_at=datetime.now(timezone.utc),
                    updated_at=datetime.now(timezone.utc),
                )

    def _extract_texts(
        self, put_ops: dict[tuple[tuple[str, ...], str], PutOp]
    ) -> dict[str, list[tuple[tuple[str, ...], str, str]]]:
        if put_ops and self.index_config and self.embeddings:
            to_embed = defaultdict(list)

            for op in put_ops.values():
                if op.value is not None and op.index is not False:
                    if op.index is None:
                        paths = self.index_config["__tokenized_fields"]
                    else:
                        paths = [(ix, tokenize_path(ix)) for ix in op.index]
                    for path, field in paths:
                        texts = get_text_at_path(op.value, field)
                        if texts:
                            if len(texts) > 1:
                                for i, text in enumerate(texts):
                                    to_embed[text].append(
                                        (op.namespace, op.key, f"{path}.{i}")
                                    )

                            else:
                                to_embed[texts[0]].append((op.namespace, op.key, path))

            return to_embed

        return {}

    def _insertinmem_store(
        self,
        to_embed: dict[str, list[tuple[tuple[str, ...], str, str]]],
        embeddings: list[list[float]],
    ) -> None:
        indices = [index for indices in to_embed.values() for index in indices]
        if len(indices) != len(embeddings):
            raise ValueError(
                f"Number of embeddings ({len(embeddings)}) does not"
                f" match number of indices ({len(indices)})"
            )
        for embedding, (ns, key, path) in zip(embeddings, indices):
            self._vectors[ns][key][path] = embedding

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


@functools.lru_cache(maxsize=1)
def _check_numpy() -> bool:
    if bool(util.find_spec("numpy")):
        return True
    logger.warning(
        "NumPy not found in the current Python environment. "
        "The InMemoryStore will use a pure Python implementation for vector operations, "
        "which may significantly impact performance, especially for large datasets or frequent searches. "
        "For optimal speed and efficiency, consider installing NumPy: "
        "pip install numpy"
    )
    return False


def _cosine_similarity(X: list[float], Y: list[list[float]]) -> list[float]:
    """
    Compute cosine similarity between a vector X and a matrix Y.
    Lazy import numpy for efficiency.
    """
    if not Y:
        return []
    if _check_numpy():
        import numpy as np  # type: ignore

        X_arr = np.array(X) if not isinstance(X, np.ndarray) else X
        Y_arr = np.array(Y) if not isinstance(Y, np.ndarray) else Y
        X_norm = np.linalg.norm(X_arr)
        Y_norm = np.linalg.norm(Y_arr, axis=1)

        # Avoid division by zero
        mask = Y_norm != 0
        similarities = np.zeros_like(Y_norm)
        similarities[mask] = np.dot(Y_arr[mask], X_arr) / (Y_norm[mask] * X_norm)
        return similarities.tolist()

    similarities = []
    for y in Y:
        dot_product = sum(a * b for a, b in zip(X, y))
        norm1 = sum(a * a for a in X) ** 0.5
        norm2 = sum(a * a for a in y) ** 0.5
        similarity = dot_product / (norm1 * norm2) if norm1 > 0 and norm2 > 0 else 0.0
        similarities.append(similarity)

    return similarities


def _does_match(match_condition: MatchCondition, key: tuple[str, ...]) -> bool:
    """Whether a namespace key matches a match condition."""
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


def _compare_values(item_value: Any, filter_value: Any) -> bool:
    """Compare values in a JSONB-like way, handling nested objects."""
    if isinstance(filter_value, dict):
        if any(k.startswith("$") for k in filter_value):
            return all(
                _apply_operator(item_value, op_key, op_value)
                for op_key, op_value in filter_value.items()
            )
        if not isinstance(item_value, dict):
            return False
        return all(
            _compare_values(item_value.get(k), v) for k, v in filter_value.items()
        )
    elif isinstance(filter_value, (list, tuple)):
        return (
            isinstance(item_value, (list, tuple))
            and len(item_value) == len(filter_value)
            and all(_compare_values(iv, fv) for iv, fv in zip(item_value, filter_value))
        )
    else:
        return item_value == filter_value


def _apply_operator(value: Any, operator: str, op_value: Any) -> bool:
    """Apply a comparison operator, matching PostgreSQL's JSONB behavior."""
    if operator == "$eq":
        return value == op_value
    elif operator == "$gt":
        return float(value) > float(op_value)
    elif operator == "$gte":
        return float(value) >= float(op_value)
    elif operator == "$lt":
        return float(value) < float(op_value)
    elif operator == "$lte":
        return float(value) <= float(op_value)
    elif operator == "$ne":
        return value != op_value
    else:
        raise ValueError(f"Unsupported operator: {operator}")
