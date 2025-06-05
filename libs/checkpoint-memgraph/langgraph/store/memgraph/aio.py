"""
Async MemgraphStore for LangGraph.
Implements an asynchronous version of the store using the AsyncDriver.
"""
import asyncio
import json
from contextlib import asynccontextmanager
from typing import Any, List, Optional, Tuple, Callable, Dict

from neo4j import AsyncDriver
from langchain_core.embeddings import Embeddings
from langgraph.store.base.batch import AsyncBatchedBaseStore
from langgraph.store.base import (
    Item,
    SearchItem,
    GetOp,
    PutOp,
    SearchOp,
    ListNamespacesOp,
    Op,
    Result,
    get_text_at_path
)
from langgraph.store.memgraph.memgraph_util import (
    MemgraphIndexConfig,
    init_memgraph_index,
    place_vector_index_call
)
from langgraph.checkpoint.memgraph._ainternal import Conn, AsyncMemgraphConn, aget_session


class AsyncMemgraphStore(AsyncBatchedBaseStore):
    """Async Memgraph-based store with optional vector index for semantic search."""

    supports_ttl = False

    def __init__(
        self,
        conn: Conn,
        index: Optional[MemgraphIndexConfig] = None,
        deserializer: Optional[Callable[[str], Any]] = None,
    ):
        super().__init__()
        self.conn = conn
        self.index_config = index
        self._deserializer = deserializer or self._default_deserializer
        self.embeddings: Optional[Embeddings] = None
        self._setup_done = False
        self._lock = asyncio.Lock()

        if self.index_config:
            self.embeddings, self.index_config = init_memgraph_index(self.index_config)

    @classmethod
    @asynccontextmanager
    async def from_conn_string(
        cls,
        uri: str,
        user: Optional[str] = None,
        password: Optional[str] = None,
        index: Optional[MemgraphIndexConfig] = None,
    ):
        """
        Usage:
            async with AsyncMemgraphStore.from_conn_string("bolt://localhost:7687") as store:
                await store.setup()
                ...
        """
        mgconn = AsyncMemgraphConn(uri, user, password)
        try:
            yield cls(mgconn, index=index)
        finally:
            await mgconn.close()

    async def setup(self) -> None:
        if self._setup_done:
            return
        if self.index_config:
            async with aget_session(self.conn) as session:
                stmt = place_vector_index_call(self.index_config)
                if stmt:
                    try:
                        await session.run(stmt)
                    except:
                        pass
        self._setup_done = True

    async def abatch(self, ops) -> List[Result]:
        grouped, total = self._group_ops(ops)
        results: List[Result] = [None] * total
        async with aget_session(self.conn) as session:
            if GetOp in grouped:
                await self._batch_get_ops(grouped[GetOp], results, session)
            if PutOp in grouped:
                await self._batch_put_ops(grouped[PutOp], results, session)
            if SearchOp in grouped:
                await self._batch_search_ops(grouped[SearchOp], results, session)
            if ListNamespacesOp in grouped:
                await self._batch_list_namespaces_ops(grouped[ListNamespacesOp], results, session)
        return results

    async def _batch_get_ops(self, ops, results, session) -> None:
        by_ns = {}
        for idx, op in ops:
            by_ns.setdefault(op.namespace, []).append((idx, op))
        for namespace, items in by_ns.items():
            ns_str = self._ns_to_str(namespace)
            keys = [iop.key for _, iop in items]
            query = """
            MATCH (m:Memory {namespace:$ns})
            WHERE m.key IN $keys
            RETURN m.key AS key, m.value AS val, m.embedding AS embedding
            """
            recs = await session.run(query, ns=ns_str, keys=keys)
            rows = await recs.data()
            found = {r["key"]: r for r in rows}
            for (batch_idx, iop) in items:
                row = found.get(iop.key)
                if row:
                    val = row["val"]
                    if isinstance(val, str):
                        val = self._deserializer(val)
                    results[batch_idx] = Item(key=iop.key, namespace=namespace, value=val)
                else:
                    results[batch_idx] = None

    async def _batch_put_ops(self, ops, results, session) -> None:
        for (ridx, op) in ops:
            if op.value is None:
                # delete
                query = """
                MATCH (m:Memory {namespace:$ns, key:$k})
                DETACH DELETE m
                """
                await session.run(query, ns=self._ns_to_str(op.namespace), k=op.key)
                continue

            ns_str = self._ns_to_str(op.namespace)
            val_str = json.dumps(op.value) if not isinstance(op.value, str) else op.value
            embed_vector = None
            if (op.index != False) and self.index_config and self.embeddings:
                text = await self._gather_text_for_embedding_async(op)
                vectors = await self.embeddings.aembed_documents([text])
                if vectors:
                    embed_vector = vectors[0]
            query = """
            MERGE (m:Memory {namespace:$ns, key:$k})
            SET m.value = $val
            """
            params = {"ns": ns_str, "k": op.key, "val": val_str}
            if embed_vector:
                query += ", m.embedding = $vec"
                params["vec"] = embed_vector
            else:
                query += ", m.embedding = NULL"
            await session.run(query, **params)

    async def _batch_search_ops(self, ops, results, session) -> None:
        for (ridx, op) in ops:
            ns_str = self._ns_to_str(op.namespace_prefix) if op.namespace_prefix else ""
            if op.query and self.index_config and self.embeddings:
                qvec = (await self.embeddings.aembed_documents([op.query]))[0]
                expanded_limit = op.limit * 2
                cypher = f"""
                CALL vector_search.search("{self.index_config["index_name"]}", $k, $qvec) YIELD node, distance
                WHERE node.namespace STARTS WITH $ns
                RETURN node, distance
                ORDER BY distance ASC
                LIMIT $limit
                """
                params = {"k": expanded_limit, "qvec": qvec, "ns": ns_str, "limit": op.limit}
                try:
                    recs = await session.run(cypher, **params)
                    rows = await recs.data()
                    items = []
                    for row in rows:
                        node = row["node"]
                        distance = row["distance"]
                        score = 1/(1+distance)
                        val = node["value"]
                        if isinstance(val, str):
                            val = self._deserializer(val)
                        key = node["key"]
                        items.append(SearchItem(value=val, key=key, namespace=self._str_to_ns(node["namespace"]), score=score))
                    results[ridx] = items
                except:
                    results[ridx] = []
            else:
                # substring fallback
                query = """
                MATCH (m:Memory)
                WHERE m.namespace STARTS WITH $ns
                AND m.value CONTAINS $q
                RETURN m.key AS key, m.value AS val
                LIMIT $lim
                """
                params = dict(ns=ns_str, q=op.query or "", lim=op.limit)
                recs = await session.run(query, **params)
                rows = await recs.data()
                items = []
                for row in rows:
                    val = row["val"]
                    if isinstance(val, str):
                        val = self._deserializer(val)
                    items.append(SearchItem(value=val, key=row["key"], namespace=self._str_to_ns(ns_str)))
                results[ridx] = items

    async def _batch_list_namespaces_ops(self, ops, results, session) -> None:
        for (ridx, op) in ops:
            # We just get distinct m.namespace
            query = """
            MATCH (m:Memory)
            RETURN DISTINCT m.namespace as ns
            """
            recs = await session.run(query)
            rows = await recs.data()
            all_ns = []
            for row in rows:
                ns_str = row["ns"]
                all_ns.append(self._str_to_ns(ns_str))
            results[ridx] = all_ns

    async def _gather_text_for_embedding_async(self, op: PutOp) -> str:
        if isinstance(op.index, list):
            texts = []
            for path_str in op.index:
                parts = get_text_at_path(op.value, path_str.split("."))
                for p in parts:
                    texts.append(p if isinstance(p, str) else json.dumps(p))
            if not texts:
                return json.dumps(op.value)
            return " ".join(texts)
        return json.dumps(op.value)

    def _default_deserializer(self, val_str: str) -> Any:
        return json.loads(val_str)

    def _ns_to_str(self, ns: tuple) -> str:
        return ".".join(ns)

    def _str_to_ns(self, ns_str: str) -> tuple:
        return tuple(ns_str.split("."))
