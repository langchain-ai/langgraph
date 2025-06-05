"""
Memgraph-based Store (sync) for LangGraph, enabling knowledge storage, vector search, and subgraph queries.
"""
import json
import threading
from contextlib import contextmanager
from typing import Any, Callable, Iterator, List, Optional, Tuple, Union

from langchain_core.embeddings import Embeddings
from neo4j import Driver
from langgraph.store.base import (
    BaseStore,
    Item,
    SearchItem,
    TTLConfig,
    ensure_embeddings,
    get_text_at_path,
)
from langgraph.store.base import (
    GetOp,
    PutOp,
    SearchOp,
    ListNamespacesOp,
    Op,
    Result,
)
from langgraph.store.memgraph.memgraph_util import (
    MemgraphIndexConfig,
    init_memgraph_index,
    place_vector_index_call,
)
from langgraph.checkpoint.memgraph._internal import Conn, MemgraphConn, get_session


class MemgraphStore(BaseStore):
    """
    Synchronous Memgraph store for arbitrary key-value data, with optional vector embeddings for semantic search.
    Each item is stored as a node labeled `Memory` with properties:
      - namespace (string)
      - key (string)
      - value (string) -> JSON-serialized
      - embedding (list of float) -> optional

    The user can specify an index config with:
      index={
        "dims": 1536,
        "embed": <Embeddings or callable>,
        "fields": ["$"]
      }

    or omit to skip semantic search.
    """

    supports_ttl: bool = False  # Not implemented here

    def __init__(
        self,
        conn: Conn,
        index: Optional[MemgraphIndexConfig] = None,
        deserializer: Optional[Callable[[str], Any]] = None,
        ttl: Optional[TTLConfig] = None,
    ) -> None:
        super().__init__()
        self.conn = conn
        self.index_config = index
        self._deserializer = deserializer or self._default_deserializer
        self._lock = threading.Lock()
        self.embeddings: Optional[Embeddings] = None
        self._setup_done = False

        if self.index_config is not None:
            self.embeddings, self.index_config = init_memgraph_index(self.index_config)

    @classmethod
    @contextmanager
    def from_conn_string(
        cls,
        uri: str,
        user: Optional[str] = None,
        password: Optional[str] = None,
        index: Optional[MemgraphIndexConfig] = None,
    ):
        """
        Example:
            with MemgraphStore.from_conn_string("bolt://localhost:7687", index={"dims":768, "embed": embedder}) as store:
                store.setup()
                ...
        """
        mgconn = MemgraphConn(uri, user, password)
        try:
            yield cls(mgconn, index=index)
        finally:
            mgconn.close()

    def close(self):
        if isinstance(self.conn, MemgraphConn):
            self.conn.close()
        elif isinstance(self.conn, Driver):
            self.conn.close()

    def setup(self) -> None:
        if self._setup_done:
            return
        if self.index_config is not None:
            # Create or ensure vector index
            with get_session(self.conn) as session:
                session.run(place_vector_index_call(self.index_config))
        self._setup_done = True

    def batch(self, ops: Iterator[Op]) -> List[Result]:
        grouped, total = self._group_ops(ops)
        results: List[Result] = [None] * total

        with self._cursor() as session:
            if GetOp in grouped:
                self._batch_get_ops(grouped[GetOp], results, session)
            if PutOp in grouped:
                self._batch_put_ops(grouped[PutOp], results, session)
            if SearchOp in grouped:
                self._batch_search_ops(grouped[SearchOp], results, session)
            if ListNamespacesOp in grouped:
                self._batch_list_namespaces_ops(grouped[ListNamespacesOp], results, session)

        return results

    @contextmanager
    def _cursor(self):
        with get_session(self.conn) as session:
            yield session

    #
    # Batched GET
    #
    def _batch_get_ops(self, ops: List[Tuple[int, GetOp]], results: List[Result], session) -> None:
        # We'll collect them by namespace
        by_ns = {}
        for idx, op in ops:
            by_ns.setdefault(op.namespace, []).append((idx, op))
        for namespace, items in by_ns.items():
            ns_str = self._ns_to_str(namespace)
            keys = [iop.key for _, iop in items]
            query = f"""
            MATCH (m:Memory {{namespace:$ns}}) WHERE m.key IN $keys
            RETURN m.key AS key, m.value AS val, m.embedding as embedding
            """
            recs = session.run(query, ns=ns_str, keys=keys)
            found = {}
            for row in recs:
                found[row["key"]] = row
            for (batch_idx, iop) in items:
                row = found.get(iop.key)
                if row:
                    val = row["val"]
                    if isinstance(val, str):
                        val = self._deserializer(val)
                    results[batch_idx] = Item(
                        key=iop.key, namespace=namespace, value=val
                    )
                else:
                    results[batch_idx] = None

    #
    # Batched PUT
    #
    def _batch_put_ops(self, ops: List[Tuple[int, PutOp]], results: List[Result], session) -> None:
        # For each PutOp, if value is None => delete, else upsert
        for (ridx, op) in ops:
            if op.value is None:
                # DELETE
                ns_str = self._ns_to_str(op.namespace)
                query = """
                MATCH (m:Memory {namespace:$ns, key:$k})
                DETACH DELETE m
                """
                session.run(query, ns=ns_str, k=op.key)
                # results[ridx] = True or None
                continue

            # Upsert
            ns_str = self._ns_to_str(op.namespace)
            val_str = json.dumps(op.value) if not isinstance(op.value, str) else op.value
            embed_vector = None
            if (op.index != False) and self.index_config and self.embeddings:
                # embed
                text_to_embed = self._gather_text_for_embedding(op)
                vectors = self.embeddings.embed_documents([text_to_embed])
                if vectors:
                    embed_vector = vectors[0]
            query = f"""
            MERGE (m:Memory {{namespace:$ns, key:$k}})
            SET m.value = $val
            """
            params = dict(ns=ns_str, k=op.key, val=val_str)
            if embed_vector:
                query += ", m.embedding = $vec"
                params["vec"] = embed_vector
            else:
                query += ", m.embedding = NULL"
            session.run(query, **params)
            # results[ridx] = True or maybe an Item

    #
    # Batched SEARCH
    #
    def _batch_search_ops(self, ops: List[Tuple[int, SearchOp]], results: List[Result], session) -> None:
        # For each search op, if we have embedding we do vector search, else basic substring
        for (ridx, op) in ops:
            ns_str = self._ns_to_str(op.namespace_prefix) if op.namespace_prefix else ""
            if op.query and self.index_config and self.embeddings:
                # vector search
                qvec = self.embeddings.embed_documents([op.query])[0]
                # We'll do up to limit * 2 if there's a filter
                expanded_limit = op.limit * 2
                # Memgraph vector search: e.g. CALL vector_search.search("my_index", $topk, $vec) YIELD node, distance
                # then filter node.namespace STARTS WITH ...
                q = place_vector_index_call(self.index_config, for_search=True)
                # We'll assume the index name is something like "memory_embeddings"
                # Then do a YIELD node, distance
                # We'll do the final filter in the same query

                cypher = f"""
                CALL vector_search.search("{self.index_config["index_name"]}", $k, $qvec) YIELD node, distance
                WHERE node.namespace STARTS WITH $ns
                RETURN node, distance
                ORDER BY distance ASC
                LIMIT $lim
                """
                # filter or scoring
                params = dict(k=expanded_limit, qvec=qvec, ns=ns_str, lim=op.limit)
                try:
                    recs = session.run(cypher, **params)
                    items = []
                    for row in recs:
                        node = row["node"]
                        distance = row["distance"]
                        score = 1/(1+distance)  # transform distance -> score
                        val = node["value"]
                        if isinstance(val, str):
                            val = self._deserializer(val)
                        key = node["key"]
                        ns_raw = node["namespace"]
                        items.append(SearchItem(value=val, key=key, namespace=self._str_to_ns(ns_raw), score=score))
                    results[ridx] = items
                except Exception as e:
                    results[ridx] = []
            else:
                # fallback substring search
                # We'll find items where namespace starts with ns_str and value CONTAINS op.query
                query = f"""
                MATCH (m:Memory)
                WHERE m.namespace STARTS WITH $ns
                AND m.value CONTAINS $q
                RETURN m.key AS key, m.value AS val
                LIMIT $lim
                """
                params = dict(ns=ns_str, q=op.query or "", lim=op.limit)
                recs = session.run(query, **params)
                items = []
                for row in recs:
                    val = row["val"]
                    if isinstance(val, str):
                        val = self._deserializer(val)
                    items.append(SearchItem(value=val, key=row["key"], namespace=self._str_to_ns(ns_str)))
                results[ridx] = items

    #
    # Batched LIST NAMESPACES
    #
    def _batch_list_namespaces_ops(self, ops: List[Tuple[int, ListNamespacesOp]], results: List[Result], session) -> None:
        # We'll just get distinct m.namespace and optionally filter
        for (ridx, op) in ops:
            q = """
            MATCH (m:Memory)
            RETURN DISTINCT m.namespace as ns
            """
            recs = session.run(q)
            all_ns = []
            for row in recs:
                ns_str = row["ns"]
                if op.match_conditions:
                    # apply conditions
                    pass
                all_ns.append(self._str_to_ns(ns_str))
            results[ridx] = all_ns

    #
    # Helpers
    #
    def _gather_text_for_embedding(self, op: PutOp) -> str:
        # if op.index is a list, gather those paths from op.value
        if isinstance(op.index, list):
            texts = []
            for path_str in op.index:
                parts = get_text_at_path(op.value, path_str.split("."))
                for p in parts:
                    texts.append(p if isinstance(p, str) else json.dumps(p))
            if not texts:
                return json.dumps(op.value)
            return " ".join(texts)
        else:
            # default gather everything
            return json.dumps(op.value)

    def _default_deserializer(self, val_str: str) -> Any:
        return json.loads(val_str)

    def _ns_to_str(self, ns: tuple) -> str:
        return ".".join(ns)

    def _str_to_ns(self, ns_str: str) -> tuple:
        return tuple(ns_str.split("."))
