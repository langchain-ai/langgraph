"""
Utility classes and functions for Memgraph store indexing, vector creation, etc.
"""
from typing import Any, Dict, Optional, Tuple, Union

from langchain_core.embeddings import Embeddings
from langgraph.store.base import ensure_embeddings

MemgraphIndexConfig = Dict[str, Any]
# Example:
# {
#   "dims": 1536,
#   "embed": <Embeddings object or callable>,
#   "index_name": "mem_idx",
#   "fields": ["$"]
# }


def init_memgraph_index(index_conf: MemgraphIndexConfig) -> Tuple[Optional[Embeddings], MemgraphIndexConfig]:
    """
    Initialize the MemgraphIndexConfig by setting up the embeddings, index name, etc.
    """
    index_conf = index_conf.copy()
    embeddings = ensure_embeddings(index_conf.get("embed"))
    if "index_name" not in index_conf:
        # default
        index_conf["index_name"] = "memory_embeddings"
    return embeddings, index_conf


def place_vector_index_call(index_conf: MemgraphIndexConfig, for_search: bool=False) -> str:
    """
    Return the Cypher call to create or search Memgraph's vector index.
    Memgraph uses: CREATE VECTOR INDEX <name> ON :Node(property) WITH {dimension:..., distance:..., ...}

    for_search: if True, doesn't produce create index statement (that's for actual search usage).
    """
    if for_search:
        # This is not a creation statement, it's a usage snippet. Typically "CALL vector_search.search()"
        return ""
    name = index_conf["index_name"]
    dims = index_conf["dims"]
    distance = index_conf.get("distance", "cosine")  # or 'l2'
    # Memgraph docs: CREATE VECTOR INDEX <index_name> ON :Memory(embedding) WITH {dimension: 1536}
    # distance can't be explicitly stated yet, but they might allow 'WITH {metric: "cosine"}'
    return f'CREATE VECTOR INDEX IF NOT EXISTS {name} ON :Memory(embedding) WITH {{dimension: {dims}}}'
