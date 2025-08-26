from collections import defaultdict
from typing import Iterable

from langchain_core.embeddings import Embeddings
from typing_extensions import Literal

from langgraph.store.base import (
    IndexConfig,
    Op,
    ensure_embeddings,
    tokenize_path,
)


class MCPStoreIndexConfig(IndexConfig):
    embeddings: Embeddings | None
    field: list[str]
    __tokenized_fields: list[tuple[str, Literal["$"] | list[str]]]
    __estimated_num_vectors: int


class BaseMCPStore:
    def _ensure_index_config(
        self, index_config
    ) -> tuple[MCPStoreIndexConfig | None, Embeddings | None]:
        if not index_config:
            return None, None
        index_config = index_config.copy()
        tokenized: list[tuple[str, Literal["$"] | list[str]]] = []
        tot = 0
        fields = index_config.get("fields") or ["$"]
        if isinstance(fields, str):
            fields = [fields]
        if not isinstance(fields, list):
            raise ValueError(f"Text fields must be a list or a string. Got {fields}")
        for p in fields:
            if p == "$":
                tokenized.append((p, "$"))
                tot += 1
            else:
                toks = tokenize_path(p)
                tokenized.append((p, toks))
                tot += len(toks)
        index_config["__tokenized_fields"] = tokenized
        index_config["__estimated_num_vectors"] = tot
        return index_config, ensure_embeddings(index_config.get("embed"))

    def _group_ops(
        self, ops: Iterable[Op]
    ) -> tuple[dict[type, list[tuple[int, Op]]], int]:
        grouped_ops: dict[type, list[tuple[int, Op]]] = defaultdict(list)
        tot = 0
        for idx, op in enumerate(ops):
            grouped_ops[type(op)].append((idx, op))
            tot += 1
        return grouped_ops, tot
