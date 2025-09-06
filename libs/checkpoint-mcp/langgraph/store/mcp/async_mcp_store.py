import asyncio
import json
import logging
from contextlib import asynccontextmanager
from datetime import datetime
from typing import Any, Iterable, Sequence

from langgraph.store.base import (
    GetOp,
    Item,
    ListNamespacesOp,
    Op,
    PutOp,
    Result,
    SearchItem,
    SearchOp,
)
from langgraph.store.base.batch import AsyncBatchedBaseStore
from langgraph.store.base.embed import (
    get_text_at_path,
    tokenize_path,
)
from langgraph.store.mcp.base import BaseMCPStore, MCPStoreIndexConfig

from .async_mcp_store_client import AsyncMCPStoreClient

logger = logging.getLogger(__name__)


class AsyncMCPStore(AsyncBatchedBaseStore, BaseMCPStore):
    def __init__(
        self,
        client: AsyncMCPStoreClient,
        index_config: MCPStoreIndexConfig | None = None,
    ) -> None:
        super().__init__()
        self.client = client
        self.index_config = index_config
        if self.index_config:
            self.index_config, self.embeddings = self._ensure_index_config(
                self.index_config
            )
        else:
            self.embeddings = None
        # TODO: Add pipeline assignment

    @classmethod
    @asynccontextmanager
    async def from_mcp_config(
        cls, host="localhost", port=8000, index_config=None, **kwargs
    ):
        try:
            client = AsyncMCPStoreClient(host=host, port=port, **kwargs)
            if client is None:
                raise RuntimeError("Failed to create MCPStoreClient")
            await client.__aenter__()
            store = cls(client=client, index_config=index_config)
            yield store
        except Exception as e:
            logger.error(f"Error occurred while creating MCP store: {e}")

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if hasattr(self.client, "__aexit__"):
            await self.client.__aexit__(exc_type, exc_val, exc_tb)

    def _process_embeddings(
        self, value: dict[str, Any], index
    ) -> dict[str, Any] | None:
        if not self.embeddings or not self.index_config:
            return value

        if index is None:
            if "__tokenized_fields" in self.index_config:
                paths = self.index_config["__tokenized_fields"]
            else:
                return value
        else:
            # Use specified fields
            if isinstance(index, (list, tuple)):
                paths = [(field, tokenize_path(field)) for field in index]
            else:
                return value

        embeddings_data: dict[str, Sequence[float]] = {}
        texts_to_embed: list[str] = []
        text_metadata: list[str] = []

        for path, tokenized_path in paths:
            texts = get_text_at_path(value, tokenized_path)
            for i, text in enumerate(texts):
                pathname = f"{path}.{i}" if len(texts) > 1 else path
                texts_to_embed.append(text)
                text_metadata.append(pathname)

        # Generate embeddings for all texts at once
        if texts_to_embed:
            vectors = self.embeddings.embed_documents(texts_to_embed)
            for pathname, vector in zip(text_metadata, vectors):
                embeddings_data[pathname] = vector

        enhanced_value = {
            "original_data": value,
            "embeddings": embeddings_data,
            "_mcp_store_metadata": {
                "has_embeddings": bool(embeddings_data),
                "embedding_fields": list(embeddings_data.keys()),
                "created_at": datetime.now().isoformat(),
            },
        }

        return enhanced_value

    async def abatch(self, ops: Iterable[Op]) -> list[Result]:
        grouped_ops, num_ops = self._group_ops(ops)
        results: list[Result] = [None] * num_ops
        await self._execute_batch(grouped_ops, results)
        return results

    async def _handle_put_ops(
        self, op_tuples: list[tuple[int, PutOp]], results: list[Result]
    ) -> None:
        try:
            tasks = []
            for index, op in op_tuples:
                if op.value is None:
                    tasks.append((index, self.client.adelete(op.namespace, op.key)))
                else:
                    enhanced_value = self._process_embeddings(op.value, op.index)
                    tasks.append(
                        (index, self.client.aput(op.namespace, op.key, enhanced_value))
                    )

            responses = await asyncio.gather(
                *[task[1] for task in tasks], return_exceptions=True
            )
            for i, resp in enumerate(responses):
                index = tasks[i][0]
                if isinstance(resp, Exception) or isinstance(resp, BaseException):
                    logger.error(f"Put operation failed: {resp}")
                    raise resp
                else:
                    results[index] = None
        except Exception as e:
            logger.error(f"Error processing put responses: {e}")

    async def _handle_get_ops(
        self, ops_tuples: list[tuple[int, GetOp]], results: list[Result]
    ) -> None:
        try:
            tasks = []
            for index, op in ops_tuples:
                tasks.append((index, self.client.aget(op.namespace, op.key)))

            responses = await asyncio.gather(
                *[task[1] for task in tasks], return_exceptions=True
            )
            get_items: list[Item] = []
            for i, resp in enumerate(responses):
                index = tasks[i][0]
                if isinstance(resp, Exception) or isinstance(resp, BaseException):
                    logger.error(f"Get operation failed: {resp}")
                    raise resp
                else:
                    if resp is None:
                        continue
                    resp_json = json.loads(resp)
                    value = resp_json.get("value")
                    original_data: dict | None = None
                    if value and isinstance(value, dict) and "original_data" in value:
                        original_data = value.get("original_data")
                    else:
                        original_data = value
                    get_item = Item(
                        namespace=tuple(resp_json.get("namespace", "").split("."))
                        if resp_json.get("namespace")
                        else (),
                        key=resp_json.get("key"),
                        value=original_data,
                        created_at=resp_json.get("created_at"),
                        updated_at=resp_json.get("updated_at"),
                    )
                    get_items.append(get_item)
                results[index] = get_items
        except Exception as e:
            logger.error(f"Error processing get responses: {e}")

    async def _handle_search_ops(
        self, ops_tuples: list[tuple[int, SearchOp]], results: list[Result]
    ) -> None:
        try:
            tasks = []
            for index, op in ops_tuples:
                tasks.append(
                    (index, self.client.asearch(op.namespace_prefix, op.query))
                )

            responses = await asyncio.gather(
                *[task[1] for task in tasks], return_exceptions=True
            )
            search_items: list[SearchItem] = []
            for i, resp in enumerate(responses):
                index = tasks[i][0]
                if isinstance(resp, Exception) or isinstance(resp, BaseException):
                    logger.error(f"Search operation failed: {resp}")
                    raise resp
                else:
                    # We get a list of json string for each responses
                    for item in resp:
                        resp_json = json.loads(item)
                        value = resp_json.get("value")
                        original_data: dict | None = None
                        if (
                            value
                            and isinstance(value, dict)
                            and "original_data" in value
                        ):
                            original_data = value.get("original_data")
                        else:
                            original_data = value
                        search_item = SearchItem(
                            namespace=tuple(resp_json.get("namespace", "").split("."))
                            if resp_json.get("namespace")
                            else (),
                            key=resp_json.get("key"),
                            value=original_data,
                            created_at=resp_json.get("created_at"),
                            updated_at=resp_json.get("updated_at"),
                            score=resp_json.get("score"),
                        )
                        search_items.append(search_item)
            results[index] = search_items
        except Exception as e:
            logger.error(f"Error processing search responses: {e}")

    async def _handle_list_namespaces_ops(
        self, ops_tuples: list[tuple[int, ListNamespacesOp]], results: list[Result]
    ) -> None:
        try:
            tasks = []
            for index, op in ops_tuples:
                tasks.append((index, self.client.alist_namespaces()))

            responses = await asyncio.gather(
                *[task[1] for task in tasks], return_exceptions=True
            )
            for i, resp in enumerate(responses):
                index = tasks[i][0]
                if isinstance(resp, Exception) or isinstance(resp, BaseException):
                    logger.error(f"List namespaces operation failed: {resp}")
                    results[index] = []
                else:
                    # Convert response to list of tuples, handling potential exceptions
                    namespace_tuples = []
                    if resp:
                        for ns in resp:
                            try:
                                if isinstance(ns, (list, tuple)):
                                    namespace_tuples.append(
                                        tuple(str(part) for part in ns)
                                    )
                                elif isinstance(ns, str):
                                    namespace_tuples.append(tuple(ns.split(".")))
                                else:
                                    namespace_tuples.append((str(ns),))
                            except Exception as e:
                                logger.warning(f"Failed to process namespace {ns}: {e}")
                    results[index] = namespace_tuples
        except Exception as e:
            logger.error(f"Error processing list namespaces responses: {e}")

    async def _execute_batch(self, grouped_ops: dict, results: list[Result]) -> None:
        for op_type, ops_tuples in grouped_ops.items():
            if op_type == PutOp:
                await self._handle_put_ops(ops_tuples, results)
            elif op_type == GetOp:
                await self._handle_get_ops(ops_tuples, results)
            elif op_type == SearchOp:
                await self._handle_search_ops(ops_tuples, results)
            elif op_type == ListNamespacesOp:
                await self._handle_list_namespaces_ops(ops_tuples, results)
