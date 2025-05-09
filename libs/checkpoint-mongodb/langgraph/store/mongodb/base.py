import logging
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import datetime, timezone
from importlib.metadata import version
from typing import Any, Iterable, Literal, Optional, TypeVar, Union

from bson import SON
from pymongo import (
    DeleteOne,
    MongoClient,
    UpdateOne,
)
from pymongo.collection import Collection, ReturnDocument
from pymongo.driver_info import DriverInfo

from langchain_core.runnables import run_in_executor
from langgraph.store.base import (
    NOT_PROVIDED,
    BaseStore,
    Item,
    GetOp,
    ListNamespacesOp,
    NamespacePath,
    Op,
    PutOp,
    Result,
    SearchItem,
    SearchOp,
    TTLConfig,
)

K = TypeVar("K")
V = TypeVar("V")

logger = logging.getLogger(__name__)


class MongoDBStore(BaseStore):
    """MongoDB's persistent key-value stores for long-term memory.

    Stores enable persistence and memory that can be shared across threads,
    scoped to user IDs, assistant IDs, or other arbitrary namespaces.

    Supports semantic search capabilities through
    an optional `index` configuration.
    """

    supports_ttl: True
    """TTL is supported by a TTL index of field: updated_at."""

    def __init__(
        self,
        collection: Collection,
        ttl_config: Optional[TTLConfig] = None,
        **kwargs: Any,
    ):
        """Construct store and its indexes.

        Args:
            collection: Collection of Items backing the store.
            ttl_config: Optionally define a TTL and whether to update on reads(get/search).

        Returns:
            Instance of MongoDBStore.
        """
        self.collection = collection
        self.ttl_config = ttl_config
        # Create indexes if not present
        # Create a unique index, akin to primary key, on namespace + key
        idx_keys = [idx["key"] for idx in self.collection.list_indexes()]
        if SON([("namespace", 1), ("key", 1)]) not in idx_keys:
            self.collection.create_index(keys=["namespace", "key"], unique=True)

        # Optionally, expire values using [TTL Index](https://www.mongodb.com/docs/manual/core/index-ttl/)
        if self.ttl_config is not None and SON([("updated_at", 1)]) not in idx_keys:
            self.ttl = float(self.ttl_config["default_ttl"])
            self.collection.create_index("updated_at", expireAfterSeconds=self.ttl)

    def put(
        self,
        namespace: tuple[str, ...],
        key: str,
        value: dict[str, Any],
        index: Optional[Union[Literal[False], list[str]]] = None,
        *,
        ttl: Union[Optional[float], "NotProvided"] = NOT_PROVIDED,
    ) -> None:
        """
        _id may be a problem namespace+key is unique. put will insert or update, but _id will always be unique

        """

        if ttl:
            logger.warning(
                "ttl argument ignored. MongoDBStore TTL behavior is performed via a TTL Index."
            )

        if index:
            raise NotImplementedError()  # TODO

        op = UpdateOne(
            filter={"namespace": list(namespace), "key": key},
            update={
                "$set": {"value": value, "updated_at": datetime.now(tz=timezone.utc)},
                "$setOnInsert": {
                    "created_at": datetime.now(tz=timezone.utc),
                },
            },
            upsert=True,
        )
        self.collection.bulk_write([op])

    def get(
        self,
        namespace: tuple[str, ...],
        key: str,
        *,
        refresh_ttl: Optional[bool] = None,
    ) -> Optional[Item]:
        """Retrieve a single item.

        Args:
            namespace: Hierarchical path for the item.
            key: Unique identifier within the namespace.
            refresh_ttl: Whether to refresh TTLs for the returned item.
                If None (default), uses the store's default refresh_ttl setting.
                If no TTL is specified, this argument is ignored.

        Returns:
            The retrieved item or None if not found.
        """
        if refresh_ttl is False or (
            self.ttl_config and not self.ttl_config["refresh_on_read"]
        ):
            res = self.collection.find_one(
                filter={"namespace": namespace, "key": key},
            )
        else:
            res = self.collection.find_one_and_update(
                filter={"namespace": namespace, "key": key},
                update={"$set": {"updated_at": datetime.now(tz=timezone.utc)}},
                return_document=ReturnDocument.AFTER,
            )
        if res:
            return Item(
                value=res["value"],
                key=res["key"],
                namespace=tuple(res["namespace"]),
                created_at=res["created_at"],
                updated_at=res["updated_at"],
            )

    def delete(self, namespace: tuple[str, ...], key: str) -> None:
        """Delete an item.

        Args:
            namespace: Hierarchical path for the item.
            key: Unique identifier within the namespace.
        """
        self.collection.delete_one({"namespace": list(namespace), "key": key})

    @staticmethod
    def _match_prefix(prefix: NamespacePath):
        """Helper for list_namespaces."""
        if not prefix or prefix == "*":
            return {}
        if "*" not in prefix:
            return {"$eq": [{"$slice": ["$namespace", len(prefix)]}, list(prefix)]}
        matches = []
        for i, p in enumerate(prefix):
            if p != "*":
                matches.append({"$eq": [{"$arrayElemAt": ["$namespace", i]}, p]})
        return {"$and": matches}

    @staticmethod
    def _match_suffix(suffix: NamespacePath):
        """Helper for list_namespaces."""
        if not suffix or suffix == "*":
            return {}
        if "*" not in suffix:
            return {"$eq": [{"$slice": ["$namespace", -1 * len(suffix)]}, list(suffix)]}
        matches = []
        for i, p in enumerate(suffix):
            if p != "*":
                matches.append(
                    {
                        "$eq": [
                            {
                                "$arrayElemAt": [
                                    "$namespace",
                                    {"$subtract": [{"$size": "$namespace"}, i]},
                                ]
                            },
                            p,
                        ]
                    }
                )
        return {"$and": matches}

    def list_namespaces(
        self,
        *,
        prefix: Optional[NamespacePath] = None,
        suffix: Optional[NamespacePath] = None,
        max_depth: Optional[int] = None,
        limit: int = 100,
        offset: int = 0,
    ) -> list[tuple[str, ...]]:
        """List and filter namespaces in the store."""
        pipeline = []
        expr = {}
        if prefix:
            pcond = self._match_prefix(prefix)
            expr = {"$expr": pcond}
        if suffix:
            scond = self._match_suffix(suffix)
            expr = {"$expr": scond}
        if prefix and suffix:
            expr = {"$expr": {"$and": [pcond, scond]}}

        pipeline.append({"$match": expr})

        if max_depth:
            pipeline.append(
                {
                    "$project": {
                        "namespace": {"$slice": ["$namespace", max_depth]},
                        "_id": 0,
                    }
                }
            )
        else:
            pipeline.append({"$project": {"namespace": 1, "_id": 0}})

        if limit:
            pipeline.append({"$limit": limit})
        # Deduplicate
        pipeline.extend(
            [
                {"$group": {"_id": "$namespace"}},
                {"$project": {"_id": 0, "namespace": "$_id"}},
            ]
        )

        if offset:
            raise NotImplementedError("offset is not implemented")

        results = self.collection.aggregate(pipeline)
        return [tuple(res["namespace"]) for res in results]

    def batch(self, ops: Iterable[Op]) -> list[Result]:
        """Execute multiple operations synchronously in a single batch.

        Args:
            ops: An iterable of operations to execute.

        Returns:
            A list of results, where each result corresponds to an operation in the input.
            The order of results matches the order of input operations.
            The length of output may not match the input as PutOp returns None.
        """
        results = []
        curr_batch = []
        for op in ops:
            if isinstance(op, PutOp):
                if op.value is None:
                    # mark the item for deletion.
                    curr_batch.append(
                        DeleteOne(
                            filter={"namespace": list(op.namespace), "key": op.key}
                        )
                    )
                else:
                    # Add or Upsert the value
                    curr_batch.append(
                        UpdateOne(
                            filter={"namespace": list(op.namespace), "key": op.key},
                            update={
                                "$set": {
                                    "value": op.value,
                                    "updated_at": datetime.now(tz=timezone.utc),
                                },
                                "$setOnInsert": {
                                    "created_at": datetime.now(tz=timezone.utc),
                                },
                            },
                            upsert=True,
                        )
                    )
            elif isinstance(op, GetOp):
                if curr_batch:
                    self.collection.bulk_write(curr_batch)
                    curr_batch = []
                results.append(
                    self.get(
                        namespace=list(op.namespace),
                        key=op.key,
                        refresh_ttl=op.refresh_ttl,
                    )
                )
            elif isinstance(op, SearchOp):
                if curr_batch:
                    self.collection.bulk_write(curr_batch)
                    curr_batch = []
                results.append(
                    self.search(
                        list(op.namespace_prefix),
                        query=op.query,
                        filter=op.filter,
                        limit=op.limit,
                        offset=op.offset,
                        refresh_ttl=op.refresh_ttl,
                    )
                )
            elif isinstance(op, ListNamespacesOp):
                if curr_batch:
                    self.collection.bulk_write(curr_batch)
                    curr_batch = []

                prefix = None
                suffix = None
                if op.match_conditions:
                    for cond in op.match_conditions:
                        if cond.match_type == "prefix":
                            prefix = cond.path
                        elif cond.match_type == "suffix":
                            suffix = cond.path
                        else:
                            raise ValueError(
                                f"Match type {cond.match_type} must be prefix or suffix."
                            )
                results.append(
                    self.list_namespaces(
                        prefix=prefix,
                        suffix=suffix,
                        max_depth=op.max_depth,
                        limit=op.limit,
                        offset=op.offset,
                    )
                )
        if curr_batch:
            self.collection.bulk_write(curr_batch)
        return results

    async def abatch(self, ops: Iterable[Op]) -> list[Result]:
        """Execute multiple operations asynchronously in a single batch.

        Args:
            ops: An iterable of operations to execute.

        Returns:
            A list of results, where each result corresponds to an operation in the input.
            The order of results matches the order of input operations.
        """
        await run_in_executor(None, self.batch, ops)

    @classmethod
    @contextmanager
    def from_conn_string(
        cls,
        conn_string: Optional[str] = None,
        db_name: str = "checkpointing_db",
        collection_name: str = "persistent-store",
        ttl_config: Optional[TTLConfig] = None,
        **kwargs: Any,
    ) -> Iterator["MongoDBStore"]:
        """Context manager to create a persistent MongoDB key-value store.

        A unique compound index as shown below will be added to the collections
        backing the store (namespace, key). If the collection exists,
        and have indexes already, nothing will be done during initialization.

        If the `ttl` argument is provided, TTL functionality will be employed.
        This is done automatically via MongoDB's TTL Indexes, based on the
        `updated_at` field of the collection. The index will be created if it
        does not already exist.

        Args:
            conn_string: MongoDB connection string. See [class:~pymongo.MongoClient].
            db_name: Database name. It will be created if it doesn't exist.
            collection_name: Collection name backing the store. Created if it doesn't exist.
            ttl_config: Defines a TTL (in seconds) and whether to update on reads(get/search).
        Yields: A new MongoDBStore.
        """
        client: Optional[MongoClient] = None
        try:
            client = MongoClient(
                conn_string,
                driver=DriverInfo(
                    name="Langgraph", version=version("langgraph-checkpoint-mongodb")
                ),
            )
            db = client[db_name]
            if collection_name not in db.list_collection_names():
                db.create_collection(collection_name)
            collection = client[db_name][collection_name]

            yield MongoDBStore(
                collection=collection,
                ttl_config=ttl_config,
                **kwargs,
            )
        finally:
            if client:
                client.close()

    def search(
        self,
        namespace_prefix: tuple[str, ...],
        /,
        *,
        query: Optional[str] = None,
        filter: Optional[dict[str, Any]] = None,
        limit: int = 10,
        offset: int = 0,
        refresh_ttl: Optional[bool] = None,
    ) -> list[SearchItem]:
        """Search for items within a namespace prefix.

        Values are stored in the collection as a document of name 'value'.
        One uses dot notation to access embedded fields. For example,
        `value.text`, `value.address.city` and for arrays `value.titles.3`.

        Args:
            namespace_prefix: Hierarchical path prefix to search within.
            query: Optional query for natural language search.
            filter: Key-value pairs to filter results.
            limit: Maximum number of items to return.
            offset: Number of items to skip before returning results.
            refresh_ttl: TTL is not supported for search. Use get if needed.

        Returns:
            List of items matching the search criteria.

        ???+ example "Examples"
            Basic filtering:
            ```python
            # Search for documents with specific metadata
            results = store.search(
                ("docs",),
                filter={"value.type": "article", "value.status": "published"}
            )
            ```

            Natural language search (requires vector store implementation):
            ```python
            # Initialize store with embedding configuration
            store = YourStore( # e.g., InMemoryStore, AsyncPostgresStore
                index={
                    "dims": 1536,  # embedding dimensions
                    "embed": your_embedding_function,  # function to create embeddings
                    "fields": ["text"]  # fields to embed. Defaults to ["$"]
                }
            )

        """

        if query:
            raise NotImplementedError("Natural language search not yet implemented.")

        if offset:
            logger.warning("offset is not implemented in MongoDBStore")

        pipeline = []
        match_cond = {}
        if namespace_prefix:
            match_cond = {"$expr": self._match_prefix(namespace_prefix)}
        if filter:
            filter_cond = [{k: v} for k, v in filter.items()]
            match_cond = {"$and": [match_cond] + filter_cond}
        pipeline.append({"$match": match_cond})

        if limit:
            pipeline.append({"$limit": limit})

        """
        if refresh_ttl is True or (
            self.ttl_config
            and refresh_ttl is None
            and self.ttl_config["refresh_on_read"]
        ):
            pipeline.append({"$set": {"updated_at": "$$NOW"}})

            
            pipeline.append(
                {
                    "$merge": {
                        "into": self.collection.name,
                        "on": "_id",  # compound key
                        "whenMatched": "merge",  # merge with existing document
                        "whenNotMatched": "fail",    # or "insert" if you want to add new docs
                    }
                }
            )
        """
        results = self.collection.aggregate(pipeline)

        return [
            SearchItem(
                namespace=tuple(res["namespace"]),
                key=res["key"],
                value=res["value"],
                created_at=res["created_at"],
                updated_at=res["updated_at"],
                score=res.get("score"),
            )
            for res in results
        ]
