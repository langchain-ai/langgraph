import warnings
from abc import ABC, abstractmethod
from collections.abc import AsyncIterator, Iterator, Sequence
from contextlib import contextmanager
from datetime import datetime, timezone
from importlib.metadata import version
from typing import Any, Generic, Iterable, Literal, Optional, TypeVar, Union

from bson import SON
from langchain_core.exceptions import LangChainException
from langchain_core.runnables import run_in_executor
from pymongo import (
    DeleteMany,
    DeleteOne,
    MongoClient,
    ReplaceOne,
    UpdateMany,
    UpdateOne,
)
from pymongo.collection import Collection
from pymongo.driver_info import DriverInfo
from pymongo.errors import OperationFailure
from pymongo.collection import ReturnDocument

from langgraph.store.base import (
    NOT_PROVIDED,
    BaseStore,
    Item,
    NamespacePath,
    Op,
    PutOp,
    Result,
    TTLConfig,
)

K = TypeVar("K")
V = TypeVar("V")


class MongoDBStore(BaseStore):
    """MongoDB's persistent key-value stores for long-term memory.

    Stores enable persistence and memory that can be shared across threads,
    scoped to user IDs, assistant IDs, or other arbitrary namespaces.

    Supports semantic search capabilities through
    an optional `index` configuration.

    """

    supports_ttl: True
    # TODO - Do this via TTL Indexes (these are specified in seconds, not minutes.)

    ttl_config: Optional[TTLConfig] = (
        None  # TODO - Create index via this. This may be sufficient
    )

    """Database _collection backing the store."""

    def __init__(
        self,
        collection=Collection,
        ttl: Optional[float] = None,
        **kwargs: Any,
    ):
        self._collection = collection

        # Create indexes if not present
        # Create a unique index, akin to primary key, on namespace + key
        idx_keys = [idx["key"] for idx in self._collection.list_indexes()]
        if SON([("namespace", 1), ("key", 1)]) not in idx_keys:
            self._collection.create_index(keys=["namespace", "key"], unique=True)

        # Optionally, expire values using [TTL Index](https://www.mongodb.com/docs/manual/core/index-ttl/)
        if ttl is not None and SON([("updated_at", 1)]) not in idx_keys:
            self.ttl = float(ttl)
            self._collection.create_index("updated_at", expireAfterSeconds=ttl)

    def get_time(self) -> float:  # TODO - Not sure if we want floats
        """Get the current server time as a timestamp."""
        try:
            server_info = self._collection.database.command("hostInfo")
            local_time = server_info["system"]["currentTime"]
            timestamp = local_time.timestamp()
        except OperationFailure:
            with warnings.catch_warnings():
                warnings.simplefilter("once")
                warnings.warn(
                    "Could not get high-resolution timestamp, falling back to low-resolution",
                    stacklevel=2,
                )
            ping = self._collection.database.command("ping")
            local_time = ping["operationTime"]
            timestamp = float(local_time.time)
        return timestamp

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
        self._collection.bulk_write([op])

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
        if refresh_ttl is False:
            res = self._collection.find_one(
                filter={"namespace": namespace, "key": key},
            )
        else:
            res = self._collection.find_one_and_update(
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
        self._collection.delete_one({"namespace": list(namespace), "key": key})

    @staticmethod
    def _match_prefix(prefix: NamespacePath):
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
        """List and filter namespaces in the store.

        # TODO - Determine wildcards: slicing vs iterating on namespace.i or *
        # TODO  It seems like we can loop over prefix and add $match $and {"$namespace.i": prefix[i]
        #NamespacePath =  tuple[Union[str, Literal["*"]], ...]

        """
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

        results = self._collection.aggregate(pipeline)
        return [tuple(res["namespace"]) for res in results]

    def batch(self, ops: Iterable[Op]) -> list[Result]:
        """Execute multiple operations synchronously in a single batch.


        PostGres Version:
            1. Groups the operations, then appears to deduplicate and run by type!
            get, search, list, put - seems unusual, but no magic. See _group_ops


        TTL: We need to create index on updated_at field, with a constant TTL

        Args:
            ops: An iterable of operations to execute.

        Returns:
            A list of results, where each result corresponds to an operation in the input.
            The order of results matches the order of input operations.
        """
        raise NotImplementedError()

    async def abatch(self, ops: Iterable[Op]) -> list[Result]:
        """Execute multiple operations asynchronously in a single batch.

        Args:
            ops: An iterable of operations to execute.

        Returns:
            A list of results, where each result corresponds to an operation in the input.
            The order of results matches the order of input operations.
        """
        raise NotImplementedError()

    @classmethod
    @contextmanager
    def from_conn_string(
        cls,
        conn_string: Optional[str] = None,
        db_name: str = "checkpointing_db",
        collection_name: str = "persistent-store",
        ttl: Optional[float] = None,
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
            ttl: If provided, documents will be removed after this *number of seconds*.
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
                collection,
                ttl=ttl,
                **kwargs,
            )
        finally:
            if client:
                client.close()


# TODO
