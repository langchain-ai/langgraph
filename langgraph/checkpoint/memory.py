import asyncio
from collections import defaultdict
from functools import partial
from typing import AsyncIterator, Iterator, Optional

from langchain_core.runnables import RunnableConfig

from langgraph.checkpoint.base import (
    BaseCheckpointSaver,
    Checkpoint,
    CheckpointMetadata,
    CheckpointTuple,
    SerializerProtocol,
)


class MemorySaver(BaseCheckpointSaver):
    """An in-memory checkpoint saver.

    This checkpoint saver stores checkpoints in memory using a defaultdict.

    Note:
        Since checkpoints are saved in memory, they will be lost when the program exits.
        Only use this saver for debugging or testing purposes.

    Args:
        serde (Optional[SerializerProtocol]): The serializer to use for serializing and deserializing checkpoints. Defaults to None.

    Examples:

            import asyncio

            from langgraph.checkpoint.memory import MemorySaver
            from langgraph.graph import StateGraph

            builder = StateGraph(int)
            builder.add_node("add_one", lambda x: x + 1)
            builder.set_entry_point("add_one")
            builder.set_finish_point("add_one")

            memory = MemorySaver()
            graph = builder.compile(checkpointer=memory)
            coro = graph.ainvoke(1, {"configurable": {"thread_id": "thread-1"}})
            asyncio.run(coro)  # Output: 2
    """

    storage: defaultdict[str, dict[str, tuple[bytes, bytes]]]

    def __init__(
        self,
        *,
        serde: Optional[SerializerProtocol] = None,
    ) -> None:
        super().__init__(serde=serde)
        self.storage = defaultdict(dict)

    def get_tuple(self, config: RunnableConfig) -> Optional[CheckpointTuple]:
        """Get a checkpoint tuple from the in-memory storage.

        This method retrieves a checkpoint tuple from the in-memory storage based on the
        provided config. If the config contains a "thread_ts" key, the checkpoint with
        the matching thread ID and timestamp is retrieved. Otherwise, the latest checkpoint
        for the given thread ID is retrieved.

        Args:
            config (RunnableConfig): The config to use for retrieving the checkpoint.

        Returns:
            Optional[CheckpointTuple]: The retrieved checkpoint tuple, or None if no matching checkpoint was found.
        """
        thread_id = config["configurable"]["thread_id"]
        if ts := config["configurable"].get("thread_ts"):
            if saved := self.storage[thread_id].get(ts):
                checkpoint, metadata = saved
                return CheckpointTuple(
                    config=config,
                    checkpoint=self.serde.loads(checkpoint),
                    metadata=self.serde.loads(metadata),
                )
        else:
            if checkpoints := self.storage[thread_id]:
                ts = max(checkpoints.keys())
                checkpoint, metadata = checkpoints[ts]
                return CheckpointTuple(
                    config={"configurable": {"thread_id": thread_id, "thread_ts": ts}},
                    checkpoint=self.serde.loads(checkpoint),
                    metadata=self.serde.loads(metadata),
                )

    def list(
        self,
        config: RunnableConfig,
        *,
        before: Optional[RunnableConfig] = None,
        limit: Optional[int] = None,
    ) -> Iterator[CheckpointTuple]:
        """List checkpoints from the in-memory storage.

        This method retrieves a list of checkpoint tuples from the in-memory storage based
        on the provided config. The checkpoints are ordered by timestamp in descending order.

        Args:
            config (RunnableConfig): The config to use for listing the checkpoints.
            before (Optional[RunnableConfig]): If provided, only checkpoints before the specified timestamp are returned. Defaults to None.
            limit (Optional[int]): The maximum number of checkpoints to return. Defaults to None.

        Yields:
            Iterator[CheckpointTuple]: An iterator of checkpoint tuples.
        """
        thread_id = config["configurable"]["thread_id"]
        for ts, (checkpoint, metadata) in self.storage[thread_id].items():
            if before and ts >= before["configurable"]["thread_ts"]:
                continue
            if limit is not None and limit <= 0:
                break
            limit -= 1
            yield CheckpointTuple(
                config={"configurable": {"thread_id": thread_id, "thread_ts": ts}},
                checkpoint=self.serde.loads(checkpoint),
                metadata=self.serde.loads(metadata),
            )

    def search(
        self,
        metadata_filter: CheckpointMetadata,
        *,
        before: Optional[RunnableConfig] = None,
        limit: Optional[int] = None,
    ) -> Iterator[CheckpointTuple]:
        """Search for checkpoints by metadata.

        This method retrieves a list of checkpoint tuples from the in-memory
        storage based on the provided metadata filter. The metadata filter does
        not need to contain all keys defined in the CheckpointMetadata class.
        The checkpoints are ordered by timestamp in descending order.

        Args:
            metadata_filter (CheckpointMetadata): The metadata filter to use for searching the checkpoints.
            before (Optional[RunnableConfig]): If provided, only checkpoints before the specified timestamp are returned. Defaults to None.
            limit (Optional[int]): The maximum number of checkpoints to return. Defaults to None.

        Yields:
            Iterator[CheckpointTuple]: An iterator of checkpoint tuples.
        """
        for thread_id, checkpoints in self.storage.items():
            for ts, (checkpoint_bytes, metadata_bytes) in checkpoints.items():
                # filter by thread_ts
                if before and ts >= before["configurable"]["thread_ts"]:
                    continue

                # check if all query key/value pairs match the metadata
                metadata = self.serde.loads(metadata_bytes)
                all_keys_match = all(
                    query_value == metadata[query_key]
                    for query_key, query_value in metadata_filter.items()
                )

                # if all query key/value pairs match, yield the checkpoint
                if all_keys_match:
                    # limit search results
                    if limit is not None:
                        if limit <= 0:
                            break
                        limit -= 1

                    yield CheckpointTuple(
                        config={
                            "configurable": {"thread_id": thread_id, "thread_ts": ts}
                        },
                        checkpoint=self.serde.loads(checkpoint_bytes),
                        metadata=metadata,
                    )

    def put(
        self,
        config: RunnableConfig,
        checkpoint: Checkpoint,
        metadata: CheckpointMetadata,
    ) -> RunnableConfig:
        """Save a checkpoint to the in-memory storage.

        This method saves a checkpoint to the in-memory storage. The checkpoint is associated
        with the provided config.

        Args:
            config (RunnableConfig): The config to associate with the checkpoint.
            checkpoint (Checkpoint): The checkpoint to save.

        Returns:
            RunnableConfig: The updated config containing the saved checkpoint's timestamp.
        """
        self.storage[config["configurable"]["thread_id"]].update(
            {
                checkpoint["ts"]: (
                    self.serde.dumps(checkpoint),
                    self.serde.dumps(metadata),
                )
            }
        )
        return {
            "configurable": {
                "thread_id": config["configurable"]["thread_id"],
                "thread_ts": checkpoint["ts"],
            }
        }

    async def aget_tuple(self, config: RunnableConfig) -> Optional[CheckpointTuple]:
        """Asynchronous version of get_tuple.

        This method is an asynchronous wrapper around get_tuple that runs the synchronous
        method in a separate thread using asyncio.

        Args:
            config (RunnableConfig): The config to use for retrieving the checkpoint.

        Returns:
            Optional[CheckpointTuple]: The retrieved checkpoint tuple, or None if no matching checkpoint was found.
        """
        return await asyncio.get_running_loop().run_in_executor(
            None, self.get_tuple, config
        )

    async def alist(self, config: RunnableConfig) -> AsyncIterator[CheckpointTuple]:
        """Asynchronous version of list.

        This method is an asynchronous wrapper around list that runs the synchronous
        method in a separate thread using asyncio.

        Args:
            config (RunnableConfig): The config to use for listing the checkpoints.

        Yields:
            AsyncIterator[CheckpointTuple]: An asynchronous iterator of checkpoint tuples.
        """
        loop = asyncio.get_running_loop()
        iter = loop.run_in_executor(None, self.list, config)
        while True:
            try:
                yield await loop.run_in_executor(None, next, iter)
            except StopIteration:
                return

    async def asearch(
        self,
        metadata_filter: CheckpointMetadata,
        *,
        before: Optional[RunnableConfig] = None,
        limit: Optional[int] = None,
    ) -> AsyncIterator[CheckpointTuple]:
        """Asynchronous version of search.

        This method is an asynchronous wrapper around search that runs the synchronous
        method in a separate thread using asyncio.
        """
        loop = asyncio.get_running_loop()
        iter = await loop.run_in_executor(
            None, partial(self.search, before=before, limit=limit), metadata_filter
        )

        while True:
            if item := await loop.run_in_executor(None, next, iter, None):
                yield item
            else:
                break

    async def aput(
        self,
        config: RunnableConfig,
        checkpoint: Checkpoint,
        metadata: CheckpointMetadata,
    ) -> RunnableConfig:
        return await asyncio.get_running_loop().run_in_executor(
            None, self.put, config, checkpoint, metadata
        )
