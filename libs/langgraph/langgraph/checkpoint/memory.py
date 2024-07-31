import asyncio
from collections import defaultdict
from functools import partial
from typing import Any, AsyncIterator, Dict, Iterator, List, Optional, Tuple

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

    storage: defaultdict[str, dict[str, tuple[bytes, bytes, Optional[str]]]]

    def __init__(
        self,
        *,
        serde: Optional[SerializerProtocol] = None,
    ) -> None:
        super().__init__(serde=serde)
        self.storage = defaultdict(dict)
        self.writes = defaultdict(list)

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
                checkpoint, metadata, parent_ts = saved
                writes = self.writes[(thread_id, ts)]
                return CheckpointTuple(
                    config=config,
                    checkpoint=self.serde.loads(checkpoint),
                    metadata=self.serde.loads(metadata),
                    pending_writes=[
                        (id, c, self.serde.loads(v)) for id, c, v in writes
                    ],
                    parent_config={
                        "configurable": {
                            "thread_id": thread_id,
                            "thread_ts": parent_ts,
                        }
                    }
                    if parent_ts
                    else None,
                )
        else:
            if checkpoints := self.storage[thread_id]:
                ts = max(checkpoints.keys())
                checkpoint, metadata, parent_ts = checkpoints[ts]
                writes = self.writes[(thread_id, ts)]
                return CheckpointTuple(
                    config={"configurable": {"thread_id": thread_id, "thread_ts": ts}},
                    checkpoint=self.serde.loads(checkpoint),
                    metadata=self.serde.loads(metadata),
                    pending_writes=[
                        (id, c, self.serde.loads(v)) for id, c, v in writes
                    ],
                    parent_config={
                        "configurable": {
                            "thread_id": thread_id,
                            "thread_ts": parent_ts,
                        }
                    }
                    if parent_ts
                    else None,
                )

    def list(
        self,
        config: Optional[RunnableConfig],
        *,
        filter: Optional[Dict[str, Any]] = None,
        before: Optional[RunnableConfig] = None,
        limit: Optional[int] = None,
        as_prefix: bool = False,
    ) -> Iterator[CheckpointTuple]:
        """List checkpoints from the in-memory storage.

        This method retrieves a list of checkpoint tuples from the in-memory storage based
        on the provided criteria.

        Args:
            config (Optional[RunnableConfig]): Base configuration for filtering checkpoints.
            filter (Optional[Dict[str, Any]]): Additional filtering criteria for metadata.
            before (Optional[RunnableConfig]): List checkpoints created before this configuration.
            limit (Optional[int]): Maximum number of checkpoints to return.

        Yields:
            Iterator[CheckpointTuple]: An iterator of matching checkpoint tuples.
        """
        if config:
            config_thread_id = config["configurable"]["thread_id"]
            if as_prefix:
                thread_ids = (
                    thread_id
                    for thread_id in self.storage
                    if thread_id.startswith(config_thread_id)
                )
            else:
                thread_ids = (config_thread_id,)
        else:
            thread_ids = self.storage.keys()

        for thread_id in thread_ids:
            for ts, (checkpoint, metadata_b, parent_ts) in sorted(
                self.storage[thread_id].items(), key=lambda x: x[0], reverse=True
            ):
                # filter by thread_ts
                if before and ts >= before["configurable"]["thread_ts"]:
                    continue

                # filter by metadata
                metadata = self.serde.loads(metadata_b)
                if filter and not all(
                    query_value == metadata[query_key]
                    for query_key, query_value in filter.items()
                ):
                    continue

                # limit search results
                if limit is not None and limit <= 0:
                    break
                elif limit is not None:
                    limit -= 1

                yield CheckpointTuple(
                    config={"configurable": {"thread_id": thread_id, "thread_ts": ts}},
                    checkpoint=self.serde.loads(checkpoint),
                    metadata=metadata,
                    parent_config={
                        "configurable": {
                            "thread_id": thread_id,
                            "thread_ts": parent_ts,
                        }
                    }
                    if parent_ts
                    else None,
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
            metadata (CheckpointMetadata): Additional metadata to save with the checkpoint.

        Returns:
            RunnableConfig: The updated config containing the saved checkpoint's timestamp.
        """
        self.storage[config["configurable"]["thread_id"]].update(
            {
                checkpoint["id"]: (
                    self.serde.dumps(checkpoint),
                    self.serde.dumps(metadata),
                    config["configurable"].get("thread_ts"),  # parent
                )
            }
        )
        return {
            "configurable": {
                "thread_id": config["configurable"]["thread_id"],
                "thread_ts": checkpoint["id"],
            }
        }

    def put_writes(
        self,
        config: RunnableConfig,
        writes: List[Tuple[str, Any]],
        task_id: str,
    ) -> RunnableConfig:
        """Save a list of writes to the in-memory storage.

        This method saves a list of writes to the in-memory storage. The writes are associated
        with the provided config.

        Args:
            config (RunnableConfig): The config to associate with the writes.
            writes (list[tuple[str, Any]]): The writes to save.
            task_id (str): Identifier for the task creating the writes.

        Returns:
            RunnableConfig: The updated config containing the saved writes' timestamp.
        """
        thread_id = config["configurable"]["thread_id"]
        ts = config["configurable"]["thread_ts"]
        self.writes[(thread_id, ts)].extend(
            [(task_id, c, self.serde.dumps(v)) for c, v in writes]
        )

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

    async def alist(
        self,
        config: Optional[RunnableConfig],
        *,
        filter: Optional[Dict[str, Any]] = None,
        before: Optional[RunnableConfig] = None,
        limit: Optional[int] = None,
        as_prefix: bool = False,
    ) -> AsyncIterator[CheckpointTuple]:
        """Asynchronous version of list.

        This method is an asynchronous wrapper around list that runs the synchronous
        method in a separate thread using asyncio.

        Args:
            config (RunnableConfig): The config to use for listing the checkpoints.

        Yields:
            AsyncIterator[CheckpointTuple]: An asynchronous iterator of checkpoint tuples.
        """
        loop = asyncio.get_running_loop()
        iter = await loop.run_in_executor(
            None,
            partial(
                self.list,
                before=before,
                limit=limit,
                filter=filter,
                as_prefix=as_prefix,
            ),
            config,
        )
        while True:
            # handling StopIteration exception inside coroutine won't work
            # as expected, so using next() with default value to break the loop
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
        """Asynchronous version of put.

        Args:
            config (RunnableConfig): The config to associate with the checkpoint.
            checkpoint (Checkpoint): The checkpoint to save.
            metadata (CheckpointMetadata): Additional metadata to save with the checkpoint.

        Returns:
            RunnableConfig: The updated config containing the saved checkpoint's timestamp.
        """
        return await asyncio.get_running_loop().run_in_executor(
            None, self.put, config, checkpoint, metadata
        )

    async def aput_writes(
        self,
        config: RunnableConfig,
        writes: List[Tuple[str, Any]],
        task_id: str,
    ) -> RunnableConfig:
        """Asynchronous version of put_writes.

        This method is an asynchronous wrapper around put_writes that runs the synchronous
        method in a separate thread using asyncio.

        Args:
            config (RunnableConfig): The config to associate with the writes.
            writes (List[Tuple[str, Any]]): The writes to save, each as a (channel, value) pair.
            task_id (str): Identifier for the task creating the writes.
        """
        return await asyncio.get_running_loop().run_in_executor(
            None, self.put_writes, config, writes, task_id
        )
