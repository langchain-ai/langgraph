import asyncio
from collections import defaultdict
from contextlib import AbstractAsyncContextManager, AbstractContextManager
from functools import partial
from types import TracebackType
from typing import Any, AsyncIterator, Dict, Iterator, List, Optional, Tuple

from langchain_core.runnables import RunnableConfig

from langgraph.checkpoint.base import (
    WRITES_IDX_MAP,
    BaseCheckpointSaver,
    ChannelVersions,
    Checkpoint,
    CheckpointMetadata,
    CheckpointTuple,
    SerializerProtocol,
    get_checkpoint_id,
)
from langgraph.checkpoint.serde.types import TASKS


class MemorySaver(
    BaseCheckpointSaver, AbstractContextManager, AbstractAsyncContextManager
):
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

    # thread ID ->  checkpoint NS -> checkpoint ID -> checkpoint mapping
    storage: defaultdict[str, dict[str, dict[str, tuple[bytes, bytes, Optional[str]]]]]
    writes: defaultdict[
        tuple[str, str, str], dict[tuple[str, int], tuple[str, str, bytes]]
    ]

    def __init__(
        self,
        *,
        serde: Optional[SerializerProtocol] = None,
    ) -> None:
        super().__init__(serde=serde)
        self.storage = defaultdict(lambda: defaultdict(dict))
        self.writes = defaultdict(dict)

    def __enter__(self) -> "MemorySaver":
        return self

    def __exit__(
        self,
        exc_type: Optional[type[BaseException]],
        exc_value: Optional[BaseException],
        traceback: Optional[TracebackType],
    ) -> Optional[bool]:
        return

    async def __aenter__(self) -> "MemorySaver":
        return self

    async def __aexit__(
        self,
        __exc_type: Optional[type[BaseException]],
        __exc_value: Optional[BaseException],
        __traceback: Optional[TracebackType],
    ) -> Optional[bool]:
        return

    def get_tuple(self, config: RunnableConfig) -> Optional[CheckpointTuple]:
        """Get a checkpoint tuple from the in-memory storage.

        This method retrieves a checkpoint tuple from the in-memory storage based on the
        provided config. If the config contains a "checkpoint_id" key, the checkpoint with
        the matching thread ID and timestamp is retrieved. Otherwise, the latest checkpoint
        for the given thread ID is retrieved.

        Args:
            config (RunnableConfig): The config to use for retrieving the checkpoint.

        Returns:
            Optional[CheckpointTuple]: The retrieved checkpoint tuple, or None if no matching checkpoint was found.
        """
        thread_id = config["configurable"]["thread_id"]
        checkpoint_ns = config["configurable"].get("checkpoint_ns", "")
        if checkpoint_id := get_checkpoint_id(config):
            if saved := self.storage[thread_id][checkpoint_ns].get(checkpoint_id):
                checkpoint, metadata, parent_checkpoint_id = saved
                writes = self.writes[(thread_id, checkpoint_ns, checkpoint_id)].values()
                if parent_checkpoint_id:
                    sends = [
                        w[2]
                        for w in self.writes[
                            (thread_id, checkpoint_ns, parent_checkpoint_id)
                        ].values()
                        if w[1] == TASKS
                    ]
                else:
                    sends = []
                return CheckpointTuple(
                    config=config,
                    checkpoint={
                        **self.serde.loads_typed(checkpoint),
                        "pending_sends": [self.serde.loads_typed(s) for s in sends],
                    },
                    metadata=self.serde.loads_typed(metadata),
                    pending_writes=[
                        (id, c, self.serde.loads_typed(v)) for id, c, v in writes
                    ],
                    parent_config={
                        "configurable": {
                            "thread_id": thread_id,
                            "checkpoint_ns": checkpoint_ns,
                            "checkpoint_id": parent_checkpoint_id,
                        }
                    }
                    if parent_checkpoint_id
                    else None,
                )
        else:
            if checkpoints := self.storage[thread_id][checkpoint_ns]:
                checkpoint_id = max(checkpoints.keys())
                checkpoint, metadata, parent_checkpoint_id = checkpoints[checkpoint_id]
                writes = self.writes[(thread_id, checkpoint_ns, checkpoint_id)].values()
                if parent_checkpoint_id:
                    sends = [
                        w[2]
                        for w in self.writes[
                            (thread_id, checkpoint_ns, parent_checkpoint_id)
                        ].values()
                        if w[1] == TASKS
                    ]
                else:
                    sends = []
                return CheckpointTuple(
                    config={
                        "configurable": {
                            "thread_id": thread_id,
                            "checkpoint_ns": checkpoint_ns,
                            "checkpoint_id": checkpoint_id,
                        }
                    },
                    checkpoint={
                        **self.serde.loads_typed(checkpoint),
                        "pending_sends": [self.serde.loads_typed(s) for s in sends],
                    },
                    metadata=self.serde.loads_typed(metadata),
                    pending_writes=[
                        (id, c, self.serde.loads_typed(v)) for id, c, v in writes
                    ],
                    parent_config={
                        "configurable": {
                            "thread_id": thread_id,
                            "checkpoint_ns": checkpoint_ns,
                            "checkpoint_id": parent_checkpoint_id,
                        }
                    }
                    if parent_checkpoint_id
                    else None,
                )

    def list(
        self,
        config: Optional[RunnableConfig],
        *,
        filter: Optional[Dict[str, Any]] = None,
        before: Optional[RunnableConfig] = None,
        limit: Optional[int] = None,
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
        thread_ids = (config["configurable"]["thread_id"],) if config else self.storage
        config_checkpoint_ns = (
            config["configurable"].get("checkpoint_ns") if config else None
        )
        config_checkpoint_id = get_checkpoint_id(config) if config else None
        for thread_id in thread_ids:
            for checkpoint_ns in self.storage[thread_id].keys():
                if (
                    config_checkpoint_ns is not None
                    and checkpoint_ns != config_checkpoint_ns
                ):
                    continue

                for checkpoint_id, (
                    checkpoint,
                    metadata_b,
                    parent_checkpoint_id,
                ) in sorted(
                    self.storage[thread_id][checkpoint_ns].items(),
                    key=lambda x: x[0],
                    reverse=True,
                ):
                    # filter by checkpoint ID from config
                    if config_checkpoint_id and checkpoint_id != config_checkpoint_id:
                        continue

                    # filter by checkpoint ID from `before` config
                    if (
                        before
                        and (before_checkpoint_id := get_checkpoint_id(before))
                        and checkpoint_id >= before_checkpoint_id
                    ):
                        continue

                    # filter by metadata
                    metadata = self.serde.loads_typed(metadata_b)
                    if filter and not all(
                        query_value == metadata.get(query_key)
                        for query_key, query_value in filter.items()
                    ):
                        continue

                    # limit search results
                    if limit is not None and limit <= 0:
                        break
                    elif limit is not None:
                        limit -= 1

                    writes = self.writes[
                        (thread_id, checkpoint_ns, checkpoint_id)
                    ].values()

                    if parent_checkpoint_id:
                        sends = [
                            w[2]
                            for w in self.writes[
                                (thread_id, checkpoint_ns, parent_checkpoint_id)
                            ].values()
                            if w[1] == TASKS
                        ]
                    else:
                        sends = []

                    yield CheckpointTuple(
                        config={
                            "configurable": {
                                "thread_id": thread_id,
                                "checkpoint_ns": checkpoint_ns,
                                "checkpoint_id": checkpoint_id,
                            }
                        },
                        checkpoint={
                            **self.serde.loads_typed(checkpoint),
                            "pending_sends": [self.serde.loads_typed(s) for s in sends],
                        },
                        metadata=metadata,
                        parent_config={
                            "configurable": {
                                "thread_id": thread_id,
                                "checkpoint_ns": checkpoint_ns,
                                "checkpoint_id": parent_checkpoint_id,
                            }
                        }
                        if parent_checkpoint_id
                        else None,
                        pending_writes=[
                            (id, c, self.serde.loads_typed(v)) for id, c, v in writes
                        ],
                    )

    def put(
        self,
        config: RunnableConfig,
        checkpoint: Checkpoint,
        metadata: CheckpointMetadata,
        new_versions: ChannelVersions,
    ) -> RunnableConfig:
        """Save a checkpoint to the in-memory storage.

        This method saves a checkpoint to the in-memory storage. The checkpoint is associated
        with the provided config.

        Args:
            config (RunnableConfig): The config to associate with the checkpoint.
            checkpoint (Checkpoint): The checkpoint to save.
            metadata (CheckpointMetadata): Additional metadata to save with the checkpoint.
            new_versions (dict): New versions as of this write

        Returns:
            RunnableConfig: The updated config containing the saved checkpoint's timestamp.
        """
        c = checkpoint.copy()
        c.pop("pending_sends")
        thread_id = config["configurable"]["thread_id"]
        checkpoint_ns = config["configurable"]["checkpoint_ns"]
        self.storage[thread_id][checkpoint_ns].update(
            {
                checkpoint["id"]: (
                    self.serde.dumps_typed(c),
                    self.serde.dumps_typed(metadata),
                    config["configurable"].get("checkpoint_id"),  # parent
                )
            }
        )
        return {
            "configurable": {
                "thread_id": thread_id,
                "checkpoint_ns": checkpoint_ns,
                "checkpoint_id": checkpoint["id"],
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
        checkpoint_ns = config["configurable"]["checkpoint_ns"]
        checkpoint_id = config["configurable"]["checkpoint_id"]
        outer_key = (thread_id, checkpoint_ns, checkpoint_id)
        for idx, (c, v) in enumerate(writes):
            inner_key = (task_id, WRITES_IDX_MAP.get(c, idx))
            self.writes[outer_key][inner_key] = (task_id, c, self.serde.dumps_typed(v))

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
        new_versions: ChannelVersions,
    ) -> RunnableConfig:
        """Asynchronous version of put.

        Args:
            config (RunnableConfig): The config to associate with the checkpoint.
            checkpoint (Checkpoint): The checkpoint to save.
            metadata (CheckpointMetadata): Additional metadata to save with the checkpoint.
            new_versions (dict): New versions as of this write

        Returns:
            RunnableConfig: The updated config containing the saved checkpoint's timestamp.
        """
        return await asyncio.get_running_loop().run_in_executor(
            None, self.put, config, checkpoint, metadata, new_versions
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
