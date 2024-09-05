import asyncio
from collections import defaultdict
from typing import Any, Optional

from langchain_core.runnables import RunnableConfig

from langgraph.checkpoint.base import (
    ChannelVersions,
    Checkpoint,
    CheckpointMetadata,
    CheckpointTuple,
    SerializerProtocol,
    copy_checkpoint,
)
from langgraph.checkpoint.memory import MemorySaver


class NoopSerializer(SerializerProtocol):
    def loads_typed(self, data: tuple[str, bytes]) -> Any:
        return data[1]

    def dumps_typed(self, obj: Any) -> tuple[str, bytes]:
        return "type", obj


class MemorySaverAssertImmutable(MemorySaver):
    storage_for_copies: defaultdict[str, dict[str, dict[str, Checkpoint]]]

    def __init__(
        self,
        *,
        serde: Optional[SerializerProtocol] = None,
        put_sleep: Optional[float] = None,
    ) -> None:
        super().__init__(serde=serde)
        self.storage_for_copies = defaultdict(lambda: defaultdict(dict))
        self.put_sleep = put_sleep

    def put(
        self,
        config: dict,
        checkpoint: Checkpoint,
        metadata: CheckpointMetadata,
        new_versions: ChannelVersions,
    ) -> None:
        if self.put_sleep:
            import time

            time.sleep(self.put_sleep)
        # assert checkpoint hasn't been modified since last written
        thread_id = config["configurable"]["thread_id"]
        checkpoint_ns = config["configurable"]["checkpoint_ns"]
        if saved := super().get(config):
            assert (
                self.serde.loads_typed(
                    self.storage_for_copies[thread_id][checkpoint_ns][saved["id"]]
                )
                == saved
            )
        self.storage_for_copies[thread_id][checkpoint_ns][checkpoint["id"]] = (
            self.serde.dumps_typed(copy_checkpoint(checkpoint))
        )
        # call super to write checkpoint
        return super().put(config, checkpoint, metadata, new_versions)


class MemorySaverAssertCheckpointMetadata(MemorySaver):
    """This custom checkpointer is for verifying that a run's configurable
    fields are merged with the previous checkpoint config for each step in
    the run. This is the desired behavior. Because the checkpointer's (a)put()
    method is called for each step, the implementation of this checkpointer
    should produce a side effect that can be asserted.
    """

    def put(
        self,
        config: RunnableConfig,
        checkpoint: Checkpoint,
        metadata: CheckpointMetadata,
        new_versions: ChannelVersions,
    ) -> None:
        """The implementation of put() merges config["configurable"] (a run's
        configurable fields) with the metadata field. The state of the
        checkpoint metadata can be asserted to confirm that the run's
        configurable fields were merged with the previous checkpoint config.
        """
        configurable = config["configurable"].copy()

        # remove checkpoint_id to make testing simpler
        checkpoint_id = configurable.pop("checkpoint_id", None)
        thread_id = config["configurable"]["thread_id"]
        checkpoint_ns = config["configurable"]["checkpoint_ns"]
        self.storage[thread_id][checkpoint_ns].update(
            {
                checkpoint["id"]: (
                    self.serde.dumps_typed(checkpoint),
                    # merge configurable fields and metadata
                    self.serde.dumps_typed({**configurable, **metadata}),
                    checkpoint_id,
                )
            }
        )
        return {
            "configurable": {
                "thread_id": config["configurable"]["thread_id"],
                "checkpoint_id": checkpoint["id"],
            }
        }

    async def aput(
        self,
        config: RunnableConfig,
        checkpoint: Checkpoint,
        metadata: CheckpointMetadata,
        new_versions: ChannelVersions,
    ) -> RunnableConfig:
        return await asyncio.get_running_loop().run_in_executor(
            None, self.put, config, checkpoint, metadata, new_versions
        )


class MemorySaverNoPending(MemorySaver):
    def get_tuple(self, config: RunnableConfig) -> Optional[CheckpointTuple]:
        result = super().get_tuple(config)
        if result:
            return CheckpointTuple(result.config, result.checkpoint, result.metadata)
        return result
