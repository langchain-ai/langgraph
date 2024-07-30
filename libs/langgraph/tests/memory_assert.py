import asyncio
from collections import defaultdict
from typing import Any, Optional

from langchain_core.runnables import RunnableConfig
from langgraph_checkpoint.base import (
    Checkpoint,
    CheckpointMetadata,
    CheckpointTuple,
    SerializerProtocol,
    copy_checkpoint,
)
from langgraph_checkpoint.memory import MemorySaver


class NoopSerializer(SerializerProtocol):
    def loads(self, data: bytes) -> Any:
        return data

    def dumps(self, obj: Any) -> bytes:
        return obj


class MemorySaverAssertImmutable(MemorySaver):
    serde = NoopSerializer()

    storage_for_copies: defaultdict[str, dict[str, Checkpoint]]

    def __init__(
        self,
        *,
        serde: Optional[SerializerProtocol] = None,
        put_sleep: Optional[float] = None,
    ) -> None:
        super().__init__(serde=serde)
        self.storage_for_copies = defaultdict(dict)
        self.put_sleep = put_sleep

    def put(
        self,
        config: dict,
        checkpoint: Checkpoint,
        metadata: Optional[CheckpointMetadata] = None,
    ) -> None:
        if self.put_sleep:
            import time

            time.sleep(self.put_sleep)
        # assert checkpoint hasn't been modified since last written
        thread_id = config["configurable"]["thread_id"]
        if saved := super().get(config):
            assert (
                self.serde.loads(self.storage_for_copies[thread_id][saved["id"]])
                == saved
            )
        self.storage_for_copies[thread_id][checkpoint["id"]] = self.serde.dumps(
            copy_checkpoint(checkpoint)
        )
        # call super to write checkpoint
        return super().put(config, checkpoint, metadata)


class MemorySaverAssertCheckpointMetadata(MemorySaver):
    """This custom checkpointer is for verifying that a run's configurable
    fields are merged with the previous checkpoint config for each step in
    the run. This is the desired behavior. Because the checkpointer's (a)put()
    method is called for each step, the implementation of this checkpointer
    should produce a side effect that can be asserted.
    """

    serde = NoopSerializer()

    def __init__(
        self,
        *,
        serde: Optional[SerializerProtocol] = None,
    ) -> None:
        super().__init__(serde=serde)

    def put(
        self,
        config: RunnableConfig,
        checkpoint: Checkpoint,
        metadata: Optional[CheckpointMetadata] = None,
    ) -> None:
        """The implementation of put() merges config["configurable"] (a run's
        configurable fields) with the metadata field. The state of the
        checkpoint metadata can be asserted to confirm that the run's
        configurable fields were merged with the previous checkpoint config.
        """
        configurable = config["configurable"].copy()

        # remove thread_ts to make testing simpler
        thread_ts = configurable.pop("thread_ts", None)

        self.storage[config["configurable"]["thread_id"]].update(
            {
                checkpoint["id"]: (
                    self.serde.dumps(checkpoint),
                    # merge configurable fields and metadata
                    self.serde.dumps({**configurable, **metadata}),
                    thread_ts,
                )
            }
        )
        return {
            "configurable": {
                "thread_id": config["configurable"]["thread_id"],
                "thread_ts": checkpoint["id"],
            }
        }

    async def aput(
        self,
        config: RunnableConfig,
        checkpoint: Checkpoint,
        metadata: CheckpointMetadata,
    ) -> RunnableConfig:
        return await asyncio.get_running_loop().run_in_executor(
            None, self.put, config, checkpoint, metadata
        )


class MemorySaverNoPending(MemorySaver):
    def get_tuple(self, config: RunnableConfig) -> Optional[CheckpointTuple]:
        result = super().get_tuple(config)
        if result:
            return CheckpointTuple(result.config, result.checkpoint, result.metadata)
        return result
