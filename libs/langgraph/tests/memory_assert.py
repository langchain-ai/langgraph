import os
import tempfile
from collections import defaultdict
from functools import partial
from typing import Any, Optional

from langchain_core.runnables import RunnableConfig

from langgraph.checkpoint.base import (
    ChannelVersions,
    Checkpoint,
    CheckpointMetadata,
    CheckpointTuple,
    SerializerProtocol,
)
from langgraph.checkpoint.memory import InMemorySaver, PersistentDict


class NoopSerializer(SerializerProtocol):
    def loads_typed(self, data: tuple[str, bytes]) -> Any:
        return data[1]

    def dumps_typed(self, obj: Any) -> tuple[str, bytes]:
        return "type", obj


class MemorySaverAssertImmutable(InMemorySaver):
    storage_for_copies: defaultdict[str, dict[str, dict[str, Checkpoint]]]

    def __init__(
        self,
        *,
        serde: Optional[SerializerProtocol] = None,
        put_sleep: Optional[float] = None,
    ) -> None:
        _, filename = tempfile.mkstemp()
        super().__init__(
            serde=serde, factory=partial(PersistentDict, filename=filename)
        )
        self.storage_for_copies = defaultdict(lambda: defaultdict(dict))
        self.put_sleep = put_sleep
        self.stack.callback(os.remove, filename)

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
            ), config["configurable"]["checkpoint_ns"]
        self.storage_for_copies[thread_id][checkpoint_ns][checkpoint["id"]] = (
            self.serde.dumps_typed(checkpoint)
        )
        # call super to write checkpoint
        return super().put(config, checkpoint, metadata, new_versions)


class MemorySaverNoPending(InMemorySaver):
    def get_tuple(self, config: RunnableConfig) -> Optional[CheckpointTuple]:
        result = super().get_tuple(config)
        if result:
            return CheckpointTuple(result.config, result.checkpoint, result.metadata)
        return result
