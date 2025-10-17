import os
import tempfile
from collections import defaultdict
from functools import partial
from typing import Any

from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.base import (
    BaseCheckpointSaver,
    ChannelVersions,
    Checkpoint,
    CheckpointMetadata,
    CheckpointTuple,
    SerializerProtocol,
)
from langgraph.checkpoint.memory import InMemorySaver, PersistentDict

from langgraph.constants import TASKS


class NoopSerializer(SerializerProtocol):
    def loads_typed(self, data: tuple[str, bytes]) -> Any:
        return data[1]

    def dumps_typed(self, obj: Any) -> tuple[str, bytes]:
        return "type", obj


class MemorySaverNeedsPendingSendsMigration(BaseCheckpointSaver):
    def __init__(self) -> None:
        self.saver = InMemorySaver()

    def __getattribute__(self, name):
        if name in ("saver", "__class__", "get_tuple"):
            return object.__getattribute__(self, name)
        return getattr(self.saver, name)

    def get_tuple(self, config):
        if tup := self.saver.get_tuple(config):
            if tup.checkpoint["v"] == 4 and tup.checkpoint["channel_values"].get(TASKS):
                tup.checkpoint["v"] = 3
                tup.checkpoint["pending_sends"] = tup.checkpoint["channel_values"].pop(
                    TASKS
                )
                tup.checkpoint["channel_versions"].pop(TASKS)
                for seen in tup.checkpoint["versions_seen"].values():
                    seen.pop(TASKS, None)
        return tup


class MemorySaverAssertImmutable(InMemorySaver):
    storage_for_copies: defaultdict[str, dict[str, dict[str, Checkpoint]]]

    def __init__(
        self,
        *,
        serde: SerializerProtocol | None = None,
        put_sleep: float | None = None,
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
    def get_tuple(self, config: RunnableConfig) -> CheckpointTuple | None:
        result = super().get_tuple(config)
        if result:
            return CheckpointTuple(result.config, result.checkpoint, result.metadata)
        return result
