import asyncio
from abc import ABC
from collections import defaultdict
from copy import deepcopy
from datetime import datetime, timezone
from typing import Any, AsyncIterator, Iterator, NamedTuple, Optional, TypedDict

from langchain_core.load.serializable import Serializable
from langchain_core.runnables import RunnableConfig
from langchain_core.runnables.utils import ConfigurableFieldSpec

from langgraph.utils import StrEnum


class Checkpoint(TypedDict):
    """State snapshot at a given point in time."""

    v: int
    """The version of the checkpoint format. Currently 1."""
    ts: str
    """The timestamp of the checkpoint in ISO 8601 format."""
    channel_values: dict[str, Any]
    """The values of the channels at the time of the checkpoint.
    
    Mapping from channel name to channel snapshot value.
    """
    channel_versions: defaultdict[str, int]
    """The versions of the channels at the time of the checkpoint.
    
    The keys are channel names and the values are the logical time step
    at which the channel was last updated.
    """
    versions_seen: defaultdict[str, defaultdict[str, int]]
    """Map from node ID to map from channel name to version seen.
    
    This keeps track of the versions of the channels that each node has seen.
    
    Used to determine which nodes to execute next.
    """


def _seen_dict():
    return defaultdict(int)


def empty_checkpoint() -> Checkpoint:
    return Checkpoint(
        v=1,
        ts=datetime.now(timezone.utc).isoformat(),
        channel_values={},
        channel_versions=defaultdict(int),
        versions_seen=defaultdict(_seen_dict),
    )


def copy_checkpoint(checkpoint: Checkpoint) -> Checkpoint:
    return Checkpoint(
        v=checkpoint["v"],
        ts=checkpoint["ts"],
        channel_values=checkpoint["channel_values"].copy(),
        channel_versions=checkpoint["channel_versions"].copy(),
        versions_seen=deepcopy(checkpoint["versions_seen"]),
    )


class CheckpointAt(StrEnum):
    END_OF_STEP = "end_of_step"
    END_OF_RUN = "end_of_run"


class CheckpointTuple(NamedTuple):
    config: RunnableConfig
    checkpoint: Checkpoint
    parent_config: Optional[RunnableConfig] = None


CheckpointThreadId = ConfigurableFieldSpec(
    id="thread_id",
    annotation=str,
    name="Thread ID",
    description=None,
    default="",
    is_shared=True,
)

CheckpointThreadTs = ConfigurableFieldSpec(
    id="thread_ts",
    annotation=Optional[str],
    name="Thread Timestamp",
    description="Pass to fetch a past checkpoint. If None, fetches the latest checkpoint.",
    default=None,
    is_shared=True,
)


class BaseCheckpointSaver(Serializable, ABC):
    at: CheckpointAt = CheckpointAt.END_OF_RUN

    @property
    def config_specs(self) -> list[ConfigurableFieldSpec]:
        return [CheckpointThreadId, CheckpointThreadTs]

    def get(self, config: RunnableConfig) -> Optional[Checkpoint]:
        if value := self.get_tuple(config):
            return value.checkpoint

    def get_tuple(self, config: RunnableConfig) -> Optional[CheckpointTuple]:
        raise NotImplementedError

    def list(self, config: RunnableConfig) -> Iterator[CheckpointTuple]:
        raise NotImplementedError

    def put(self, config: RunnableConfig, checkpoint: Checkpoint) -> RunnableConfig:
        raise NotImplementedError

    async def aget(self, config: RunnableConfig) -> Optional[Checkpoint]:
        if value := await self.aget_tuple(config):
            return value.checkpoint

    async def aget_tuple(self, config: RunnableConfig) -> Optional[CheckpointTuple]:
        return await asyncio.get_running_loop().run_in_executor(
            None, self.get_tuple, config
        )

    async def alist(self, config: RunnableConfig) -> AsyncIterator[CheckpointTuple]:
        loop = asyncio.get_running_loop()
        iter = loop.run_in_executor(None, self.list, config)
        while True:
            try:
                yield await loop.run_in_executor(None, next, iter)
            except StopIteration:
                return

    async def aput(
        self, config: RunnableConfig, checkpoint: Checkpoint
    ) -> RunnableConfig:
        return await asyncio.get_running_loop().run_in_executor(
            None, self.put, config, checkpoint
        )
