import asyncio
from abc import ABC, abstractmethod
from collections import defaultdict
from datetime import datetime, timezone
from typing import Any, Optional, TypedDict

from langchain_core.load.serializable import Serializable
from langchain_core.runnables import RunnableConfig
from langchain_core.runnables.utils import ConfigurableFieldSpec

from langgraph.utils import StrEnum


class Checkpoint(TypedDict):
    v: int
    ts: str
    channel_values: dict[str, Any]
    channel_versions: defaultdict[str, int]
    versions_seen: defaultdict[str, defaultdict[str, int]]


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


class CheckpointAt(StrEnum):
    END_OF_STEP = "end_of_step"
    END_OF_RUN = "end_of_run"


class BaseCheckpointSaver(Serializable, ABC):
    at: CheckpointAt = CheckpointAt.END_OF_RUN

    @property
    def config_specs(self) -> list[ConfigurableFieldSpec]:
        return []

    @abstractmethod
    def get(self, config: RunnableConfig) -> Optional[Checkpoint]:
        ...

    @abstractmethod
    def put(self, config: RunnableConfig, checkpoint: Checkpoint) -> None:
        ...

    async def aget(self, config: RunnableConfig) -> Optional[Checkpoint]:
        return await asyncio.get_running_loop().run_in_executor(None, self.get, config)

    async def aput(self, config: RunnableConfig, checkpoint: Checkpoint) -> None:
        return await asyncio.get_running_loop().run_in_executor(
            None, self.put, config, checkpoint
        )
