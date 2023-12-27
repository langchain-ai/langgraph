import asyncio
from abc import ABC, abstractmethod
from collections import defaultdict
from typing import Any, Mapping, TypedDict

from langchain.load.serializable import Serializable
from langchain.pydantic_v1 import Field
from langchain.schema.runnable import RunnableConfig
from langchain.schema.runnable.utils import ConfigurableFieldSpec

from permchain.utils import StrEnum


class Checkpoint(TypedDict):
    v: int
    ts: str
    values: dict[str, Any]
    versions: defaultdict[str, int]
    seen: defaultdict[str, defaultdict[str, int]]


def empty_checkpoint() -> Checkpoint:
    return Checkpoint(
        values={},
        versions=defaultdict(int),
        seen=defaultdict(lambda: defaultdict(int)),
    )


class CheckpointAt(StrEnum):
    END_OF_STEP = "end_of_step"
    END_OF_RUN = "end_of_run"


class BaseCheckpointAdapter(Serializable, ABC):
    at: CheckpointAt = CheckpointAt.END_OF_RUN

    @property
    def config_specs(self) -> list[ConfigurableFieldSpec]:
        return []

    @abstractmethod
    def get(self, config: RunnableConfig) -> Checkpoint | None:
        ...

    @abstractmethod
    def put(self, config: RunnableConfig, checkpoint: Checkpoint) -> None:
        ...

    async def aget(self, config: RunnableConfig) -> Checkpoint | None:
        return await asyncio.get_running_loop().run_in_executor(None, self.get, config)

    async def aput(self, config: RunnableConfig, checkpoint: Checkpoint) -> None:
        return await asyncio.get_running_loop().run_in_executor(
            None, self.put, config, checkpoint
        )


class StepState(Serializable):
    _checkpoint: Checkpoint

    _puts: set[str] = Field(default_factory=set)

    def get(self, key: str, default: Any | None = None) -> Any | None:
        return self._checkpoint["values"].get(key)

    def put(self, key: str, value: Any) -> None:
        self._puts.add(key)
        self._checkpoint["values"][key] = value
