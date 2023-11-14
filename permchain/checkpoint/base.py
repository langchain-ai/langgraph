import asyncio
import enum
from abc import ABC, abstractmethod
from typing import Any, Mapping, Sequence

from langchain.load.serializable import Serializable
from langchain.schema.runnable import RunnableConfig
from langchain.schema.runnable.utils import ConfigurableFieldSpec


# Before Python 3.11 native StrEnum is not available
class StrEnum(str, enum.Enum):
    """A string enum."""

    pass


class CheckpointAt(StrEnum):
    END_OF_STEP = "end_of_step"
    END_OF_RUN = "end_of_run"


class BaseCheckpointAdapter(Serializable, ABC):
    at: CheckpointAt = CheckpointAt.END_OF_RUN

    @property
    def config_specs(self) -> Sequence[ConfigurableFieldSpec]:
        return []

    @abstractmethod
    def get(self, config: RunnableConfig) -> Mapping[str, Any] | None:
        ...

    @abstractmethod
    def put(self, config: RunnableConfig, checkpoint: Mapping[str, Any]) -> None:
        ...

    async def aget(self, config: RunnableConfig) -> Mapping[str, Any] | None:
        return await asyncio.get_running_loop().run_in_executor(None, self.get, config)

    async def aput(self, config: RunnableConfig, checkpoint: Mapping[str, Any]) -> None:
        return await asyncio.get_running_loop().run_in_executor(
            None, self.put, config, checkpoint
        )
