from typing import Any, Dict, Mapping, Sequence

from langchain.pydantic_v1 import Field
from langchain.schema.runnable import RunnableConfig
from langchain.schema.runnable.utils import ConfigurableFieldSpec

from permchain.checkpoint.base import BaseCheckpointAdapter


class MemoryCheckpoint(BaseCheckpointAdapter):
    storage: Dict[str, Mapping[str, Any]] = Field(default_factory=dict)

    @property
    def config_specs(self) -> Sequence[ConfigurableFieldSpec]:
        return [
            ConfigurableFieldSpec(
                id="thread_id",
                annotation=str,
                name="Thread ID",
                description=None,
                default="",
                is_shared=True,
            ),
        ]

    def get(self, config: RunnableConfig) -> Mapping[str, Any] | None:
        return self.storage.get(config["configurable"]["thread_id"], None)

    def put(self, config: RunnableConfig, checkpoint: Mapping[str, Any]) -> None:
        return self.storage.update({config["configurable"]["thread_id"]: checkpoint})
