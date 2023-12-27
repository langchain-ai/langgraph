from langchain.pydantic_v1 import Field
from langchain.schema.runnable import RunnableConfig
from langchain.schema.runnable.utils import ConfigurableFieldSpec

from permchain.checkpoint.base import BaseCheckpointAdapter, Checkpoint


class MemoryCheckpoint(BaseCheckpointAdapter):
    storage: dict[str, Checkpoint] = Field(default_factory=dict)

    @property
    def config_specs(self) -> list[ConfigurableFieldSpec]:
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

    def get(self, config: RunnableConfig) -> Checkpoint | None:
        return self.storage.get(config["configurable"]["thread_id"], None)

    def put(self, config: RunnableConfig, checkpoint: Checkpoint) -> None:
        return self.storage.update({config["configurable"]["thread_id"]: checkpoint})
