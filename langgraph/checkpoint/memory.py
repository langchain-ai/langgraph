from langchain_core.pydantic_v1 import Field
from langchain_core.runnables import RunnableConfig
from langchain_core.runnables.utils import ConfigurableFieldSpec

from langgraph.checkpoint.base import BaseCheckpointSaver, Checkpoint


class MemorySaver(BaseCheckpointSaver):
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
