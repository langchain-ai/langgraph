from typing import Any, Optional

from langchain_core.runnables import RunnableConfig

from langgraph.checkpoint.base import CheckpointMetadata
from langgraph.constants import CONFIG_KEY_CHECKPOINT_MAP


def patch_configurable(
    config: Optional[RunnableConfig], patch: dict[str, Any]
) -> RunnableConfig:
    if config is None:
        return {"configurable": patch}
    else:
        return {**config, "configurable": {**config["configurable"], **patch}}


def patch_checkpoint_map(
    config: RunnableConfig, metadata: Optional[CheckpointMetadata]
) -> RunnableConfig:
    if parents := (metadata.get("parents") if metadata else None):
        return patch_configurable(
            config,
            {
                CONFIG_KEY_CHECKPOINT_MAP: {
                    **parents,
                    config["configurable"]["checkpoint_ns"]: config["configurable"][
                        "checkpoint_id"
                    ],
                },
            },
        )
    else:
        return config
