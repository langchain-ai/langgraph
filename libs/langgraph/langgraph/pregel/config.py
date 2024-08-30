from typing import Any, Optional

from langchain_core.runnables import RunnableConfig


def patch_configurable(
    config: Optional[RunnableConfig], patch: dict[str, Any]
) -> RunnableConfig:
    if config is None:
        return {"configurable": patch}
    else:
        return {**config, "configurable": {**config["configurable"], **patch}}
