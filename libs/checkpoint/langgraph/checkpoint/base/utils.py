from typing import Optional

from langchain_core.runnables import RunnableConfig


def get_checkpoint_id(config: RunnableConfig) -> Optional[str]:
    """Get checkpoint ID in a backwards-compatible manner (fallback on thread_ts)."""
    return config["configurable"].get(
        "checkpoint_id", config["configurable"].get("thread_ts")
    )
