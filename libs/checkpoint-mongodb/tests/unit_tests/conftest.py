from typing import Any

import pytest
from langchain_core.runnables import RunnableConfig

from langgraph.checkpoint.base import (
    CheckpointMetadata,
    create_checkpoint,
    empty_checkpoint,
)


@pytest.fixture(scope="session")
def input_data() -> dict:
    """Setup and store conveniently in a single dictionary."""
    inputs: dict[str, Any] = {}

    inputs["config_1"] = RunnableConfig(
        configurable=dict(thread_id="thread-1", thread_ts="1", checkpoint_ns="")
    )  # config_1 tests deprecated thread_ts

    inputs["config_2"] = RunnableConfig(
        configurable=dict(thread_id="thread-2", checkpoint_id="2", checkpoint_ns="")
    )

    inputs["config_3"] = RunnableConfig(
        configurable=dict(
            thread_id="thread-2", checkpoint_id="2-inner", checkpoint_ns="inner"
        )
    )

    inputs["chkpnt_1"] = empty_checkpoint()
    inputs["chkpnt_2"] = create_checkpoint(inputs["chkpnt_1"], {}, 1)
    inputs["chkpnt_3"] = empty_checkpoint()

    inputs["metadata_1"] = CheckpointMetadata(
        source="input", step=2, writes={}, score=1
    )
    inputs["metadata_2"] = CheckpointMetadata(
        source="loop", step=1, writes={"foo": "bar"}, score=None
    )
    inputs["metadata_3"] = CheckpointMetadata()

    return inputs
