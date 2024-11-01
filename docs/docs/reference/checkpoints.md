# Checkpointers

::: langgraph.checkpoint.base
    options:
      members:
        - CheckpointMetadata
        - Checkpoint
        - BaseCheckpointSaver
        - create_checkpoint

::: langgraph.checkpoint.serde.base
    options:
      members:
        - SerializerProtocol

::: langgraph.checkpoint.serde.jsonplus
    options:
      members:
        - JsonPlusSerializer

::: langgraph.checkpoint.memory

::: langgraph.checkpoint.sqlite

::: langgraph.checkpoint.sqlite.aio

::: langgraph.checkpoint.postgres

::: langgraph.checkpoint.postgres.aio