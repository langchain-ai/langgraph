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
        - CipherProtocol

::: langgraph.checkpoint.serde.jsonplus
    options:
      members:
        - JsonPlusSerializer

::: langgraph.checkpoint.serde.encrypted
    options:
      members:
        - EncryptedSerializer

::: langgraph.checkpoint.memory

::: langgraph.checkpoint.sqlite

::: langgraph.checkpoint.sqlite.aio

::: langgraph.checkpoint.postgres
    options:
      members:
        - PostgresSaver

::: langgraph.checkpoint.postgres.aio
    options:
      members:
        - AsyncPostgresSaver
