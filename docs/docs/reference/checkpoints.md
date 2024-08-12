# Checkpoints

You can [compile][langgraph.graph.MessageGraph.compile] any LangGraph workflow with a [CheckPointer][basecheckpointsaver] to give your agent "memory" by persisting its state. This permits things like:

- Remembering things across multiple interactions
- Interrupting to wait for user input
- Resilience for long-running, error-prone agents
- Time travel retry and branch from a previous checkpoint

Key checkpointer interfaces and primitives are defined in [`langgraph_checkpoint`](https://github.com/langchain-ai/langgraph/tree/main/libs/checkpoint) library.

### Checkpoint

::: langgraph.checkpoint.base.Checkpoint

### CheckpointMetadata

::: langgraph.checkpoint.base.CheckpointMetadata

### BaseCheckpointSaver

::: langgraph.checkpoint.base.BaseCheckpointSaver

### SerializerProtocol

::: langgraph.checkpoint.base.SerializerProtocol

## Implementations

LangGraph also natively provides the following checkpoint implementations.

### MemorySaver

::: langgraph.checkpoint.memory.MemorySaver

### AsyncSqliteSaver

::: langgraph.checkpoint.sqlite.aio.AsyncSqliteSaver

### SqliteSaver

::: langgraph.checkpoint.sqlite.SqliteSaver

### AsyncPostgresSaver

::: langgraph.checkpoint.postgres.aio.AsyncPostgresSaver

### PostgresSaver

::: langgraph.checkpoint.postgres.PostgresSaver
handler: python


handler: python
