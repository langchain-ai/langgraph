# Checkpoints

You can [compile][langgraph.graph.MessageGraph.compile] any LangGraph workflow with a [CheckPointer][basecheckpointsaver] to give your agent "memory" by persisting its state. This permits things like:

- Remembering things across multiple interactions
- Interrupting to wait for user input
- Resilience for long-running, error-prone agents
- Time travel retry and branch from a previous checkpoint

### Checkpoint

::: langgraph.checkpoint.base.Checkpoint

### CheckpointMetadata

::: langgraph.checkpoint.base.CheckpointMetadata

### BaseCheckpointSaver

::: langgraph.checkpoint.base.BaseCheckpointSaver

### SerializerProtocol

::: langgraph.checkpoint.SerializerProtocol

## Implementations

LangGraph also natively provides the following checkpoint implementations.

### MemorySaver

::: langgraph.checkpoint.memory.MemorySaver

### AsyncSqliteSaver

::: langgraph.checkpoint.aiosqlite.AsyncSqliteSaver

### SqliteSaver

::: langgraph.checkpoint.sqlite.SqliteSaver
handler: python
