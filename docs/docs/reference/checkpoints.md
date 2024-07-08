# Checkpoints

You can [compile](/reference/graphs/#langgraph.graph.MessageGraph.compile) any LangGraph workflow with a [CheckPointer](/reference/checkpoints/#basecheckpointsaver) to give your agent "memory" by persisting its state. This permits things like:

- Remembering things across multiple interactions
- Interrupting to wait for user input
- Resilience for long-running, error-prone agents
- Time travel retry and branch from a previous checkpoint

### Checkpoint

::: langgraph.checkpoint.Checkpoint

### BaseCheckpointSaver

::: langgraph.checkpoint.base.BaseCheckpointSaver
handler: python

### SerializerProtocol

::: langgraph.checkpoint.SerializerProtocol
handler: python

## Implementations

LangGraph also natively provides the following checkpoint implementations.

### MemorySaver

::: langgraph.checkpoint.memory.MemorySaver
handler: python

### AsyncSqliteSaver

::: langgraph.checkpoint.aiosqlite.AsyncSqliteSaver
handler: python

### SqliteSaver

::: langgraph.checkpoint.sqlite.SqliteSaver
handler: python
