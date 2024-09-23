# Checkpointers

You can [compile][langgraph.graph.MessageGraph.compile] any LangGraph workflow with a [Checkpointer][basecheckpointsaver] to give your agent "memory" by persisting its state. This permits things like:

- Remembering things across multiple interactions
- Interrupting to wait for user input
- Resilience for long-running, error-prone agents
- Time travel retry and branch from a previous checkpoint

Key checkpointer interfaces and primitives are defined in [`langgraph_checkpoint`](https://github.com/langchain-ai/langgraph/tree/main/libs/checkpoint) library. Additional checkpointer implementations are also available as installable libraries:
* [`langgraph-checkpoint-sqlite`](https://github.com/langchain-ai/langgraph/tree/main/libs/checkpoint-sqlite): An implementation of LangGraph checkpointer that uses SQLite database. Ideal for experimentation and local workflows.  
* [`langgraph-checkpoint-postgres`](https://github.com/langchain-ai/langgraph/tree/main/libs/checkpoint-postgres): An advanced checkpointer that uses Postgres database, used in LangGraph Cloud. Ideal for using in production.  

### Checkpoint

::: langgraph.checkpoint.base.Checkpoint

### CheckpointMetadata

::: langgraph.checkpoint.base.CheckpointMetadata

### BaseCheckpointSaver

::: langgraph.checkpoint.base.BaseCheckpointSaver

## Serialization / deserialization

### SerializerProtocol

::: langgraph.checkpoint.base.SerializerProtocol

### JsonPlusSerializer

::: langgraph.checkpoint.serde.jsonplus.JsonPlusSerializer

## Checkpointer Implementations

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
