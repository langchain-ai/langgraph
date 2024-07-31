# Checkpoints

You can [compile][langgraph.graph.MessageGraph.compile] any LangGraph workflow with a [CheckPointer][https://github.com/langchain-ai/langgraph/blob/main/libs/checkpoint/langgraph_checkpoint/base.py#L151] to give your agent "memory" by persisting its state. This permits things like:

- Remembering things across multiple interactions
- Interrupting to wait for user input
- Resilience for long-running, error-prone agents
- Time travel retry and branch from a previous checkpoint

Key checkpointer interfaces and primitives are defined in [`langgraph_checkpoint`](https://github.com/langchain-ai/langgraph/tree/main/libs/checkpoint) library.

## Implementations

LangGraph natively provides the following checkpoint implementations:

### MemorySaver (via `langgraph_checkpoint`)

### AsyncSqliteSaver

::: langgraph.checkpoint.aiosqlite.AsyncSqliteSaver

### SqliteSaver

::: langgraph.checkpoint.sqlite.SqliteSaver
handler: python
