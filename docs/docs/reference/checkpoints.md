# Checkpoints

::: langgraph.checkpoint
    handler: python

## BaseCheckpointSaver

::: langgraph.checkpoint.base.BaseCheckpointSaver
    handler: python

## Implementations

LangGraph also natively provides the following checkpoint implementations.

## AsyncSqliteSaver

::: langgraph.checkpoint.aiosqlite.AsyncSqliteSaver
    handler: python


## SqliteSaver

::: langgraph.checkpoint.sqlite.SqliteSaver
    handler: python
    members:
      - put
      - list
      - get_tuple


## MemorySaver

::: langgraph.checkpoint.memory.MemorySaver
    handler: python