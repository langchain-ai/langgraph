# LangGraph Checkpoint

This library defines the base interface for LangGraph checkpointers. Checkpointers provide persistence layer for LangGraph. They allow you to interact with and manage the graph's state. When you use a graph with a checkpointer, the checkpointer saves a _checkpoint_ of the graph state at every superstep, enabling several powerful capabilities like human-in-the-loop, "memory" between interactions and more.

## Key concepts

### Checkpoint

Checkpoint is a snapshot of the graph state at a given point in time. Checkpoint tuple refers to an object containing checkpoint and the associated config, metadata and pending writes.

### Thread

Threads enable the checkpointing of multiple different runs, making them essential for multi-tenant chat applications and other scenarios where maintaining separate states is necessary. A thread is a unique ID assigned to a series of checkpoints saved by a checkpointer. When using a checkpointer, you must specify a `thread_id` and optionally `checkpoint_ns` / `checkpoint_id` when running the graph.

- `thread_id` is simply the ID of a thread. This is always required
- `checkpoint_ns` can optionally be passed. This is a path to the subgraph the checkpoint originates from. Defaults to "" (root graph).
- `checkpoint_id` can optionally be passed. This identifier refers to a specific checkpoint within a thread. This can be used to kick of a run of a graph from some point halfway through a thread.

You must pass these when invoking the graph as part of the configurable part of the config, e.g.

```python
{"configurable": {"thread_id": "1"}}  # valid config
{"configurable": {"thread_id": "1", "checkpoint_id": "0c62ca34-ac19-445d-bbb0-5b4984975b2a"}}  # also valid config
```

### Serde

`langgraph_checkpoint` also defines protocol for serialization/deserialization (serde) and provides an default implementation (`langgraph_checkpoint.serde.jsonplus.JsonPlusSerializer`) that handles a wide variety of types, including LangChain and LangGraph primitives, datetimes, enums and more.

### Pending writes

When a graph node fails mid-execution at a given superstep, LangGraph stores pending checkpoint writes from any other nodes that completed successfully at that superstep, so that whenever we resume graph execution from that superstep we don't re-run the successful nodes.

## Interface

Each checkpointer should conform to `langgraph_checkpoint.BaseCheckpointSaver` interface and must implement the following methods:

- `.put` - Store a checkpoint with its configuration and metadata.
- `.put_writes` - Store intermediate writes linked to a checkpoint (i.e. pending writes).
- `.get_tuple` - Fetch a checkpoint tuple using for a given configuration (`thread_id` and `thread_ts`).
- `.list` - List checkpoints that match a given configuration and filter criteria.

If the checkpointer will be used with asynchronous graph execution (i.e. executing the graph via `.ainvoke`, `.astream`, `.abatch`), checkpointer must implement asynchronous versions of the above methods (`.aput`, `.aput_writes`, `.aget_tuple`, `.alist`).

## Usage

```shell
export ANTHROPIC_API_KEY=sk-...
```

```python
from langchain_core.tools import tool
from langchain_anthropic import ChatAnthropic
from langgraph.prebuilt import create_react_agent
from langgraph_checkpoint.memory import MemorySaver

@tool
def search(query: str):
    """Call to surf the web."""
    # This is a placeholder, but don't tell the LLM that...
    if "sf" in query.lower() or "san francisco" in query.lower():
        return "It's 60 degrees and foggy."
    return "It's 90 degrees and sunny."

llm = ChatAnthropic(model="claude-3-5-sonnet-20240620")
tools = [search]
checkpointer = MemorySaver()

agent = create_react_agent(llm, tools, checkpointer=checkpointer)
config = {"configurable": {"thread_id": "1"}}
agent.invoke({"messages": [("human", "what's the weather in sf")]}, config)
```