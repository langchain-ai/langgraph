# Getting Started with LangGraph Checkpoint Kusto

A beginner-friendly guide to using Azure Data Explorer (Kusto) as your LangGraph checkpoint storage.

## Table of Contents

1. [What is This?](#what-is-this)
2. [Prerequisites](#prerequisites)
3. [Installation](#installation)
4. [Setting Up Kusto](#setting-up-kusto)
5. [Your First Checkpoint](#your-first-checkpoint)
6. [Building a Simple Agent](#building-a-simple-agent)
7. [Understanding Checkpoints](#understanding-checkpoints)
8. [Common Patterns](#common-patterns)
9. [Troubleshooting](#troubleshooting)

## What is This?

When you build AI agents with LangGraph, you want them to remember their progress - like saving a game. This library lets you save those "checkpoints" to Azure Data Explorer (Kusto), a powerful cloud database.

**Why Kusto?**
- ✅ Scales to millions of checkpoints
- ✅ Fast queries across all your agent data
- ✅ Built-in analytics capabilities
- ✅ Enterprise-grade reliability

## Prerequisites

Before starting, you need:

1. **Python 3.10 or newer**
   ```bash
   python --version  # Should show 3.10+
   ```

2. **A Kusto cluster endpoint**
   
   You have several options:
   
   - **Azure Data Explorer**: [Create cluster](https://learn.microsoft.com/azure/data-explorer/create-cluster-database-portal) - Production ready with free tier
   - **Microsoft Fabric Eventhouse**: [Create Eventhouse](https://learn.microsoft.com/fabric/real-time-intelligence/create-eventhouse) - Integrated with Fabric
   - **Free Cluster**: [Get free access](https://dataexplorer.azure.com/freecluster) - No Azure subscription needed!
   
   Note your cluster URI (e.g., `https://mycluster.eastus.kusto.windows.net`)

3. **Azure credentials**
   - Use your Azure CLI login, or
   - Create a service principal with permissions

4. **Basic Python knowledge**
   - How to use async/await
   - How to install packages

## Installation

### Step 1: Install the Package

```bash
pip install langgraph-checkpoint-kusto
```

This installs:
- `langgraph-checkpoint-kusto` - This checkpoint library
- `langgraph` - The LangGraph framework
- `azure-kusto-data` - Kusto query client
- `azure-kusto-ingest` - Kusto streaming ingestion client

### Step 2: Verify Installation

```python
import langgraph
from langgraph.checkpoint.kusto.aio import AsyncKustoSaver

print("✓ Installation successful!")
```

## Setting Up Kusto

### Step 1: Set Environment Variables

Create a `.env` file in your project:

```bash
# .env
KUSTO_CLUSTER_URI=https://your-cluster.region.kusto.windows.net
KUSTO_DATABASE=langgraph
```

### Step 2: Create Database Tables

Download the `provision.kql` script from this repository and run it in your Kusto cluster:

```kql
// Creates these tables:
// - Checkpoints: Stores checkpoint data
// - CheckpointWrites: Stores pending writes
// - LatestCheckpoints: Materialized view for fast queries
```

You can run this in the [Azure Data Explorer Web UI](https://dataexplorer.azure.com/).

### Step 3: Grant Permissions

Your Azure identity needs these permissions:
- **Database Viewer** (for reading checkpoints)
- **Database Ingestor** (for writing checkpoints)

```kql
// In Kusto, run:
.add database YourDatabase viewers ('aaduser=your-email@domain.com')
.add database YourDatabase ingestors ('aaduser=your-email@domain.com')
```

## Your First Checkpoint

Let's save and load a simple checkpoint.

### Complete Working Example

```python
"""
tutorial_01_first_checkpoint.py

This example shows the basics of saving and loading checkpoints.
"""

import asyncio
import os
from langgraph.checkpoint.kusto.aio import AsyncKustoSaver

async def main():
    # Configuration
    cluster_uri = os.getenv("KUSTO_CLUSTER_URI", "https://your-cluster.eastus.kusto.windows.net")
    database = os.getenv("KUSTO_DATABASE", "langgraph")
    
    print("📊 Connecting to Kusto...")
    
    # Create checkpointer
    async with AsyncKustoSaver.from_connection_string(
        cluster_uri=cluster_uri,
        database=database,
    ) as checkpointer:
        # Verify tables exist
        await checkpointer.setup()
        print("✓ Connected successfully!")
        
        # Create a simple checkpoint
        from langgraph.checkpoint.base import Checkpoint
        
        checkpoint = Checkpoint(
            v=1,
            id="checkpoint-001",
            ts="2025-10-27T10:00:00",
            channel_values={"messages": ["Hello, world!"]},
            channel_versions={"messages": 1},
            versions_seen={"messages": {1}},
            pending_sends=[],
        )
        
        # Save it
        config = {"configurable": {"thread_id": "tutorial-thread-1"}}
        
        print("\n💾 Saving checkpoint...")
        await checkpointer.aput(
            config=config,
            checkpoint=checkpoint,
            metadata={"user": "tutorial", "step": 1},
            new_versions={"messages": 1},
        )
        
        # Flush to ensure it's written
        await checkpointer.flush()
        print("✓ Checkpoint saved!")
        
        # Wait a moment for streaming ingestion
        print("\n⏳ Waiting for data to be available...")
        await asyncio.sleep(2)
        
        # Load it back
        print("\n📖 Loading checkpoint...")
        loaded = await checkpointer.aget_tuple(config)
        
        if loaded:
            print(f"✓ Loaded checkpoint: {loaded.checkpoint['id']}")
            print(f"  Messages: {loaded.checkpoint['channel_values']['messages']}")
            print(f"  Metadata: {loaded.metadata}")
        else:
            print("⚠ Checkpoint not found yet (streaming ingestion delay)")

if __name__ == "__main__":
    asyncio.run(main())
```

**Run it:**
```bash
python tutorial_01_first_checkpoint.py
```

**Expected output:**
```
📊 Connecting to Kusto...
✓ Connected successfully!

💾 Saving checkpoint...
✓ Checkpoint saved!

⏳ Waiting for data to be available...

📖 Loading checkpoint...
✓ Loaded checkpoint: checkpoint-001
  Messages: ['Hello, world!']
  Metadata: {'user': 'tutorial', 'step': 1}
```

## Building a Simple Agent

Now let's build a real agent that uses checkpoints.

### Complete Working Example

```python
"""
tutorial_02_simple_agent.py

This example builds a simple chatbot agent that remembers conversation history.
"""

import asyncio
import os
from typing import Annotated
from typing_extensions import TypedDict

from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.kusto.aio import AsyncKustoSaver


# Define the state structure
class State(TypedDict):
    """State contains all the data that flows through our agent."""
    messages: Annotated[list, add_messages]
    step_count: int


def chatbot_node(state: State) -> dict:
    """
    Our chatbot logic - in a real app, this would call an LLM.
    For this tutorial, we'll just echo back with a simple response.
    """
    user_message = state["messages"][-1]
    
    # Simple echo response (replace with your LLM call)
    response = f"Echo: {user_message}"
    
    return {
        "messages": [response],
        "step_count": state.get("step_count", 0) + 1,
    }


async def main():
    # Configuration
    cluster_uri = os.getenv("KUSTO_CLUSTER_URI", "https://your-cluster.eastus.kusto.windows.net")
    database = os.getenv("KUSTO_DATABASE", "langgraph")
    
    print("🤖 Building chatbot with Kusto checkpoints...\n")
    
    # Create checkpointer
    async with AsyncKustoSaver.from_connection_string(
        cluster_uri=cluster_uri,
        database=database,
    ) as checkpointer:
        await checkpointer.setup()
        
        # Build the graph
        graph = StateGraph(State)
        graph.add_node("chatbot", chatbot_node)
        graph.add_edge(START, "chatbot")
        graph.add_edge("chatbot", END)
        
        # Compile with checkpointing
        app = graph.compile(checkpointer=checkpointer)
        
        # Thread ID identifies this conversation
        thread_id = "tutorial-conversation-1"
        config = {"configurable": {"thread_id": thread_id}}
        
        # Conversation 1: Initial messages
        print("👤 User: Hello!")
        result = await app.ainvoke(
            {"messages": ["Hello!"], "step_count": 0},
            config=config,
        )
        print(f"🤖 Bot: {result['messages'][-1]}")
        print(f"   (Step {result['step_count']})\n")
        
        await checkpointer.flush()
        await asyncio.sleep(1)  # Wait for streaming ingestion
        
        # Conversation 2: Continue the thread
        print("👤 User: How are you?")
        result = await app.ainvoke(
            {"messages": ["How are you?"]},
            config=config,
        )
        print(f"🤖 Bot: {result['messages'][-1]}")
        print(f"   (Step {result['step_count']})\n")
        
        await checkpointer.flush()
        await asyncio.sleep(1)
        
        # Show full history
        print("📜 Full conversation history:")
        checkpoint = await checkpointer.aget_tuple(config)
        if checkpoint:
            messages = checkpoint.checkpoint.get("channel_values", {}).get("messages", [])
            for i, msg in enumerate(messages, 1):
                print(f"   {i}. {msg}")
            print(f"\n   Total steps: {checkpoint.checkpoint.get('channel_values', {}).get('step_count', 0)}")

if __name__ == "__main__":
    asyncio.run(main())
```

**Run it:**
```bash
python tutorial_02_simple_agent.py
```

**Expected output:**
```
🤖 Building chatbot with Kusto checkpoints...

👤 User: Hello!
🤖 Bot: Echo: Hello!
   (Step 1)

👤 User: How are you?
🤖 Bot: Echo: How are you?
   (Step 2)

📜 Full conversation history:
   1. Hello!
   2. Echo: Hello!
   3. How are you?
   4. Echo: How are you?

   Total steps: 2
```

## Understanding Checkpoints

### What Gets Saved?

Every checkpoint contains:

1. **State Data**: Your agent's current state (messages, variables, etc.)
2. **Metadata**: Extra info like timestamps, user IDs, tags
3. **Version Info**: Tracks which channels changed
4. **Pending Writes**: Operations waiting to execute

### When Are Checkpoints Saved?

Automatically saved:
- ✅ Before each node executes
- ✅ After each node completes
- ✅ When you call `flush()`

### Streaming Ingestion

This library uses **streaming ingestion**:
- Data appears in Kusto within **<1 second** after flushing
- Good for interactive applications
- Automatic batching for efficiency

## Common Patterns

### Pattern 1: Multiple Conversations

Track different conversations with different thread IDs:

```python
# User 1's conversation
config_user1 = {"configurable": {"thread_id": "user-1-chat"}}
await app.ainvoke({"messages": ["Hi"]}, config=config_user1)

# User 2's conversation (completely separate)
config_user2 = {"configurable": {"thread_id": "user-2-chat"}}
await app.ainvoke({"messages": ["Hello"]}, config=config_user2)
```

### Pattern 2: Resume from Any Point

Load a specific checkpoint by ID:

```python
# Save with a specific checkpoint
config = {
    "configurable": {
        "thread_id": "my-thread",
        "checkpoint_id": "important-state-001",
    }
}
await app.ainvoke(state, config=config)

# Later, resume from that exact checkpoint
loaded = await checkpointer.aget_tuple(config)
```

### Pattern 3: Search with Metadata

Find checkpoints using metadata filters:

```python
# Save with metadata
await checkpointer.aput(
    config=config,
    checkpoint=checkpoint,
    metadata={"user_id": "user-123", "category": "support"},
    new_versions={},
)

# Search by metadata
async for checkpoint_tuple in checkpointer.alist(
    None,
    filter={"user_id": "user-123"},
):
    print(f"Found: {checkpoint_tuple.metadata}")
```

### Pattern 4: Cleanup Old Data

Delete old conversations:

```python
# Delete all checkpoints for a thread
await checkpointer.adelete_thread("old-thread-id")
```

### Pattern 5: Performance Tuning

Adjust batching for your workload:

```python
async with AsyncKustoSaver.from_connection_string(
    cluster_uri=cluster_uri,
    database=database,
    batch_size=100,        # Larger = better throughput
    flush_interval=30.0,   # Auto-flush every 30 seconds
) as checkpointer:
    # Your code here
    pass
```

## Troubleshooting

### Issue: "Data not appearing immediately"

**Solution**: Remember streaming ingestion has a small delay:

```python
await checkpointer.flush()
await asyncio.sleep(1)  # Wait 1 second for data availability
```

### Issue: "Table not found"

**Solution**: Run the `provision.kql` script first:

```bash
# Check tables exist
.show tables
```

You should see: `Checkpoints`, `CheckpointWrites`, `LatestCheckpoints`

### Issue: "Authorization failed"

**Solution**: Grant permissions in Kusto:

```kql
.add database langgraph viewers ('aaduser=your-email@domain.com')
.add database langgraph ingestors ('aaduser=your-email@domain.com')
```

### Issue: "Cannot import AsyncKustoSaver"

**Solution**: Verify installation:

```bash
pip list | grep langgraph-checkpoint-kusto
```

If missing:
```bash
pip install --upgrade langgraph-checkpoint-kusto
```

### Issue: "Slow queries"

**Solution**: The `LatestCheckpoints` materialized view optimizes "get latest" queries:

```python
# This is fast (uses materialized view)
latest = await checkpointer.aget_tuple({"configurable": {"thread_id": "my-thread"}})

# This is also fast (indexed by thread_id)
async for checkpoint in checkpointer.alist({"configurable": {"thread_id": "my-thread"}}):
    print(checkpoint.metadata)
```

## Next Steps

Now that you understand the basics:

1. **Add Real LLM Integration**: Replace the echo logic with actual LLM calls
2. **Add Error Handling**: Use try/except blocks around checkpoint operations
3. **Monitor Performance**: Check Kusto metrics in Azure Portal
4. **Scale Up**: Increase batch_size for high-volume workloads
5. **Add Analytics**: Query your checkpoint data with KQL for insights

### Example: Query Your Checkpoints

Use Kusto Query Language (KQL) to analyze your data:

```kql
// Find most active conversations
Checkpoints
| summarize CheckpointCount = count() by thread_id
| order by CheckpointCount desc
| take 10

// See checkpoint growth over time
Checkpoints
| summarize count() by bin(created_at, 1h)
| render timechart

// Find checkpoints with specific metadata
Checkpoints
| where metadata contains "user_id"
| project thread_id, checkpoint_id, metadata, created_at
```

## Complete Reference Code

Here's a production-ready example with error handling:

```python
"""
tutorial_03_production_ready.py

Production-ready example with proper error handling.
"""

import asyncio
import logging
import os
from typing import Annotated
from typing_extensions import TypedDict

from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.kusto.aio import AsyncKustoSaver
from azure.core.exceptions import AzureError

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class State(TypedDict):
    messages: Annotated[list, add_messages]


def chatbot_node(state: State) -> dict:
    """Simple chatbot node."""
    user_message = state["messages"][-1]
    response = f"Echo: {user_message}"
    return {"messages": [response]}


async def run_agent_with_checkpoints():
    """Run agent with proper error handling."""
    
    # Configuration
    cluster_uri = os.getenv("KUSTO_CLUSTER_URI")
    database = os.getenv("KUSTO_DATABASE", "langgraph")
    
    if not cluster_uri:
        logger.error("KUSTO_CLUSTER_URI environment variable not set")
        return
    
    try:
        # Create checkpointer
        async with AsyncKustoSaver.from_connection_string(
            cluster_uri=cluster_uri,
            database=database,
        ) as checkpointer:
            logger.info("Connected to Kusto")
            
            # Verify setup
            try:
                await checkpointer.setup()
                logger.info("Schema validated")
            except AzureError as e:
                logger.error(f"Schema validation failed: {e}")
                return
            
            # Build graph
            graph = StateGraph(State)
            graph.add_node("chatbot", chatbot_node)
            graph.add_edge(START, "chatbot")
            graph.add_edge("chatbot", END)
            app = graph.compile(checkpointer=checkpointer)
            
            # Run agent
            thread_id = "production-thread-1"
            config = {"configurable": {"thread_id": thread_id}}
            
            try:
                result = await app.ainvoke(
                    {"messages": ["Hello, production!"]},
                    config=config,
                )
                logger.info(f"Agent response: {result['messages'][-1]}")
                
                # Ensure data is persisted
                await checkpointer.flush()
                logger.info("Checkpoints flushed")
                
            except Exception as e:
                logger.error(f"Agent execution failed: {e}")
                raise
            
    except AzureError as e:
        logger.error(f"Azure connection failed: {e}")
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        raise


if __name__ == "__main__":
    asyncio.run(run_agent_with_checkpoints())
```

## Learn More

- **LangGraph Documentation**: https://langchain-ai.github.io/langgraph/
- **Kusto Query Language**: https://learn.microsoft.com/en-us/azure/data-explorer/kusto/query/
- **Azure Data Explorer**: https://learn.microsoft.com/en-us/azure/data-explorer/

## Need Help?

- 📖 Check the [README.md](./README.md) for detailed API reference
- 🐛 Report issues on [GitHub Issues](https://github.com/cosh/langgraph-kusto/issues)
- 💬 Ask questions in [LangChain Discord](https://discord.gg/langchain)

---

**Congratulations!** 🎉 You now know how to use LangGraph with Kusto checkpoints. Build amazing agents!
