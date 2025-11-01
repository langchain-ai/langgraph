"""Example usage of the Kusto checkpointer."""

import asyncio
import os
from typing import TypedDict

from langgraph.graph import StateGraph

from langgraph.checkpoint.kusto.aio import AsyncKustoSaver


# Define state
class State(TypedDict):
    """Simple state for demonstration."""

    messages: list[str]
    count: int


# Define nodes
def node_1(state: State) -> State:
    """First node in the graph."""
    return {
        "messages": state["messages"] + ["Node 1 executed"],
        "count": state["count"] + 1,
    }


def node_2(state: State) -> State:
    """Second node in the graph."""
    return {
        "messages": state["messages"] + ["Node 2 executed"],
        "count": state["count"] + 1,
    }


async def main() -> None:
    """Run the example."""
    # Get Kusto connection details from environment
    cluster_uri = os.getenv(
        "KUSTO_CLUSTER_URI",
        "https://your-cluster.region.kusto.windows.net",
    )
    database = os.getenv("KUSTO_DATABASE", "langgraph")

    print(f"Connecting to Kusto cluster: {cluster_uri}")
    print(f"Database: {database}")

    # Create checkpointer with streaming ingestion
    async with AsyncKustoSaver.from_connection_string(
        cluster_uri=cluster_uri,
        database=database,
        batch_size=10,
    ) as checkpointer:
        # Validate schema
        print("\nValidating Kusto schema...")
        await checkpointer.setup()
        print("✓ Schema validation successful")

        # Create graph
        graph = StateGraph(State)
        graph.add_node("node_1", node_1)
        graph.add_node("node_2", node_2)
        graph.add_edge("node_1", "node_2")
        graph.set_entry_point("node_1")
        graph.set_finish_point("node_2")

        # Compile with checkpointer
        app = graph.compile(checkpointer=checkpointer)

        # Run with checkpointing
        print("\nRunning graph with checkpointing...")
        config = {"configurable": {"thread_id": "example-thread-1"}}

        initial_state = {"messages": [], "count": 0}
        result = await app.ainvoke(initial_state, config)

        print("\n✓ Graph execution complete")
        print(f"Final state: {result}")

        # Flush pending writes
        print("\nFlushing checkpoints to Kusto...")
        await checkpointer.flush()
        print("✓ Flush complete")

        # Retrieve checkpoint
        print("\nRetrieving latest checkpoint...")
        checkpoint_tuple = await checkpointer.aget_tuple(config)

        if checkpoint_tuple:
            print(f"✓ Found checkpoint: {checkpoint_tuple.checkpoint['id']}")
            print(
                f"  Thread ID: {checkpoint_tuple.config['configurable']['thread_id']}"
            )
            print(f"  Metadata: {checkpoint_tuple.metadata}")
        else:
            print("⚠ No checkpoint found (streaming ingestion delay possible)")

        # List all checkpoints for this thread
        print("\nListing all checkpoints for thread...")
        checkpoint_count = 0
        async for cp in checkpointer.alist(config, limit=5):
            checkpoint_count += 1
            print(f"  - Checkpoint {cp.checkpoint['id']}")

        print(f"✓ Found {checkpoint_count} checkpoints")

        # Clean up
        print("\nCleaning up test data...")
        await checkpointer.adelete_thread("example-thread-1")
        print("✓ Thread deleted")

    print("\n✅ Example complete!")


if __name__ == "__main__":
    asyncio.run(main())
