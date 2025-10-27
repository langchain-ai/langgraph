"""
tutorial_03_production_ready.py

Production-ready example with proper error handling, logging, and best practices.

Prerequisites:
1. Set KUSTO_CLUSTER_URI environment variable
2. Set KUSTO_DATABASE environment variable
3. Run provision.kql to create tables
"""

import asyncio
import logging
import os
import sys
from typing import Annotated
from typing_extensions import TypedDict


def check_dependencies():
    """Check if required dependencies are installed."""
    try:
        from langgraph.graph import StateGraph, START, END
        from langgraph.graph.message import add_messages
        from langgraph.checkpoint.kusto.aio import AsyncKustoSaver
        from azure.core.exceptions import AzureError
        return StateGraph, START, END, add_messages, AsyncKustoSaver, AzureError
    except ImportError as e:
        print("❌ Missing dependencies!")
        print(f"   Error: {e}")
        print("\n📦 To install dependencies:")
        print("   cd libs\\checkpoint-kusto")
        print("   pip install -e .")
        print("\nOr install manually:")
        print("   pip install langgraph")
        print("   pip install azure-kusto-data azure-kusto-ingest azure-identity")
        print("   pip install aiohttp")
        print("\nFor detailed setup instructions, see: ../SETUP.md")
        sys.exit(1)


# Get dependencies
StateGraph, START, END, add_messages, AsyncKustoSaver, AzureError = check_dependencies()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
    ]
)
logger = logging.getLogger(__name__)


class State(TypedDict):
    """Agent state definition."""
    messages: Annotated[list, add_messages]
    error_count: int


def chatbot_node(state: State) -> dict:
    """
    Chatbot node with error handling.
    
    In production, this would call your LLM with proper error handling.
    """
    try:
        user_message = state["messages"][-1]
        
        # Simulate LLM call (replace with actual LLM in production)
        response = f"Echo: {user_message}"
        
        return {
            "messages": [response],
            "error_count": state.get("error_count", 0),
        }
    except Exception as e:
        logger.error(f"Error in chatbot node: {e}")
        return {
            "messages": ["Sorry, I encountered an error."],
            "error_count": state.get("error_count", 0) + 1,
        }


async def run_agent_with_checkpoints():
    """
    Run agent with proper error handling and resource management.
    
    Returns:
        bool: True if successful, False otherwise
    """
    
    # Configuration with validation
    cluster_uri = os.getenv("KUSTO_CLUSTER_URI")
    database = os.getenv("KUSTO_DATABASE", "langgraph")
    
    if not cluster_uri:
        logger.error("KUSTO_CLUSTER_URI environment variable not set")
        logger.info("Set it with: export KUSTO_CLUSTER_URI=https://your-cluster.region.kusto.windows.net")
        return False
    
    logger.info(f"Connecting to Kusto cluster: {cluster_uri}")
    logger.info(f"Database: {database}")
    
    try:
        # Create checkpointer with context manager for automatic cleanup
        async with AsyncKustoSaver.from_connection_string(
            cluster_uri=cluster_uri,
            database=database,
            batch_size=10,  # Smaller batch for faster testing
            flush_interval=30.0,
        ) as checkpointer:
            logger.info("✓ Connected to Kusto")
            
            # Verify setup with error handling
            try:
                await checkpointer.setup()
                logger.info("✓ Schema validated")
            except AzureError as e:
                logger.error(f"Schema validation failed: {e}")
                logger.info("Make sure you've run provision.kql to create tables")
                return False
            
            # Build graph
            graph = StateGraph(State)
            graph.add_node("chatbot", chatbot_node)
            graph.add_edge(START, "chatbot")
            graph.add_edge("chatbot", END)
            
            # Compile with checkpointing
            app = graph.compile(checkpointer=checkpointer)
            logger.info("✓ Agent compiled with checkpointing")
            
            # Run agent with error handling
            thread_id = "production-thread-1"
            config = {"configurable": {"thread_id": thread_id}}
            
            try:
                logger.info(f"Starting conversation in thread: {thread_id}")
                
                # First message
                result = await app.ainvoke(
                    {"messages": ["Hello, production!"], "error_count": 0},
                    config=config,
                )
                logger.info(f"Agent response: {result['messages'][-1]}")
                
                # Ensure data is persisted
                await checkpointer.flush()
                logger.info("✓ Checkpoints flushed to Kusto")
                
                # Wait for streaming ingestion
                await asyncio.sleep(1)
                
                # Verify checkpoint was saved
                loaded = await checkpointer.aget_tuple(config)
                if loaded:
                    logger.info("✓ Checkpoint verified in Kusto")
                    logger.info(f"  Checkpoint ID: {loaded.checkpoint['id']}")
                    logger.info(f"  Message count: {len(loaded.checkpoint.get('channel_values', {}).get('messages', []))}")
                else:
                    logger.warning("Checkpoint not found yet (streaming delay)")
                
                # Second message to show conversation continuity
                result = await app.ainvoke(
                    {"messages": ["How are you?"]},
                    config=config,
                )
                logger.info(f"Agent response: {result['messages'][-1]}")
                
                await checkpointer.flush()
                await asyncio.sleep(1)
                
                # Show final state
                final_checkpoint = await checkpointer.aget_tuple(config)
                if final_checkpoint:
                    messages = final_checkpoint.checkpoint.get("channel_values", {}).get("messages", [])
                    logger.info(f"✓ Conversation has {len(messages)} messages")
                
                logger.info("🎉 Production example completed successfully!")
                return True
                
            except Exception as e:
                logger.error(f"Agent execution failed: {e}")
                raise
            
    except AzureError as e:
        logger.error(f"Azure connection failed: {e}")
        logger.info("Check your Azure credentials and network connectivity")
        return False
    except Exception as e:
        logger.error(f"Unexpected error: {e}", exc_info=True)
        return False


async def cleanup_old_threads(checkpointer: AsyncKustoSaver, thread_ids: list[str]):
    """
    Example: Clean up old conversation threads.
    
    Args:
        checkpointer: The Kusto checkpointer instance
        thread_ids: List of thread IDs to delete
    """
    logger.info(f"Cleaning up {len(thread_ids)} threads...")
    
    for thread_id in thread_ids:
        try:
            await checkpointer.adelete_thread(thread_id)
            logger.info(f"✓ Deleted thread: {thread_id}")
        except Exception as e:
            logger.error(f"Failed to delete thread {thread_id}: {e}")
    
    logger.info("Cleanup completed")


async def main():
    """Main entry point with overall error handling."""
    try:
        success = await run_agent_with_checkpoints()
        
        if success:
            logger.info("\n" + "=" * 60)
            logger.info("Production example completed successfully!")
            logger.info("=" * 60)
            logger.info("\nNext steps:")
            logger.info("1. Replace the echo logic with real LLM calls")
            logger.info("2. Add your business logic to the agent")
            logger.info("3. Monitor performance in Azure Portal")
            logger.info("4. Query your data with KQL for analytics")
            return 0
        else:
            logger.error("Production example failed - check logs above")
            return 1
            
    except KeyboardInterrupt:
        logger.info("\nCancelled by user")
        return 130
    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
