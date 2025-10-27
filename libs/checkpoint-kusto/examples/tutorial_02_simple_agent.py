"""
tutorial_02_simple_agent.py

This example builds a simple chatbot agent that remembers conversation history
using Kusto checkpoints.

Prerequisites:
1. Set KUSTO_CLUSTER_URI environment variable
2. Set KUSTO_DATABASE environment variable
3. Run provision.kql to create tables
"""

import asyncio
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
        return StateGraph, START, END, add_messages, AsyncKustoSaver
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
StateGraph, START, END, add_messages, AsyncKustoSaver = check_dependencies()


# Define the state structure
class State(TypedDict):
    """
    State contains all the data that flows through our agent.
    
    Attributes:
        messages: List of messages in the conversation (automatically merged)
        step_count: Number of steps executed
    """
    messages: Annotated[list, add_messages]
    step_count: int


def chatbot_node(state: State) -> dict:
    """
    Our chatbot logic.
    
    In a real application, this would call an LLM (like OpenAI GPT).
    For this tutorial, we'll just echo back with a simple response.
    
    Args:
        state: Current state with messages and step count
        
    Returns:
        Updated state with new message and incremented step count
    """
    user_message = state["messages"][-1]
    
    # Simple echo response (replace with your LLM call in production)
    response = f"Echo: {user_message}"
    
    return {
        "messages": [response],
        "step_count": state.get("step_count", 0) + 1,
    }


async def main():
    """Run the chatbot example."""
    # Configuration from environment
    cluster_uri = os.getenv(
        "KUSTO_CLUSTER_URI")
    database = os.getenv("KUSTO_DATABASE", "langgraph")
    
    # Validate configuration
    if cluster_uri.startswith("https://your-cluster"):
        print("⚠️  Using default KUSTO_CLUSTER_URI!")
        print("   Set your actual cluster:")
        print('   PowerShell: $env:KUSTO_CLUSTER_URI = "https://your-cluster.region.kusto.windows.net"')
        print('   Bash: export KUSTO_CLUSTER_URI="https://your-cluster.region.kusto.windows.net"')
        print("\nFor detailed setup instructions, see: ../SETUP.md\n")
    
    print("🤖 Building chatbot with Kusto checkpoints...")
    print(f"   Cluster: {cluster_uri}")
    print(f"   Database: {database}\n")
    
    try:
        # Create checkpointer - keep context open for entire usage
        async with AsyncKustoSaver.from_connection_string(
            cluster_uri=cluster_uri,
            database=database,
        ) as checkpointer:
            # Verify setup
            await checkpointer.setup()
            print("✓ Connected to Kusto\n")
            
            # Build the graph
            graph = StateGraph(State)
            graph.add_node("chatbot", chatbot_node)
            graph.add_edge(START, "chatbot")
            graph.add_edge("chatbot", END)
            
            # Compile with checkpointing enabled
            app = graph.compile(checkpointer=checkpointer)
            print("✓ Agent compiled with checkpointing\n")
            
            # Thread ID identifies this conversation
            # Different thread IDs = different conversations
            thread_id = "tutorial-conversation-1"
            config = {"configurable": {"thread_id": thread_id}}
            
            # Message 1: Start conversation
            print("=" * 50)
            print("👤 User: Hello!")
            result = await app.ainvoke(
                {"messages": ["Hello!"], "step_count": 0},
                config=config,
            )
            print(f"🤖 Bot: {result['messages'][-1]}")
            print(f"   (Step {result['step_count']})")
            
            # Flush and wait for streaming ingestion
            await checkpointer.flush()
            await asyncio.sleep(1)
            
            # Message 2: Continue conversation
            print("\n" + "=" * 50)
            print("👤 User: How are you?")
            result = await app.ainvoke(
                {"messages": ["How are you?"]},
                config=config,
            )
            print(f"🤖 Bot: {result['messages'][-1]}")
            print(f"   (Step {result['step_count']})")
            
            await checkpointer.flush()
            await asyncio.sleep(1)
            
            # Message 3: Ask about previous messages
            print("\n" + "=" * 50)
            print("👤 User: What did I say first?")
            result = await app.ainvoke(
                {"messages": ["What did I say first?"]},
                config=config,
            )
            print(f"🤖 Bot: {result['messages'][-1]}")
            print(f"   (Step {result['step_count']})")
            
            await checkpointer.flush()
            await asyncio.sleep(1)
            
            # Show full conversation history from checkpoint
            print("\n" + "=" * 50)
            print("📜 Full conversation history (from checkpoint):")
            checkpoint = await checkpointer.aget_tuple(config)
            
            if checkpoint:
                messages = checkpoint.checkpoint.get("channel_values", {}).get("messages", [])
                step_count = checkpoint.checkpoint.get("channel_values", {}).get("step_count", 0)
                
                for i, msg in enumerate(messages, 1):
                    emoji = "👤" if i % 2 == 1 else "🤖"
                    print(f"   {emoji} {i}. {msg}")
                
                print(f"\n   Total steps: {step_count}")
                print(f"   Thread ID: {thread_id}")
                print(f"   Checkpoint ID: {checkpoint.checkpoint['id']}")
                
                print("\n✨ Try running this again - it will create a new conversation!")
                print("   Each thread_id creates a separate conversation history.")
            else:
                print("   ⚠ No checkpoint found yet")
    
    except Exception as e:
        print(f"\n❌ Error: {e}")
        print("\n🔍 Troubleshooting:")
        print("   1. Check if Azure CLI is logged in: az login")
        print("   2. Verify tables exist in Kusto (run provision.kql)")
        print("   3. Check environment variables are set correctly")
        print("\nFor detailed help, see: ../SETUP.md")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n\n👋 Cancelled by user")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        print("\nTroubleshooting tips:")
        print("1. Check your Kusto cluster URI and database name")
        print("2. Verify tables exist (run provision.kql)")
        print("3. Check Azure permissions (Database Viewer + Ingestor)")
        raise
