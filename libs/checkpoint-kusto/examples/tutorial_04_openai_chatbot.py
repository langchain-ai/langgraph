"""
tutorial_04_openai_chatbot.py

Real-world chatbot using OpenAI or Azure OpenAI with Kusto checkpointing.
This example demonstrates:
- Using actual LLM calls (OpenAI GPT or Azure OpenAI)
- Proper message handling with LangChain
- Conversation memory with checkpointing
- Support for both OpenAI and Azure OpenAI

Prerequisites:
1. Set KUSTO_CLUSTER_URI environment variable
2. Set KUSTO_DATABASE environment variable
3. For OpenAI: Set OPENAI_API_KEY
   OR
   For Azure OpenAI: Set AZURE_OPENAI_API_KEY, AZURE_OPENAI_ENDPOINT, AZURE_OPENAI_DEPLOYMENT
4. Run provision.kql to create tables
5. Install: pip install langchain-openai
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
        from langchain_openai import ChatOpenAI, AzureChatOpenAI
        from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
        return StateGraph, START, END, add_messages, AsyncKustoSaver, ChatOpenAI, AzureChatOpenAI, HumanMessage, AIMessage, SystemMessage
    except ImportError as e:
        print("❌ Missing dependencies!")
        print(f"   Error: {e}")
        print("\n📦 To install dependencies:")
        print("   cd libs\\checkpoint-kusto")
        print("   pip install -e .")
        print("   pip install langchain-openai")
        print("\nOr install manually:")
        print("   pip install langgraph langchain-openai langchain-core")
        print("   pip install azure-kusto-data azure-kusto-ingest azure-identity")
        print("   pip install aiohttp")
        print("\nFor detailed setup instructions, see: ../SETUP.md")
        sys.exit(1)


# Get dependencies
StateGraph, START, END, add_messages, AsyncKustoSaver, ChatOpenAI, AzureChatOpenAI, HumanMessage, AIMessage, SystemMessage = check_dependencies()


# Define the state structure
class State(TypedDict):
    """
    State contains all the data that flows through our agent.
    
    Attributes:
        messages: List of conversation messages (HumanMessage, AIMessage, etc.)
    """
    messages: Annotated[list, add_messages]


async def chatbot_node(state: State) -> dict:
    """
    Chatbot logic using OpenAI or Azure OpenAI.
    
    This node:
    1. Gets the conversation history from state
    2. Calls OpenAI's API (or Azure OpenAI)
    3. Returns the AI response
    
    Args:
        state: Current state with message history
        
    Returns:
        Updated state with AI response added
    """
    # Check which OpenAI service to use
    azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
    
    if azure_endpoint:
        # Use Azure OpenAI
        llm = AzureChatOpenAI(
            azure_endpoint=azure_endpoint,
            azure_deployment=os.getenv("AZURE_OPENAI_DEPLOYMENT", "gpt-4o-mini"),
            api_version="2024-08-01-preview",
            temperature=0.7,
        )
    else:
        # Use OpenAI
        llm = ChatOpenAI(
            model="gpt-4o-mini",  # Use gpt-4o, gpt-4o-mini, or gpt-3.5-turbo
            temperature=0.7,
        )
    
    # Get all messages from state (includes full conversation history)
    messages = state["messages"]
    
    # Call the LLM
    # The LLM sees the full conversation history automatically
    response = await llm.ainvoke(messages)
    
    # Return the response (it will be added to messages via add_messages)
    return {"messages": [response]}


async def main():
    """Run the OpenAI chatbot example."""
    # Configuration from environment
    cluster_uri = os.getenv(
        "KUSTO_CLUSTER_URI",
        "https://your-cluster.region.kusto.windows.net"
    )
    database = os.getenv("KUSTO_DATABASE", "langgraph")
    
    # Check which OpenAI service to use
    azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
    azure_key = os.getenv("AZURE_OPENAI_API_KEY")
    openai_key = os.getenv("OPENAI_API_KEY")
    
    # Validate API credentials
    if azure_endpoint:
        # Using Azure OpenAI
        if not azure_key:
            print("❌ AZURE_OPENAI_API_KEY environment variable is not set!")
            print("\n📝 To set your Azure OpenAI API key:")
            print('   PowerShell: $env:AZURE_OPENAI_API_KEY = "your-key"')
            print('   Bash: export AZURE_OPENAI_API_KEY="your-key"')
            print("\nAlso set:")
            print('   $env:AZURE_OPENAI_ENDPOINT = "https://your-resource.openai.azure.com/"')
            print('   $env:AZURE_OPENAI_DEPLOYMENT = "gpt-4o-mini"  # Your deployment name')
            print("\nGet your credentials from Azure Portal > Azure OpenAI > Keys and Endpoint")
            sys.exit(1)
        deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT", "gpt-4o-mini")
        print(f"🤖 Using Azure OpenAI")
        print(f"   Endpoint: {azure_endpoint}")
        print(f"   Deployment: {deployment}")
    else:
        # Using OpenAI
        if not openai_key:
            print("❌ OPENAI_API_KEY environment variable is not set!")
            print("\n📝 To set your OpenAI API key:")
            print('   PowerShell: $env:OPENAI_API_KEY = "sk-..."')
            print('   Bash: export OPENAI_API_KEY="sk-..."')
            print('   Windows CMD: set OPENAI_API_KEY=sk-...')
            print("\nGet your API key from: https://platform.openai.com/api-keys")
            print("\n💡 Using Azure OpenAI instead? Set:")
            print('   $env:AZURE_OPENAI_ENDPOINT = "https://your-resource.openai.azure.com/"')
            print('   $env:AZURE_OPENAI_API_KEY = "your-key"')
            print('   $env:AZURE_OPENAI_DEPLOYMENT = "gpt-4o-mini"')
            sys.exit(1)
        print(f"🤖 Using OpenAI")
        print(f"   Model: gpt-4o-mini")
    
    # Validate Kusto configuration
    if cluster_uri.startswith("https://your-cluster"):
        print("⚠️  Using default KUSTO_CLUSTER_URI!")
        print("   Set your actual cluster:")
        print('   PowerShell: $env:KUSTO_CLUSTER_URI = "https://your-cluster.region.kusto.windows.net"')
        print('   Bash: export KUSTO_CLUSTER_URI="https://your-cluster.region.kusto.windows.net"')
        print("\nFor detailed setup instructions, see: ../SETUP.md\n")
    
    print("\n🤖 Building chatbot with Kusto checkpoints...")
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
            # Using a unique thread ID for this example
            thread_id = "openai-demo-conversation"
            config = {"configurable": {"thread_id": thread_id}}
            
            # Optional: Add a system message to set the assistant's behavior
            # This is stored in the checkpoint and persists across runs
            print("=" * 70)
            print("💬 Setting up AI assistant personality...")
            
            system_msg = SystemMessage(
                content="You are a helpful AI assistant. Be concise and friendly."
            )
            
            # Initialize with system message
            result = await app.ainvoke(
                {"messages": [system_msg]},
                config=config,
            )
            print("✓ System message configured\n")
            
            # Flush to ensure system message is persisted
            await checkpointer.flush()
            await asyncio.sleep(0.5)
            
            # Check if this is a continuing conversation
            checkpoint = await checkpointer.aget_tuple(config)
            is_new_conversation = checkpoint is None or len(checkpoint.checkpoint.get("channel_values", {}).get("messages", [])) <= 1
            
            if not is_new_conversation:
                # This is a continuing conversation - test memory first
                print("=" * 70)
                print("💡 Detected existing conversation - Testing memory...")
                
                # Ask a more specific question that references the conversation history
                memory_question = (
                    "Looking at our conversation history above, what topics have we discussed so far? "
                    "Please list the main subjects we've talked about."
                )
                print(f"👤 User: {memory_question}")
                result = await app.ainvoke(
                    {"messages": [HumanMessage(content=memory_question)]},
                    config=config,
                )
                ai_response = result['messages'][-1]
                print(f"🤖 AI: {ai_response.content}\n")
                
                await checkpointer.flush()
                await asyncio.sleep(0.5)
                
                print("=" * 70)
                print("💡 Memory test complete! Now continuing with new questions...\n")
            
            # Message 1: First user question
            print("=" * 70)
            print("👤 User: What is LangGraph?")
            result = await app.ainvoke(
                {"messages": [HumanMessage(content="What is LangGraph?")]},
                config=config,
            )
            ai_response = result['messages'][-1]
            print(f"🤖 AI: {ai_response.content}\n")
            
            # Flush and wait
            await checkpointer.flush()
            await asyncio.sleep(0.5)
            
            # Message 2: Follow-up question (AI remembers context)
            print("=" * 70)
            print("👤 User: What are its main benefits?")
            result = await app.ainvoke(
                {"messages": [HumanMessage(content="What are its main benefits?")]},
                config=config,
            )
            ai_response = result['messages'][-1]
            print(f"🤖 AI: {ai_response.content}\n")
            
            await checkpointer.flush()
            await asyncio.sleep(0.5)
            
            # Message 3: Test memory within current session
            print("=" * 70)
            print("👤 User: Can you summarize what we just discussed?")
            result = await app.ainvoke(
                {"messages": [HumanMessage(content="Can you summarize what we just discussed?")]},
                config=config,
            )
            ai_response = result['messages'][-1]
            print(f"🤖 AI: {ai_response.content}\n")
            
            await checkpointer.flush()
            await asyncio.sleep(0.5)
            
            # Show full conversation history from checkpoint
            print("=" * 70)
            print("📜 Full conversation history (from Kusto checkpoint):")
            checkpoint = await checkpointer.aget_tuple(config)
            
            if checkpoint:
                messages = checkpoint.checkpoint.get("channel_values", {}).get("messages", [])
                
                print(f"\n   Total messages: {len(messages)}")
                print(f"   Thread ID: {thread_id}")
                print(f"   Checkpoint ID: {checkpoint.checkpoint['id']}\n")
                
                for i, msg in enumerate(messages, 1):
                    # Format based on message type
                    if isinstance(msg, SystemMessage):
                        print(f"   ⚙️  {i}. SYSTEM: {msg.content[:60]}...")
                    elif isinstance(msg, HumanMessage):
                        print(f"   👤 {i}. USER: {msg.content}")
                    elif isinstance(msg, AIMessage):
                        print(f"   🤖 {i}. AI: {msg.content[:80]}...")
                    else:
                        print(f"   📝 {i}. {type(msg).__name__}: {str(msg)[:60]}...")
                
                print("\n✨ Run this script again - the conversation will continue!")
                print("   The AI will remember everything from the checkpoint.")
                print(f"   Change thread_id to start a fresh conversation.")
            else:
                print("   ⚠ No checkpoint found")
    
    except Exception as e:
        print(f"\n❌ Error: {e}")
        print("\n🔍 Troubleshooting:")
        print("   1. Check OPENAI_API_KEY is set correctly")
        print("   2. Check if Azure CLI is logged in: az login")
        print("   3. Verify tables exist in Kusto (run provision.kql)")
        print("   4. Check environment variables are set correctly")
        print("   5. Verify you have OpenAI API credits")
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
        print("1. Check your OpenAI API key")
        print("2. Check your Kusto cluster URI and database name")
        print("3. Verify tables exist (run provision.kql)")
        print("4. Check Azure permissions (Database Viewer + Ingestor)")
        raise
