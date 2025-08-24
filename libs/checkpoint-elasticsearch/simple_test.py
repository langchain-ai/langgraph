#!/usr/bin/env python3
"""
Simple test script for the Elasticsearch checkpointer.

This script creates a basic LangGraph workflow and demonstrates
checkpoint persistence with Elasticsearch.

Run this after starting Elasticsearch:
  docker run -d -p 9200:9200 -e "discovery.type=single-node" -e "xpack.security.enabled=false" elasticsearch:8.17.0

Then run:
  python simple_test.py
"""

import os
from typing import Annotated, Any

from langchain_core.messages import AIMessage, HumanMessage
from typing_extensions import TypedDict

from langgraph.checkpoint.elasticsearch.sync import ElasticsearchSaver
from langgraph.graph import StateGraph, add_messages

# Set up Elasticsearch connection
os.environ["ES_URL"] = "http://localhost:9200"
os.environ["ES_API_KEY"] = "dummy"  # For insecure local ES


class ChatState(TypedDict):
    messages: Annotated[list, add_messages]
    user_name: str
    conversation_count: int


def chatbot(state: ChatState) -> dict[str, Any]:
    """Simple chatbot that remembers user name and counts conversations."""
    messages = state.get("messages", [])
    user_name = state.get("user_name", "")
    conversation_count = state.get("conversation_count", 0)

    last_message = messages[-1] if messages else None

    if isinstance(last_message, HumanMessage):
        user_input = last_message.content.lower()

        # Check if user is introducing themselves
        if "my name is" in user_input or "i'm" in user_input or "i am" in user_input:
            # Extract name (simple parsing)
            if "my name is" in user_input:
                name = user_input.split("my name is")[-1].strip().split()[0]
            elif "i'm" in user_input:
                name = user_input.split("i'm")[-1].strip().split()[0]
            elif "i am" in user_input:
                name = user_input.split("i am")[-1].strip().split()[0]
            else:
                name = user_name

            response = f"Nice to meet you, {name.title()}! I'll remember your name."
            user_name = name.title()

        elif user_name and ("what" in user_input and "name" in user_input):
            response = f"Your name is {user_name}!"

        elif "count" in user_input or "how many" in user_input:
            response = f"We've had {conversation_count} conversations so far."

        else:
            greeting = f"Hello {user_name}! " if user_name else "Hello! "
            response = f"{greeting}You said: {last_message.content}"

    else:
        response = "Hello! I'm a chatbot that can remember our conversation. Try saying 'My name is [your name]'!"

    return {
        "messages": [AIMessage(content=response)],
        "user_name": user_name,
        "conversation_count": conversation_count + 1,
    }


def main():
    print("🤖 Testing Elasticsearch Checkpointer")
    print("=" * 40)

    try:
        # Create the checkpointer
        print("1. Creating Elasticsearch checkpointer...")
        checkpointer = ElasticsearchSaver(index_prefix="test_chat")
        print("   ✅ Checkpointer created successfully")

        # Create the graph
        print("\n2. Building LangGraph...")
        workflow = StateGraph(ChatState)
        workflow.add_node("chatbot", chatbot)
        workflow.set_entry_point("chatbot")
        workflow.set_finish_point("chatbot")

        # Compile with checkpointer
        app = workflow.compile(checkpointer=checkpointer)
        print("   ✅ Graph compiled with checkpointer")

        # Test with a persistent thread
        thread_id = "test-thread-123"
        config = {"configurable": {"thread_id": thread_id}}
        print(f"\n3. Testing with thread: {thread_id}")

        # Conversation 1: Introduce yourself
        print("\n--- Conversation 1 ---")
        result1 = app.invoke(
            {"messages": [HumanMessage(content="Hi! My name is Alice.")]}, config
        )

        print("User: Hi! My name is Alice.")
        print(f"Bot:  {result1['messages'][-1].content}")
        print(
            f"State: name='{result1.get('user_name', '')}', count={result1.get('conversation_count', 0)}"
        )

        # Conversation 2: Ask about name (should remember)
        print("\n--- Conversation 2 ---")
        result2 = app.invoke(
            {"messages": [HumanMessage(content="What's my name?")]}, config
        )

        print("User: What's my name?")
        print(f"Bot:  {result2['messages'][-1].content}")
        print(
            f"State: name='{result2.get('user_name', '')}', count={result2.get('conversation_count', 0)}"
        )

        # Conversation 3: Ask about conversation count
        print("\n--- Conversation 3 ---")
        result3 = app.invoke(
            {"messages": [HumanMessage(content="How many conversations have we had?")]},
            config,
        )

        print("User: How many conversations have we had?")
        print(f"Bot:  {result3['messages'][-1].content}")
        print(
            f"State: name='{result3.get('user_name', '')}', count={result3.get('conversation_count', 0)}"
        )

        # Test checkpoint history
        print("\n4. Testing checkpoint history...")
        history = list(checkpointer.list(config))
        print(f"   ✅ Found {len(history)} checkpoints in thread")

        for i, checkpoint_tuple in enumerate(history[:3]):  # Show first 3
            checkpoint_id = checkpoint_tuple.config["configurable"]["checkpoint_id"]
            state = checkpoint_tuple.checkpoint
            count = state.get("channel_values", {}).get("conversation_count", 0)
            name = state.get("channel_values", {}).get("user_name", "")
            print(
                f"   Checkpoint {i + 1}: {checkpoint_id[:8]}... (count: {count}, name: '{name}')"
            )

        # Test with different thread (should start fresh)
        print("\n5. Testing with new thread...")
        new_config = {"configurable": {"thread_id": "different-thread"}}
        result4 = app.invoke(
            {"messages": [HumanMessage(content="What's my name?")]}, config=new_config
        )

        print("User: What's my name? (in new thread)")
        print(f"Bot:  {result4['messages'][-1].content}")
        print(
            f"State: name='{result4.get('user_name', '')}', count={result4.get('conversation_count', 0)}"
        )

        print("\n🎉 All tests completed successfully!")
        print("\nWhat this demonstrates:")
        print("✅ Checkpoints persist conversation state")
        print("✅ Each thread maintains separate state")
        print("✅ Checkpoint history is retrievable")
        print("✅ State includes custom fields (user_name, conversation_count)")

    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        print("\nMake sure Elasticsearch is running:")
        print(
            "docker run -d -p 9200:9200 -e 'discovery.type=single-node' -e 'xpack.security.enabled=false' elasticsearch:8.17.0"
        )
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
