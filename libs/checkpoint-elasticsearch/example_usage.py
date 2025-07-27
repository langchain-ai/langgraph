#!/usr/bin/env python3
"""Example usage of the Elasticsearch checkpointer.

This example shows how to use the ElasticsearchSaver and AsyncElasticsearchSaver
with a LangGraph agent.

Requirements:
1. Running Elasticsearch cluster (version 8.17+)
2. Set environment variables: ES_URL and ES_API_KEY

To run this example:
1. Start Elasticsearch: docker run -d -p 9200:9200 -e "discovery.type=single-node" -e "xpack.security.enabled=false" elasticsearch:8.17.0
2. Set ES_URL=http://localhost:9200 and ES_API_KEY=dummy (if security disabled)
3. Run: python example_usage.py
"""

import os
from typing import Annotated

from langchain_core.messages import AIMessage
from typing_extensions import TypedDict

from langgraph.checkpoint.elasticsearch.sync import ElasticsearchSaver
from langgraph.graph import StateGraph, add_messages


class State(TypedDict):
    messages: Annotated[list, add_messages]


def chatbot(state: State) -> State:
    """Simple chatbot that echoes back what the user said."""
    return {
        "messages": [
            AIMessage(content="Hello! You said: " + state["messages"][-1].content)
        ]
    }


def main():
    """Example usage of ElasticsearchSaver with LangGraph."""
    # Configure Elasticsearch connection
    # In production, use environment variables:
    # os.environ["ES_URL"] = "https://your-elasticsearch-cluster:9200"
    # os.environ["ES_API_KEY"] = "your-api-key"

    # For local testing with insecure Elasticsearch:
    if not os.environ.get("ES_URL"):
        print("ES_URL environment variable not set. Using default localhost:9200")
        os.environ["ES_URL"] = "http://localhost:9200"

    if not os.environ.get("ES_API_KEY"):
        print(
            "ES_API_KEY environment variable not set. Using dummy key for insecure ES"
        )
        os.environ["ES_API_KEY"] = "dummy"

    try:
        # Create the checkpointer
        checkpointer = ElasticsearchSaver(
            # es_url and api_key will be read from environment variables
            index_prefix="example_agent"  # Custom prefix for this application
        )

        print("✓ Elasticsearch checkpointer initialized successfully")

        # Create and compile the graph
        workflow = StateGraph(State)
        workflow.add_node("chatbot", chatbot)
        workflow.set_entry_point("chatbot")
        workflow.set_finish_point("chatbot")

        # Compile with checkpointer
        app = workflow.compile(checkpointer=checkpointer)

        print("✓ LangGraph compiled with Elasticsearch checkpointer")

        # Example interaction with thread persistence
        config = {"configurable": {"thread_id": "example-thread-123"}}

        # First interaction
        response1 = app.invoke(
            {"messages": [("user", "Hello, my name is Alice")]}, config=config
        )
        print(f"Response 1: {response1['messages'][-1].content}")

        # Second interaction - should remember context via checkpoints
        response2 = app.invoke(
            {"messages": [("user", "What's my name?")]}, config=config
        )
        print(f"Response 2: {response2['messages'][-1].content}")

        # List all checkpoints for this thread
        checkpoints = list(app.get_state_history(config))
        print(f"✓ Found {len(checkpoints)} checkpoints in thread")

        print("✓ Example completed successfully!")

    except Exception as e:
        print(f"✗ Error: {e}")
        print("\nTo run this example:")
        print("1. Start Elasticsearch:")
        print(
            "   docker run -d -p 9200:9200 -e 'discovery.type=single-node' -e 'xpack.security.enabled=false' elasticsearch:8.17.0"
        )
        print("2. Set environment variables:")
        print("   export ES_URL=http://localhost:9200")
        print("   export ES_API_KEY=dummy")
        print("3. Run the example:")
        print("   python example_usage.py")


if __name__ == "__main__":
    main()
