#!/usr/bin/env python3
"""
Test runner for Elasticsearch checkpointer.

This script provides comprehensive testing options:
1. Unit tests (mocked, no ES required)
2. Integration tests (requires running Elasticsearch)
3. Interactive demo with real checkpointing

Usage:
    python test_runner.py                    # Run unit tests only
    python test_runner.py --integration      # Run integration tests (needs ES)
    python test_runner.py --demo            # Interactive demo
    python test_runner.py --all             # Run everything
"""

import argparse
import asyncio
import os
import subprocess
import sys
import time
from typing import Annotated

try:
    from langchain_core.messages import AIMessage, HumanMessage
    from typing_extensions import TypedDict

    from langgraph.checkpoint.elasticsearch.aio import AsyncElasticsearchSaver
    from langgraph.checkpoint.elasticsearch.sync import ElasticsearchSaver
    from langgraph.graph import StateGraph, add_messages
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("Make sure you're in the right directory and have dependencies installed:")
    print("  cd libs/checkpoint-elasticsearch")
    print("  uv sync")
    sys.exit(1)


def run_command(cmd: str, description: str) -> bool:
    """Run a command and return success status."""
    print(f"🔄 {description}...")
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        if result.returncode == 0:
            print(f"✅ {description} - PASSED")
            return True
        else:
            print(f"❌ {description} - FAILED")
            print(f"Error: {result.stderr}")
            return False
    except Exception as e:
        print(f"❌ {description} - ERROR: {e}")
        return False


def run_unit_tests() -> bool:
    """Run unit tests with mocked Elasticsearch."""
    return run_command(
        "uv run pytest tests/test_unit.py -v", "Running unit tests (mocked ES)"
    )


def check_elasticsearch() -> bool:
    """Check if Elasticsearch is running and accessible."""
    es_url = os.environ.get("ES_URL", "http://localhost:9200")
    try:
        import requests

        response = requests.get(es_url, timeout=5)
        if response.status_code == 200:
            print(f"✅ Elasticsearch is running at {es_url}")
            return True
        else:
            print(f"❌ Elasticsearch returned status {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Cannot connect to Elasticsearch at {es_url}: {e}")
        return False


def setup_elasticsearch_env():
    """Set up environment variables for Elasticsearch."""
    if not os.environ.get("ES_URL"):
        os.environ["ES_URL"] = "http://localhost:9200"
        print(f"📝 Set ES_URL={os.environ['ES_URL']}")

    if not os.environ.get("ES_API_KEY"):
        os.environ["ES_API_KEY"] = "dummy"
        print(f"📝 Set ES_API_KEY={os.environ['ES_API_KEY']}")


def test_basic_functionality() -> bool:
    """Test basic checkpointer functionality."""
    try:
        print("🔄 Testing basic ElasticsearchSaver functionality...")

        # Test sync version
        saver = ElasticsearchSaver(index_prefix="test_basic")
        print("✅ Sync ElasticsearchSaver created successfully")

        # Test document ID generation
        doc_id = saver._get_document_id("thread1", "ns1", "checkpoint1")
        assert doc_id == "thread1#ns1#checkpoint1"
        print("✅ Document ID generation works")

        return True
    except Exception as e:
        print(f"❌ Basic functionality test failed: {e}")
        return False


async def test_async_functionality() -> bool:
    """Test async checkpointer functionality."""
    try:
        print("🔄 Testing async AsyncElasticsearchSaver functionality...")

        saver = AsyncElasticsearchSaver(index_prefix="test_async")
        print("✅ Async ElasticsearchSaver created successfully")

        # Test ensure indices
        await saver._ensure_indices()
        print("✅ Indices creation works")

        # Close the client
        await saver.aclose()
        print("✅ Async client closed successfully")

        return True
    except Exception as e:
        print(f"❌ Async functionality test failed: {e}")
        return False


class State(TypedDict):
    messages: Annotated[list, add_messages]
    count: int


def chatbot_node(state: State) -> State:
    """Simple chatbot that counts messages."""
    count = state.get("count", 0) + 1
    last_message = state["messages"][-1] if state["messages"] else None

    if isinstance(last_message, HumanMessage):
        response = f"[Message #{count}] Hello! You said: {last_message.content}"
    else:
        response = f"[Message #{count}] Hello! How can I help you?"

    return {"messages": [AIMessage(content=response)], "count": count}


def run_interactive_demo() -> bool:
    """Run an interactive demo with a simple chatbot."""
    try:
        print("🔄 Setting up interactive demo...")

        # Create checkpointer
        checkpointer = ElasticsearchSaver(index_prefix="demo_chat")
        print("✅ Checkpointer created")

        # Create graph
        workflow = StateGraph(State)
        workflow.add_node("chatbot", chatbot_node)
        workflow.set_entry_point("chatbot")
        workflow.set_finish_point("chatbot")

        app = workflow.compile(checkpointer=checkpointer)
        print("✅ LangGraph compiled with checkpointer")

        thread_id = f"demo-{int(time.time())}"
        config = {"configurable": {"thread_id": thread_id}}

        print(f"\n🤖 Interactive Demo (Thread: {thread_id})")
        print("Type 'quit' to exit, 'history' to see checkpoints")
        print("-" * 50)

        while True:
            user_input = input("You: ").strip()

            if user_input.lower() == "quit":
                break
            elif user_input.lower() == "history":
                try:
                    history = list(checkpointer.list(config))
                    print(f"📚 Found {len(history)} checkpoints:")
                    for i, checkpoint in enumerate(history[:5]):  # Show last 5
                        state = checkpoint.checkpoint
                        count = state.get("channel_values", {}).get("count", 0)
                        print(
                            f"  {i + 1}. Checkpoint {checkpoint.config['configurable']['checkpoint_id'][:8]}... (Count: {count})"
                        )
                except Exception as e:
                    print(f"❌ Error getting history: {e}")
                continue
            elif not user_input:
                continue

            try:
                # Invoke the graph
                result = app.invoke(
                    {"messages": [HumanMessage(content=user_input)]}, config=config
                )

                bot_response = result["messages"][-1].content
                print(f"Bot: {bot_response}")
                print(f"📊 Total messages in thread: {result.get('count', 0)}")

            except Exception as e:
                print(f"❌ Error in conversation: {e}")

        print("\n✅ Demo completed successfully!")
        return True

    except Exception as e:
        print(f"❌ Interactive demo failed: {e}")
        return False


def run_integration_tests() -> bool:
    """Run integration tests with real Elasticsearch."""
    if not check_elasticsearch():
        print("\n❌ Integration tests require running Elasticsearch.")
        print("Start Elasticsearch with:")
        print(
            "  docker run -d -p 9200:9200 -e 'discovery.type=single-node' -e 'xpack.security.enabled=false' elasticsearch:8.17.0"
        )
        return False

    success = True

    # Test basic functionality
    success &= test_basic_functionality()

    # Test async functionality
    success &= asyncio.run(test_async_functionality())

    return success


def main():
    """Main test runner."""
    parser = argparse.ArgumentParser(description="Test the Elasticsearch checkpointer")
    parser.add_argument("--unit", action="store_true", help="Run unit tests only")
    parser.add_argument(
        "--integration", action="store_true", help="Run integration tests"
    )
    parser.add_argument("--demo", action="store_true", help="Run interactive demo")
    parser.add_argument("--all", action="store_true", help="Run all tests and demo")

    args = parser.parse_args()

    # Set up environment
    setup_elasticsearch_env()

    print("🚀 Elasticsearch Checkpointer Test Runner")
    print("=" * 50)

    results = []

    # Default to unit tests if no specific option
    if not any([args.unit, args.integration, args.demo, args.all]):
        args.unit = True

    if args.unit or args.all:
        print("\n📋 UNIT TESTS")
        print("-" * 20)
        results.append(("Unit Tests", run_unit_tests()))

    if args.integration or args.all:
        print("\n🔗 INTEGRATION TESTS")
        print("-" * 20)
        results.append(("Integration Tests", run_integration_tests()))

    if args.demo or args.all:
        print("\n🎮 INTERACTIVE DEMO")
        print("-" * 20)
        results.append(("Interactive Demo", run_interactive_demo()))

    # Summary
    print("\n" + "=" * 50)
    print("📊 TEST SUMMARY")
    print("-" * 20)

    all_passed = True
    for test_name, passed in results:
        status = "✅ PASSED" if passed else "❌ FAILED"
        print(f"{test_name}: {status}")
        all_passed &= passed

    if all_passed:
        print("\n🎉 All tests passed!")
        return 0
    else:
        print("\n💥 Some tests failed!")
        return 1


if __name__ == "__main__":
    sys.exit(main())
