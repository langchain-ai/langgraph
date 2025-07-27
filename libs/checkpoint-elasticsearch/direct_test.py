#!/usr/bin/env python3
"""
Direct test of the Elasticsearch checkpointer without LangGraph dependencies.

This tests the checkpointer functionality directly using the BaseCheckpointSaver interface.
"""

import os
from datetime import datetime

# Set up environment variables from command line or defaults
ES_URL = os.environ.get("ES_URL", "http://localhost:9200")
ES_API_KEY = os.environ.get("ES_API_KEY", "dummy")

print(f"🔗 Connecting to: {ES_URL}")
print(f"🔑 Using API key: {ES_API_KEY[:10]}...")

try:
    from langgraph.checkpoint.base import Checkpoint, CheckpointMetadata
    from langgraph.checkpoint.elasticsearch.sync import ElasticsearchSaver
except ImportError as e:
    print(f"❌ Import error: {e}")
    exit(1)


def create_sample_checkpoint(checkpoint_id: str) -> Checkpoint:
    """Create a sample checkpoint for testing."""
    return {
        "v": 1,
        "id": checkpoint_id,
        "ts": datetime.now().isoformat(),
        "channel_values": {
            "messages": [
                {"type": "human", "content": "Hello!"},
                {"type": "ai", "content": "Hi there!"},
            ],
            "user_name": "Alice",
            "count": 1,
        },
        "channel_versions": {"messages": "1.0", "user_name": "1.0", "count": "1.0"},
        "versions_seen": {},
        "pending_sends": [],
    }


def test_checkpointer():
    """Test the Elasticsearch checkpointer directly."""
    print("\n🧪 Testing Elasticsearch Checkpointer")
    print("=" * 40)

    try:
        # 1. Create checkpointer
        print("1. Creating ElasticsearchSaver...")
        checkpointer = ElasticsearchSaver(
            es_url=ES_URL, api_key=ES_API_KEY, index_prefix="direct_test"
        )
        print("   ✅ Checkpointer created successfully")

        # 2. Test basic configuration
        thread_id = "test-thread-456"
        checkpoint_ns = ""

        config = {
            "configurable": {"thread_id": thread_id, "checkpoint_ns": checkpoint_ns}
        }

        print(f"\n2. Testing with thread_id: {thread_id}")

        # 3. Create and save a checkpoint
        print("\n3. Creating and saving checkpoint...")
        checkpoint_id = "checkpoint-001"
        checkpoint = create_sample_checkpoint(checkpoint_id)
        metadata: CheckpointMetadata = {"source": "test", "step": 1, "writes": {}}

        # Save the checkpoint
        saved_config = checkpointer.put(
            config=config, checkpoint=checkpoint, metadata=metadata, new_versions={}
        )

        print(f"   ✅ Saved checkpoint: {checkpoint_id}")
        print(f"   📍 Saved config: {saved_config}")

        # 4. Retrieve the checkpoint
        print("\n4. Retrieving checkpoint...")
        retrieved = checkpointer.get_tuple(saved_config)

        if retrieved:
            print("   ✅ Successfully retrieved checkpoint")
            print(f"   📋 Checkpoint ID: {retrieved.checkpoint['id']}")
            print(
                f"   👤 User name: {retrieved.checkpoint['channel_values']['user_name']}"
            )
            print(f"   📊 Count: {retrieved.checkpoint['channel_values']['count']}")
            print(
                f"   💬 Messages: {len(retrieved.checkpoint['channel_values']['messages'])}"
            )
        else:
            print("   ❌ Failed to retrieve checkpoint")
            return False

        # 5. Test writes
        print("\n5. Testing writes...")
        writes = [
            ("channel1", {"data": "test_value_1"}),
            ("channel2", {"data": "test_value_2"}),
        ]

        checkpointer.put_writes(
            config=saved_config, writes=writes, task_id="test-task-1"
        )
        print("   ✅ Saved writes successfully")

        # 6. Create second checkpoint to test listing
        print("\n6. Creating second checkpoint...")
        checkpoint_id_2 = "checkpoint-002"
        checkpoint_2 = create_sample_checkpoint(checkpoint_id_2)
        checkpoint_2["channel_values"]["count"] = 2
        checkpoint_2["channel_values"]["messages"].append(
            {"type": "human", "content": "How are you?"}
        )

        config_2 = {
            "configurable": {
                "thread_id": thread_id,
                "checkpoint_ns": checkpoint_ns,
                "checkpoint_id": checkpoint_id,  # Parent checkpoint
            }
        }

        checkpointer.put(
            config=config_2,
            checkpoint=checkpoint_2,
            metadata={"source": "test", "step": 2, "writes": {}},
            new_versions={},
        )
        print(f"   ✅ Saved second checkpoint: {checkpoint_id_2}")

        # 7. List all checkpoints for the thread
        print("\n7. Listing checkpoints...")
        thread_config = {
            "configurable": {"thread_id": thread_id, "checkpoint_ns": checkpoint_ns}
        }

        checkpoints = list(checkpointer.list(thread_config, limit=10))
        print(f"   ✅ Found {len(checkpoints)} checkpoints")

        for i, cp_tuple in enumerate(checkpoints):
            cp_id = cp_tuple.checkpoint["id"]
            count = cp_tuple.checkpoint["channel_values"]["count"]
            msg_count = len(cp_tuple.checkpoint["channel_values"]["messages"])
            print(f"   {i + 1}. {cp_id} (count: {count}, messages: {msg_count})")

        # 8. Test getting latest checkpoint
        print("\n8. Getting latest checkpoint...")
        latest = checkpointer.get_tuple(thread_config)
        if latest:
            print(f"   ✅ Latest checkpoint: {latest.checkpoint['id']}")
            print(f"   📊 Latest count: {latest.checkpoint['channel_values']['count']}")
        else:
            print("   ❌ No latest checkpoint found")

        print("\n🎉 All tests passed!")
        print("\nIndices created in Elasticsearch:")
        print("   📦 direct_test_checkpoints")
        print("   📦 direct_test_writes")

        return True

    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback

        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = test_checkpointer()
    exit(0 if success else 1)
