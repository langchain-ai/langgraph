"""Test that checkpoint fields are properly populated."""
import asyncio
import os
from datetime import datetime
from langchain_core.messages import HumanMessage
from langgraph.checkpoint.kusto import AsyncKustoSaver
from langgraph.checkpoint.kusto.json_serializer import JsonStringSerializer

async def test_checkpoint_fields():
    """Test that checkpoint records have all fields populated."""
    print("🧪 Testing Checkpoint Field Population...")
    
    # Connect to Kusto
    async with AsyncKustoSaver.from_connection_string(
        cluster_uri=os.environ["KUSTO_CLUSTER_URI"],
        database=os.environ["KUSTO_DATABASE"],
    ) as checkpointer:
        await checkpointer.setup()
        print("✓ Connected to Kusto")
        
        # Create a test checkpoint
        from langgraph.checkpoint.base import Checkpoint, empty_checkpoint
        
        config = {"configurable": {"thread_id": "test-fields-thread", "checkpoint_id": "test-123"}}
        
        checkpoint = empty_checkpoint()
        checkpoint["id"] = "test-123"
        checkpoint["channel_values"] = {"messages": [HumanMessage("Test message")]}
        checkpoint["channel_versions"] = {"messages": "1"}
        
        # Put the checkpoint
        await checkpointer.aput(
            config,
            checkpoint,
            {"source": "test"},
            None
        )
        
        # Flush to ensure it's written
        await checkpointer.flush()
        print("✓ Checkpoint written")
        
        # Wait a moment for streaming ingestion
        await asyncio.sleep(2)
        
        # Query to check fields
        query = """
        Checkpoints
        | where thread_id == "test-fields-thread"
        | project thread_id, checkpoint_ns, checkpoint_id, type, created_at
        | take 1
        """
        
        async with checkpointer._query() as client:
            response = await client.execute(checkpointer.database, query)
            
            if response.primary_results and len(response.primary_results[0]) > 0:
                row = response.primary_results[0][0]
                print(f"\n✓ Checkpoint fields:")
                print(f"  thread_id: {row['thread_id']}")
                print(f"  checkpoint_ns: '{row['checkpoint_ns']}'")
                print(f"  checkpoint_id: {row['checkpoint_id']}")
                print(f"  type: '{row['type']}'")
                print(f"  created_at: {row['created_at']}")
                
                # Verify fields are not empty
                assert row['checkpoint_ns'] == "", "checkpoint_ns should be empty string (default)"
                assert row['type'] == "json", f"type should be 'json', got '{row['type']}'"
                assert row['created_at'] is not None and row['created_at'] != "", "created_at should not be empty"
                
                print(f"\n✅ All fields properly populated!")
            else:
                print(f"❌ No checkpoint found!")

if __name__ == "__main__":
    asyncio.run(test_checkpoint_fields())
