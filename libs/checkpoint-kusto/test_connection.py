"""Quick test to verify AsyncKustoSaver connection."""

import asyncio
import os


async def test_connection():
    """Test basic connection to Kusto."""
    from langgraph.checkpoint.kusto.aio import AsyncKustoSaver
    
    # Configuration
    cluster_uri = os.getenv(
        "KUSTO_CLUSTER_URI",
        "https://your-cluster.eastus.kusto.windows.net"
    )
    database = os.getenv("KUSTO_DATABASE", "langgraph")
    
    print(f"Testing connection to:")
    print(f"  Cluster: {cluster_uri}")
    print(f"  Database: {database}")
    print()
    
    try:
        async with AsyncKustoSaver.from_connection_string(
            cluster_uri=cluster_uri,
            database=database,
        ) as checkpointer:
            print("✓ Connection created")
            print("✓ Context manager working")
            print("✓ Cleanup successful")
            print("\n🎉 Connection test passed!")
            
    except Exception as e:
        print(f"❌ Connection failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True


if __name__ == "__main__":
    success = asyncio.run(test_connection())
    exit(0 if success else 1)
