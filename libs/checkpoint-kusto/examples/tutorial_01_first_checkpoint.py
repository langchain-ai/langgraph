"""
tutorial_01_first_checkpoint.py

This example shows the basics of saving and loading checkpoints.

Prerequisites:
1. Install dependencies: cd libs/checkpoint-kusto && pip install -e .
2. Set KUSTO_CLUSTER_URI environment variable
3. Set KUSTO_DATABASE environment variable
4. Run provision.kql to create tables

For detailed setup instructions, see: ../SETUP.md
"""

import asyncio
import os
import sys


def check_dependencies():
    """Check if required dependencies are installed."""
    try:
        from langgraph.checkpoint.kusto.aio import AsyncKustoSaver
        from langgraph.checkpoint.base import Checkpoint
        return AsyncKustoSaver, Checkpoint
    except ImportError as e:
        print("❌ Missing dependencies!")
        print(f"   Error: {e}")
        print("\n📦 To install all dependencies:")
        print("   cd libs\\checkpoint-kusto")
        print("   pip install -e .")
        print("\n   If that doesn't work, install manually:")
        print("   pip install langgraph")
        print('   pip install "azure-kusto-data[aio]>=4.3.1"')
        print("   pip install azure-kusto-ingest azure-identity aiohttp")
        print("\n   Or run the setup check:")
        print("   python setup_check.py")
        print("\n📖 For detailed setup instructions, see: ../SETUP.md")
        sys.exit(1)


async def main():
    # Check dependencies first
    AsyncKustoSaver, Checkpoint = check_dependencies()
    
    # Configuration from environment variables
    cluster_uri = os.getenv(
        "KUSTO_CLUSTER_URI",
        "https://trd-tmsyxf11yg21na1kuv.z7.kusto.fabric.microsoft.com"
    )
    database = os.getenv("KUSTO_DATABASE", "langgraph")
    
    # Validate configuration
    if "your-cluster" in cluster_uri:
        print("⚠️  Warning: Using default cluster URI")
        print("   Set KUSTO_CLUSTER_URI environment variable to your cluster")
        print("\n   Example (PowerShell):")
        print('   $env:KUSTO_CLUSTER_URI = "https://mycluster.eastus.kusto.windows.net"')
        print("\n   Example (Bash):")
        print('   export KUSTO_CLUSTER_URI="https://mycluster.eastus.kusto.windows.net"')
        print()
    
    print("📊 Connecting to Kusto...")
    print(f"   Cluster: {cluster_uri}")
    print(f"   Database: {database}")
    
    # Create checkpointer using context manager for automatic cleanup
    async with AsyncKustoSaver.from_connection_string(
        cluster_uri=cluster_uri,
        database=database,
    ) as checkpointer:
        # Verify tables exist
        print("\n🔍 Verifying database schema...")
        await checkpointer.setup()
        print("✓ Connected successfully!")
        
        # Create a simple checkpoint
        checkpoint = Checkpoint(
            v=1,
            id="checkpoint-001",
            ts="2025-10-27T10:00:00",
            channel_values={"messages": ["Hello, world!"]},
            channel_versions={"messages": 1},
            versions_seen={},
            pending_sends=[],
        )
        
        # Configuration identifies this thread
        config = {"configurable": {"thread_id": "tutorial-thread-1"}}
        
        # Save the checkpoint
        print("\n💾 Saving checkpoint...")
        await checkpointer.aput(
            config=config,
            checkpoint=checkpoint,
            metadata={"user": "tutorial", "step": 1},
            new_versions={"messages": 1},
        )
        
        # Flush ensures data is written to Kusto
        await checkpointer.flush()
        print("✓ Checkpoint saved!")
        
        # Wait for streaming ingestion to complete
        print("\n⏳ Waiting for data to be available (streaming ingestion)...")
        await asyncio.sleep(2)
        
        # Load it back
        print("\n📖 Loading checkpoint...")
        loaded = await checkpointer.aget_tuple(config)
        
        if loaded:
            print(f"✓ Loaded checkpoint: {loaded.checkpoint['id']}")
            print(f"  Timestamp: {loaded.checkpoint['ts']}")
            print(f"  Messages: {loaded.checkpoint['channel_values']['messages']}")
            print(f"  Metadata: {loaded.metadata}")
            print("\n🎉 Success! Your first checkpoint works!")
        else:
            print("⚠ Checkpoint not found yet")
            print("  Try increasing the wait time or check Kusto permissions")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n\n👋 Cancelled by user")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        print("\n🔧 Troubleshooting tips:")
        print("1. Check KUSTO_CLUSTER_URI is set correctly")
        print("2. Check KUSTO_DATABASE exists")
        print("3. Run provision.kql to create tables")
        print("4. Verify you have 'Database Viewer' and 'Database Ingestor' permissions")
        print("5. Check Azure authentication (az login)")
        print("\n📖 For detailed help, see: ../SETUP.md")
        raise
