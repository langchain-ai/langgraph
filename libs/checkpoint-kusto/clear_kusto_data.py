"""Clear old data from Kusto to start fresh with JSON serializer."""
import asyncio
import os
from azure.identity.aio import DefaultAzureCredential as AsyncDefaultAzureCredential
from azure.kusto.data import KustoConnectionStringBuilder
from azure.kusto.data.aio import KustoClient

# Configuration
CLUSTER_URI = os.getenv("KUSTO_CLUSTER_URI", "https://trd-tmsyxf11yg21na1kuv.z7.kusto.fabric.microsoft.com")
DATABASE = os.getenv("KUSTO_DATABASE", "langgraph")

async def clear_kusto_data():
    """Clear all data from Kusto checkpointer tables."""
    print(f"🗑️  Clearing Kusto data...")
    print(f"   Cluster: {CLUSTER_URI}")
    print(f"   Database: {DATABASE}")
    
    # Create connection
    kcsb = KustoConnectionStringBuilder.with_azure_token_credential(
        CLUSTER_URI,
        AsyncDefaultAzureCredential()
    )
    
    async with KustoClient(kcsb) as client:
        # Clear checkpoints
        query1 = f".clear table Checkpoints data"
        print(f"\n📝 Executing: {query1}")
        response1 = await client.execute(DATABASE, query1)
        print(f"✓ Cleared Checkpoints table")
        
        # Clear writes
        query2 = f".clear table CheckpointWrites data"
        print(f"\n📝 Executing: {query2}")
        response2 = await client.execute(DATABASE, query2)
        print(f"✓ Cleared CheckpointWrites table")
        
    print(f"\n✅ All data cleared! You can now run the tutorial with fresh data.")

if __name__ == "__main__":
    asyncio.run(clear_kusto_data())
