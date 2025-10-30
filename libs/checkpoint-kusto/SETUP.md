# Setup Guide

Complete setup instructions for running the LangGraph Kusto Checkpointer tutorials.

## Prerequisites

- **Python 3.10 or higher**
- **Kusto cluster endpoint** (see options below)
- **Azure credentials** (Azure CLI, Service Principal, or Managed Identity)

## Getting a Kusto Endpoint

You need access to a Kusto cluster. Choose one of these options:

### Option 1: Azure Data Explorer (ADX) Cluster
Deploy a dedicated cluster in Azure. **Best for production workloads.**

- [Create an Azure Data Explorer cluster](https://learn.microsoft.com/azure/data-explorer/create-cluster-database-portal)
- Free tier available for development/testing
- Endpoint format: `https://<cluster-name>.<region>.kusto.windows.net`

### Option 2: Microsoft Fabric Eventhouse
Use Kusto as part of Microsoft Fabric's Real-Time Intelligence. **Best for Fabric users.**

- [Create an Eventhouse in Microsoft Fabric](https://learn.microsoft.com/fabric/real-time-intelligence/create-eventhouse)
- Integrated with Fabric workspace and OneLake
- Endpoint format: `https://<workspace>.<region>.kusto.fabric.microsoft.com`

### Option 3: Free Cluster
Get started immediately with no Azure subscription. **Best for learning and testing.**

- [Access free cluster at dataexplorer.azure.com](https://dataexplorer.azure.com/freecluster)
- No credit card required
- Perfect for tutorials and experimentation
- Endpoint: `https://help.kusto.windows.net`
- Database: Create your own (e.g., `langgraph`)

## Step 1: Install Dependencies

You have several options:

### Option A: Install from Source (Recommended for Development)

Since you're working with the source code, install in editable mode:

```bash
# Navigate to the checkpoint-kusto directory
cd libs/checkpoint-kusto

# Install the package with all dependencies
pip install -e .

# This installs:
# - langgraph (core framework)
# - azure-kusto-data (query client)
# - azure-kusto-ingest (ingestion client)
# - azure-identity (authentication)
# - orjson (fast JSON)
```

### Option B: Quick Install All Dependencies

If Option A doesn't work, install dependencies manually:

```bash
# Install LangGraph first
pip install langgraph>=0.2.0

# Install Azure Kusto packages (with async support)
pip install "azure-kusto-data[aio]>=4.3.1"
pip install azure-kusto-ingest>=4.3.1
pip install azure-identity>=1.15.0

# Install async HTTP client (required by azure-kusto-data)
pip install aiohttp>=3.8.0

# Install utilities
pip install orjson>=3.10.1

# Then install checkpoint-kusto
cd libs/checkpoint-kusto
pip install -e .
```

### Option C: Using uv (Fastest)

If you have `uv` installed:

```bash
# Linux/Mac
cd libs/checkpoint-kusto
uv sync
source .venv/bin/activate

# Windows (PowerShell)
cd libs/checkpoint-kusto
uv sync
.venv\Scripts\Activate.ps1

# Windows (Command Prompt)
cd libs/checkpoint-kusto
uv sync
.venv\Scripts\activate.bat
```

## Step 2: Verify Installation

Test that everything is installed:

```bash
python -c "from langgraph.checkpoint.kusto.aio import AsyncKustoSaver; print('✓ Installation successful!')"
```

Expected output: `✓ Installation successful!`

## Step 3: Provision Kusto Tables

Once you have your Kusto cluster endpoint (from one of the options above), you need to create the required tables.

### Run provision.kql

1. Open [Kusto Web Explorer](https://dataexplorer.azure.com/)
2. Connect to your cluster
3. Select your database
4. Copy and run the contents of `libs/checkpoint-kusto/provision.kql`

This creates:
- `Checkpoints` table
- `CheckpointWrites` table
- `LatestCheckpoints` materialized view

### Grant Permissions

You need two permissions:
- **Database Viewer** (read checkpoints)
- **Database Ingestor** (write checkpoints)

```kql
// Run in Kusto Web Explorer
.add database langgraph viewers ('aaduser=your-email@domain.com')
.add database langgraph ingestors ('aaduser=your-email@domain.com')
```

## Step 4: Configure Authentication

### Option A: Azure CLI (Easiest)

```bash
az login
```

This authenticates using your Azure account.

### Option B: Service Principal

```bash
# Windows (PowerShell)
$env:AZURE_CLIENT_ID = "your-client-id"
$env:AZURE_CLIENT_SECRET = "your-secret"
$env:AZURE_TENANT_ID = "your-tenant-id"

# Linux/Mac
export AZURE_CLIENT_ID="your-client-id"
export AZURE_CLIENT_SECRET="your-secret"
export AZURE_TENANT_ID="your-tenant-id"
```

### Option C: Managed Identity

If running on Azure (VM, App Service, etc.), Managed Identity works automatically.

## Step 5: Set Environment Variables

### Windows (PowerShell)

```powershell
$env:KUSTO_CLUSTER_URI = "https://your-cluster.eastus.kusto.windows.net"
$env:KUSTO_DATABASE = "langgraph"
```

### Windows (Command Prompt)

```cmd
set KUSTO_CLUSTER_URI=https://your-cluster.eastus.kusto.windows.net
set KUSTO_DATABASE=langgraph
```

### Linux/Mac

```bash
export KUSTO_CLUSTER_URI="https://your-cluster.eastus.kusto.windows.net"
export KUSTO_DATABASE="langgraph"
```

### Or Create .env File

Create `libs/checkpoint-kusto/examples/.env`:

```bash
KUSTO_CLUSTER_URI=https://your-cluster.eastus.kusto.windows.net
KUSTO_DATABASE=langgraph
```

## Step 6: Run Your First Tutorial

```bash
cd libs/checkpoint-kusto/examples
python tutorial_01_first_checkpoint.py
```

Expected output:

```
📊 Connecting to Kusto...
   Cluster: https://your-cluster.eastus.kusto.windows.net
   Database: langgraph

🔍 Verifying database schema...
✓ Connected successfully!

💾 Saving checkpoint...
✓ Checkpoint saved!

⏳ Waiting for data to be available (streaming ingestion)...

📖 Loading checkpoint...
✓ Loaded checkpoint: checkpoint-001
  Timestamp: 2025-10-27T10:00:00
  Messages: ['Hello, world!']
  Metadata: {'user': 'tutorial', 'step': 1}

🎉 Success! Your first checkpoint works!
```

## Troubleshooting

### Error: "Aio modules not installed"

**Full error**: `KustoAioSyntaxError: Aio modules not installed, run 'pip install azure-kusto-data[aio]'`

**Solution**: Install the async support for azure-kusto-data:

```bash
pip install "azure-kusto-data[aio]>=4.3.1"

# Or reinstall the package
cd libs/checkpoint-kusto
pip install -e .
```

The `[aio]` extras include the `asgiref` package needed for async operations.

### Error: "Can't close async token provider with sync close"

**Solution**: This is a compatibility issue between sync and async components. 

This has been fixed in v3.0.0. If you still see this:

1. Make sure you're using the latest code
2. Verify you're not mixing sync/async credentials incorrectly
3. Check that you're using `AsyncKustoSaver` for async code

### Error: "No module named 'aiohttp'"

**Solution**: The Azure Kusto SDK requires aiohttp for async operations. This should be installed automatically, but if it's not:

```bash
pip install aiohttp>=3.8.0

# Or reinstall the package
cd libs/checkpoint-kusto
pip install -e .
```

### Error: "No module named 'langgraph'"

**Solution**: Install dependencies

```bash
cd libs/checkpoint-kusto
pip install -e .
```

If that doesn't work:

```bash
pip install langgraph
```

### Error: "No module named 'langgraph.checkpoint.kusto'"

**Solution**: Make sure you're installing from the right directory

```bash
cd libs/checkpoint-kusto
pip install -e .

# Verify installation
python -c "import langgraph.checkpoint.kusto; print('OK')"
```

### Error: "Authentication failed"

**Solution**: Log in with Azure CLI

```bash
az login
az account show  # Verify you're logged in
```

### Error: "Table 'Checkpoints' does not exist"

**Solution**: Run the provision script

1. Go to https://dataexplorer.azure.com/
2. Connect to your cluster
3. Run the `provision.kql` script contents

### Error: "Permission denied"

**Solution**: Add permissions

```kql
.add database langgraph viewers ('aaduser=your-email@domain.com')
.add database langgraph ingestors ('aaduser=your-email@domain.com')
```

### Error: "Checkpoint not found"

This is normal! Streaming ingestion takes 1-5 seconds. 

**Solution**: Increase wait time in the tutorial:

```python
await asyncio.sleep(5)  # Wait longer
```

### Error: "azure.kusto.data not found"

**Solution**: Install Azure packages

```bash
pip install azure-kusto-data azure-kusto-ingest azure-identity
```

## Verification Checklist

Before running tutorials, verify:

- [ ] Python 3.10+ installed: `python --version`
- [ ] Dependencies installed: `pip list | grep langgraph`
- [ ] Azure CLI logged in: `az account show`
- [ ] Environment variables set: `echo $KUSTO_CLUSTER_URI`
- [ ] Tables created in Kusto: `.show tables` (should show 3 tables)
- [ ] Permissions granted: Check with Kusto admin

## Quick Reference

### Complete Setup (Copy-Paste)

```bash
# 1. Install dependencies
cd libs/checkpoint-kusto
pip install -e .

# 2. Login to Azure
az login

# 3. Set environment variables (Linux/Mac)
export KUSTO_CLUSTER_URI="https://your-cluster.eastus.kusto.windows.net"
export KUSTO_DATABASE="langgraph"

# 4. Run provision.kql in Kusto Web Explorer
# (Copy contents from provision.kql)

# 5. Run tutorial
cd examples
python tutorial_01_first_checkpoint.py
```

### Minimal pyproject.toml Check

If dependencies aren't installing, check `libs/checkpoint-kusto/pyproject.toml` includes:

```toml
[project]
dependencies = [
    "langgraph-checkpoint>=2.1.2,<3.0.0",
    "azure-kusto-data>=4.3.1,<5.0.0",
    "azure-kusto-ingest>=4.3.1,<5.0.0",
    "azure-identity>=1.15.0,<2.0.0",
    "orjson>=3.10.1,<4.0.0",
]
```

## Next Steps

Once setup is complete:

1. ✅ Run `tutorial_01_first_checkpoint.py` - Learn basics
2. ✅ Run `tutorial_02_simple_agent.py` - Build a chatbot
3. ✅ Run `tutorial_03_production_ready.py` - Best practices
4. ✅ Read `examples/TUTORIAL.md` - Full guide
5. ✅ Explore `examples/` - More examples

## Getting Help

- 📖 Read [examples/TUTORIAL.md](examples/TUTORIAL.md) for concepts
- 📚 Check [README.md](README.md) for API reference
- 🐛 Report issues on GitHub
- 💬 Ask in LangChain Discord

---

**Pro tip**: Use a virtual environment to avoid conflicts:

```bash
# Linux/Mac
python -m venv venv
source venv/bin/activate
cd libs/checkpoint-kusto
pip install -e .
```

```powershell
# Windows (PowerShell)
python -m venv venv
.\venv\Scripts\Activate.ps1
cd libs/checkpoint-kusto
pip install -e .
```

```cmd
# Windows (Command Prompt)
python -m venv venv
venv\Scripts\activate.bat
cd libs/checkpoint-kusto
pip install -e .
```
