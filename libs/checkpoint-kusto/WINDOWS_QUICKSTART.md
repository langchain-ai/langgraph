# Windows Quick Start Guide

Quick reference for Windows users to get started with LangGraph Kusto Checkpointer.

## Prerequisites

- Python 3.10+ installed
- PowerShell or Command Prompt
- Azure account

## Step 1: Install Dependencies

Open PowerShell in the `libs/checkpoint-kusto` directory:

```powershell
# Navigate to the directory
cd f:\langgraph\langgraph-kusto\libs\checkpoint-kusto

# Install the package
pip install -e .
```

## Step 2: Login to Azure

```powershell
az login
```

## Step 3: Set Environment Variables

**PowerShell:**
```powershell
$env:KUSTO_CLUSTER_URI = "https://your-cluster.eastus.kusto.windows.net"
$env:KUSTO_DATABASE = "langgraph"
```

**Command Prompt:**
```cmd
set KUSTO_CLUSTER_URI=https://your-cluster.eastus.kusto.windows.net
set KUSTO_DATABASE=langgraph
```

To make them permanent:
```powershell
# PowerShell (as Administrator)
[System.Environment]::SetEnvironmentVariable('KUSTO_CLUSTER_URI', 'https://your-cluster.eastus.kusto.windows.net', 'User')
[System.Environment]::SetEnvironmentVariable('KUSTO_DATABASE', 'langgraph', 'User')
```

## Step 4: Provision Kusto Tables

See [SETUP.md](SETUP.md) for detailed instructions on creating tables with `provision.kql`.

## Step 5: Run Tutorials

```powershell
cd examples
python tutorial_01_first_checkpoint.py
python tutorial_02_simple_agent.py
python tutorial_03_production_ready.py
```

## Common Windows Issues

### Issue: "Can't close async token provider with sync close"

**Problem:** Mixing sync and async components in the Azure SDK.

**Solution:** This has been fixed in the code. The `ManagedStreamingIngestClient` is synchronous and should not be awaited when closing.

If you still see this error:
```powershell
# Make sure you're using the system Python, not a broken venv
deactivate  # If in a venv
python examples\tutorial_01_first_checkpoint.py
```

### Issue: "source is not recognized"

**Problem:** You're using a Linux/Mac command in PowerShell.

**Solution:**
```powershell
# DON'T use: source .venv/bin/activate
# DO use:
.venv\Scripts\Activate.ps1

# If you get execution policy error:
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

### Issue: "Activate.ps1 cannot be loaded"

**Problem:** PowerShell execution policy blocks scripts.

**Solution:**
```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
# Then try again:
.venv\Scripts\Activate.ps1
```

### Issue: Path separators are wrong

**Problem:** Using forward slashes `/` instead of backslashes `\`.

**Solution:**
```powershell
# DON'T use: .venv/Scripts/activate
# DO use:
.venv\Scripts\Activate.ps1
```

### Issue: "pip is not recognized"

**Problem:** Python Scripts folder is not in PATH.

**Solution:**
```powershell
# Add Python to PATH (replace with your Python path)
$env:Path += ";C:\Users\YourUsername\AppData\Local\Programs\Python\Python310\Scripts"

# Or reinstall Python with "Add to PATH" option checked
```

## Virtual Environment on Windows

### PowerShell:
```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -e .
```

### Command Prompt:
```cmd
python -m venv venv
venv\Scripts\activate.bat
pip install -e .
```

## Complete Setup Script (PowerShell)

Copy and paste this entire script:

```powershell
# Navigate to checkpoint-kusto directory
cd f:\langgraph\langgraph-kusto\libs\checkpoint-kusto

# Install dependencies
pip install -e .

# Login to Azure
az login

# Set environment variables (replace with your values)
$env:KUSTO_CLUSTER_URI = "https://your-cluster.eastus.kusto.windows.net"
$env:KUSTO_DATABASE = "langgraph"

# Run first tutorial
cd examples
python tutorial_01_first_checkpoint.py
```

## Verifying Environment Variables

**PowerShell:**
```powershell
echo $env:KUSTO_CLUSTER_URI
echo $env:KUSTO_DATABASE
```

**Command Prompt:**
```cmd
echo %KUSTO_CLUSTER_URI%
echo %KUSTO_DATABASE%
```

## File Paths on Windows

Always use backslashes `\` or double backslashes `\\` in Windows paths:

```powershell
# Good:
cd C:\Users\YourName\Projects\langgraph-kusto
cd C:\\Users\\YourName\\Projects\\langgraph-kusto

# Also works in PowerShell:
cd C:/Users/YourName/Projects/langgraph-kusto
```

## Need More Help?

- Full setup guide: [SETUP.md](SETUP.md)
- Tutorial: [examples/TUTORIAL.md](examples/TUTORIAL.md)
- Examples: [examples/README.md](examples/README.md)

## Quick Commands Reference

| Task | PowerShell Command |
|------|-------------------|
| Install package | `pip install -e .` |
| Login to Azure | `az login` |
| Set env var | `$env:VAR_NAME = "value"` |
| Check env var | `echo $env:VAR_NAME` |
| Activate venv | `.\venv\Scripts\Activate.ps1` |
| Run Python | `python script.py` |
| Check Python | `python --version` |
| List packages | `pip list` |

---

**Remember:** When copying commands from Unix/Linux documentation, you often need to:
- Replace `/` with `\` in paths
- Replace `source` with running the script directly (`.venv\Scripts\Activate.ps1`)
- Use `$env:` instead of `export` for environment variables
- Use `;` instead of `:` for PATH separators
