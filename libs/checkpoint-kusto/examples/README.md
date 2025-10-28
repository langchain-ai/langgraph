# LangGraph Kusto Checkpointer - Tutorial Examples

This directory contains hands-on tutorials for the LangGraph Kusto checkpointer.

## 🚀 Quick Start

**Before running the examples:**

1. **Install dependencies:**
   ```bash
   cd libs/checkpoint-kusto
   pip install -e .
   ```

2. **Run setup check (recommended):**
   ```bash
   python setup_check.py
   ```
   This will verify your Python version, dependencies, Azure CLI, and environment variables.

3. **Set up Azure Data Explorer:**
   - Create a Kusto cluster and database
   - Run `provision.kql` to create required tables
   - See [`../SETUP.md`](../SETUP.md) for detailed instructions

4. **Configure environment variables:**
   ```powershell
   # PowerShell (Windows)
   $env:KUSTO_CLUSTER_URI = "https://your-cluster.region.kusto.windows.net"
   $env:KUSTO_DATABASE = "langgraph"
   ```
   ```bash
   # Bash (Linux/Mac)
   export KUSTO_CLUSTER_URI="https://your-cluster.region.kusto.windows.net"
   export KUSTO_DATABASE="langgraph"
   ```

5. **Login to Azure:**
   ```bash
   az login
   ```

## 📚 Tutorial Examples

### Tutorial 01: First Checkpoint
**File:** `tutorial_01_first_checkpoint.py`  
**Run time:** ~10 seconds  
**Difficulty:** Beginner

Learn the basics of checkpointing:
- How to create and save a checkpoint
- How to retrieve a saved checkpoint
- What data is stored in a checkpoint

```bash
cd examples
python tutorial_01_first_checkpoint.py
```

**What you'll see:**
- Checkpoint saved to Kusto
- Checkpoint retrieved and displayed
- Checkpoint ID and metadata

### Tutorial 02: Simple Agent
**File:** `tutorial_02_simple_agent.py`  
**Run time:** ~15 seconds  
**Difficulty:** Beginner

Build a simple chatbot with conversation history:
- Create a stateful agent using StateGraph
- Use checkpointer to remember conversations
- See how conversation history persists

```bash
cd examples
python tutorial_02_simple_agent.py
```

**What you'll see:**
- Multi-turn conversation
- Conversation history retrieved from checkpoints
- Each message builds on previous ones

### Tutorial 03: Production Ready
**File:** `tutorial_03_production_ready.py`  
**Run time:** ~20 seconds  
**Difficulty:** Intermediate

Learn production best practices:
- Structured logging
- Error handling with Azure exceptions
- Resource cleanup
- Verification steps

```bash
cd examples
python tutorial_03_production_ready.py
```

**What you'll see:**
- Detailed logging output
- Graceful error handling
- Resource cleanup procedures
- Production-ready patterns

### Tutorial 04: OpenAI Chatbot ⭐ NEW
**File:** `tutorial_04_openai_chatbot.py`  
**Run time:** ~30 seconds  
**Difficulty:** Intermediate

Build a real chatbot using **OpenAI or Azure OpenAI**:

- Real LLM calls with ChatOpenAI (gpt-4o-mini)
- **Supports both OpenAI and Azure OpenAI**
- Proper LangChain message handling (HumanMessage, AIMessage, SystemMessage)
- Conversation memory with context awareness
- System message configuration
- Full conversation persistence in Kusto

**Prerequisites:**

```bash
pip install langchain-openai
```

**Option A - OpenAI:**

```powershell
# PowerShell
$env:OPENAI_API_KEY = "sk-..."
```

```bash
# Bash
export OPENAI_API_KEY="sk-..."
```

**Option B - Azure OpenAI (Recommended for Enterprise):**

```powershell
# PowerShell
$env:AZURE_OPENAI_ENDPOINT = "https://your-resource.openai.azure.com/"
$env:AZURE_OPENAI_API_KEY = "your-key"
$env:AZURE_OPENAI_DEPLOYMENT = "gpt-4o-mini"
```

📚 **See [AZURE_OPENAI_SETUP.md](AZURE_OPENAI_SETUP.md) for detailed Azure OpenAI configuration**
export OPENAI_API_KEY="sk-..."
```

**Run:**
```bash
cd examples
python tutorial_04_openai_chatbot.py
```

**What you'll see:**
- Real AI conversations with GPT
- Context-aware responses (AI remembers conversation)
- System message for personality configuration
- LangChain message objects stored as JSON in Kusto
- Full conversation history retrieval

**Try this:** Run the script multiple times - the AI will remember your entire conversation history!

## 🐛 Troubleshooting

### "ModuleNotFoundError: No module named 'langgraph'"

**Solution:** Install dependencies:
```bash
cd libs/checkpoint-kusto
pip install -e .
```

Or use the setup check script:
```bash
python setup_check.py
```

### "KUSTO_CLUSTER_URI environment variable is not set"

**Solution:** Set your environment variables:
```powershell
# PowerShell
$env:KUSTO_CLUSTER_URI = "https://your-cluster.region.kusto.windows.net"
$env:KUSTO_DATABASE = "langgraph"
```

### "Authentication failed" or "401 Unauthorized"

**Solution:** Login to Azure:
```bash
az login
```

Make sure your account has access to the Kusto cluster.

### "Table does not exist"

**Solution:** Run the provision script:
1. Open Azure Data Explorer Web UI
2. Connect to your database
3. Run the contents of `provision.kql`

### Getting help

- **Detailed setup guide:** [`../SETUP.md`](../SETUP.md)
- **Full tutorial:** [`../TUTORIAL.md`](../TUTORIAL.md)
- **Change documentation:** [`../INGESTION_SIMPLIFICATION.md`](../INGESTION_SIMPLIFICATION.md)

## 🎯 Next Steps

After completing the tutorials:

1. **Read the full tutorial:** [`../TUTORIAL.md`](../TUTORIAL.md) for in-depth explanations
2. **Explore examples:** Check the `examples/` directory for more complex scenarios
3. **Build your own:** Use these patterns to build your own LangGraph applications
4. **Production deployment:** Review [`../SETUP.md`](../SETUP.md) for production best practices

## 📝 Notes

- **Dependencies:** All examples include built-in dependency checking
- **Error handling:** Examples provide helpful error messages and troubleshooting tips
- **Platform support:** Examples work on Windows, Linux, and macOS
- **Async patterns:** All examples use async/await for best performance

## 🔗 Additional Resources

- [LangGraph Documentation](https://langchain-ai.github.io/langgraph/)
- [Azure Data Explorer Documentation](https://docs.microsoft.com/azure/data-explorer/)
- [Azure Authentication Guide](https://docs.microsoft.com/azure/developer/python/azure-sdk-authenticate)

Happy coding! 🎉
