# Tutorial 04: OpenAI Chatbot with Memory

## Overview

This tutorial demonstrates building a real chatbot using OpenAI's GPT models (or Azure OpenAI) with persistent conversation memory through Kusto checkpoints.

## Key Features

- 🤖 **Real LLM Integration**: Uses ChatOpenAI with gpt-4o-mini
- 💬 **Natural Conversations**: Proper message handling with LangChain
- 🧠 **Persistent Memory**: Full conversation history stored in Kusto
- ☁️ **Flexible Deployment**: Supports both OpenAI and Azure OpenAI
- 🔄 **Context Awareness**: AI remembers entire conversation across sessions

## Prerequisites

### 1. Install Dependencies

```bash
pip install langchain-openai
```

### 2. Choose Your LLM Provider

#### Option A: OpenAI (Quick Start)

1. Get an API key from [OpenAI Platform](https://platform.openai.com/api-keys)
2. Set environment variable:

**Windows (PowerShell):**
```powershell
$env:OPENAI_API_KEY = "sk-..."
```

**Linux/Mac (Bash):**
```bash
export OPENAI_API_KEY="sk-..."
```

#### Option B: Azure OpenAI (Enterprise)

1. Create an Azure OpenAI resource
2. Deploy a model (e.g., gpt-4o-mini)
3. Set environment variables:

**Windows (PowerShell):**
```powershell
$env:AZURE_OPENAI_ENDPOINT = "https://your-resource.openai.azure.com/"
$env:AZURE_OPENAI_API_KEY = "your-key"
$env:AZURE_OPENAI_DEPLOYMENT = "gpt-4o-mini"
```

**Linux/Mac (Bash):**
```bash
export AZURE_OPENAI_ENDPOINT="https://your-resource.openai.azure.com/"
export AZURE_OPENAI_API_KEY="your-key"
export AZURE_OPENAI_DEPLOYMENT="gpt-4o-mini"
```

📚 **See [AZURE_OPENAI_SETUP.md](AZURE_OPENAI_SETUP.md) for detailed Azure OpenAI configuration**

### 3. Kusto Setup

Ensure you have:
- `KUSTO_CLUSTER_URI` set
- `KUSTO_DATABASE` set
- Provisioned tables (run `provision.kql`)
- Azure CLI authentication (`az login`)

## Running the Tutorial

```bash
cd examples
python tutorial_04_openai_chatbot.py
```

## What You'll See

### First Run
```
🤖 Starting OpenAI Chatbot Tutorial...
✅ Using OpenAI (standard)
🎬 Creating chatbot graph...
✅ Chatbot initialized

💬 Conversation 1 (New Session):
User: Hi, I'm Alice!
Assistant: Hello Alice! It's nice to meet you. How can I help you today?

💬 Conversation 2 (Same Session):
User: What's my name?
Assistant: Your name is Alice, as you just told me!

💬 Conversation 3 (Same Session):
User: Tell me a fun fact about AI
Assistant: Here's a fun fact: The term "Artificial Intelligence" was coined in 1956...

📊 Checkpoint Summary:
- Thread ID: f7a3b2c1-...
- Total checkpoints: 3
- Messages stored: 7 (including system message)
```

### Subsequent Runs
When you run the script again, it will load the entire conversation history from Kusto and continue where you left off!

## How It Works

### 1. Graph Structure

```python
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages

class State(TypedDict):
    messages: Annotated[list, add_messages]

# Build graph
workflow = StateGraph(State)
workflow.add_node("chatbot", chatbot_node)
workflow.add_edge(START, "chatbot")
workflow.add_edge("chatbot", END)
```

### 2. LLM Configuration

```python
from langchain_openai import ChatOpenAI, AzureChatOpenAI

# Auto-detects OpenAI vs Azure OpenAI based on environment variables
if os.getenv("AZURE_OPENAI_ENDPOINT"):
    llm = AzureChatOpenAI(
        azure_deployment=os.getenv("AZURE_OPENAI_DEPLOYMENT", "gpt-4o-mini"),
        api_version="2024-02-15-preview",
        temperature=0.7,
    )
else:
    llm = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0.7,
    )
```

### 3. Message Types

LangChain uses specific message types:
- **SystemMessage**: Instructions for the AI (personality, rules)
- **HumanMessage**: User input
- **AIMessage**: AI responses

These are automatically serialized as JSON and stored in Kusto's `dynamic` columns.

### 4. Conversation Flow

```python
# Create thread for conversation
thread_id = str(uuid.uuid4())
config = {"configurable": {"thread_id": thread_id}}

# First message - starts new checkpoint
response = await graph.ainvoke(
    {"messages": [HumanMessage(content="Hi, I'm Alice!")]},
    config=config
)

# Second message - loads checkpoint, adds to history
response = await graph.ainvoke(
    {"messages": [HumanMessage(content="What's my name?")]},
    config=config
)
```

### 5. Persistent Memory

Every conversation turn creates a checkpoint in Kusto:
- **checkpoint_id**: Unique identifier for each state
- **thread_id**: Groups messages in same conversation
- **parent_checkpoint_id**: Links to previous checkpoint
- **checkpoint_ns**: Namespace (empty string for main graph)
- **checkpoint**: Full state including all messages (JSON)

## Key Concepts

### Message Reducer

```python
messages: Annotated[list, add_messages]
```

The `add_messages` reducer:
- Appends new messages to the list
- Maintains conversation order
- Preserves all message types and metadata

### State Persistence

Each checkpoint stores:
```json
{
  "checkpoint": {
    "v": 1,
    "ts": "2025-10-30T...",
    "id": "...",
    "channel_values": {
      "messages": [
        {
          "type": "system",
          "content": "You are a helpful assistant...",
          "id": "..."
        },
        {
          "type": "human", 
          "content": "Hi, I'm Alice!",
          "id": "..."
        },
        {
          "type": "ai",
          "content": "Hello Alice!...",
          "id": "..."
        }
      ]
    }
  }
}
```

### Context Window Management

The example includes all previous messages in each request, so the AI has full context. In production, you might want to:
- Limit context to recent N messages
- Summarize older conversations
- Use semantic search to retrieve relevant past messages

## Production Considerations

### Error Handling

```python
try:
    response = await graph.ainvoke(input_data, config=config)
except Exception as e:
    logger.error(f"LLM call failed: {e}")
    # Implement retry logic, fallback responses, etc.
```

### Rate Limiting

OpenAI and Azure OpenAI have rate limits. Consider:
- Implementing exponential backoff
- Using async batching for multiple requests
- Monitoring token usage

### Cost Management

Each API call costs money based on tokens used:
- Input tokens (your messages + history)
- Output tokens (AI responses)

Monitor usage and implement budgets/alerts.

### Security

- Never commit API keys to git
- Use Azure Key Vault for production
- Implement user authentication
- Sanitize user input

## Troubleshooting

### "OPENAI_API_KEY not set"

Set your OpenAI API key:
```powershell
$env:OPENAI_API_KEY = "sk-..."
```

### "Deployment not found" (Azure OpenAI)

Verify your deployment name matches:
```powershell
$env:AZURE_OPENAI_DEPLOYMENT = "gpt-4o-mini"
```

### "Rate limit exceeded"

You're making too many requests. Wait a moment and try again, or upgrade your OpenAI tier.

### "Context length exceeded"

Your conversation is too long. Implement context window management to keep recent messages only.

## Next Steps

1. ✅ **Experiment**: Try different system messages to change AI personality
2. ✅ **Extend**: Add tools/functions the AI can call
3. ✅ **Scale**: Move to Tutorial 05 for multi-agent collaboration
4. ✅ **Deploy**: Review production best practices in `../SETUP.md`

## Related Resources

- [Tutorial 01: First Checkpoint](tutorial_01_first_checkpoint.py) - Basics
- [Tutorial 02: Simple Agent](tutorial_02_simple_agent.py) - State management
- [Tutorial 03: Production Ready](tutorial_03_production_ready.py) - Error handling
- [Tutorial 05: Multi-Agent](tutorial_05_multi_agent.py) - Advanced collaboration
- [Azure OpenAI Setup Guide](AZURE_OPENAI_SETUP.md) - Detailed Azure config

---

**Ready to build intelligent chatbots with persistent memory!** 🚀
