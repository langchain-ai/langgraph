# LangGraph Repository Architecture

## Overview
LangGraph is a low-level orchestration framework for building, managing, and deploying long-running, stateful AI agents. The repository is organized as a **monorepo** containing multiple interconnected libraries.

---

## 🏗️ Repository Structure

```mermaid
graph TD
    subgraph "Repository Root"
        direction TB
        ROOT["/langgraph"]
        ROOT --> LIBS["/libs"]
        ROOT --> DOCS["/docs"]
        ROOT --> EXAMPLES["/examples"]
    end

    subgraph "Core Libraries (/libs)"
        direction TB
        CHECKPOINT["checkpoint<br/>Base interfaces for<br/>checkpointers"]
        CP_POSTGRES["checkpoint-postgres<br/>Postgres checkpointer<br/>implementation"]
        CP_SQLITE["checkpoint-sqlite<br/>SQLite checkpointer<br/>implementation"]
        LANGGRAPH["langgraph<br/>Core framework for<br/>stateful agents"]
        PREBUILT["prebuilt<br/>High-level APIs<br/>create_react_agent"]
        CLI["cli<br/>Command-line interface<br/>dev, build, deploy"]
        SDK_PY["sdk-py<br/>Python SDK for<br/>LangGraph Server API"]
        SDK_JS["sdk-js<br/>JavaScript/TypeScript SDK<br/>for REST API"]
    end

    LIBS --> CHECKPOINT
    LIBS --> CP_POSTGRES
    LIBS --> CP_SQLITE
    LIBS --> LANGGRAPH
    LIBS --> PREBUILT
    LIBS --> CLI
    LIBS --> SDK_PY
    LIBS --> SDK_JS
```

---

## 🔗 Dependency Graph

```mermaid
graph LR
    subgraph "Base Layer"
        CHECKPOINT[checkpoint]
    end

    subgraph "Storage Implementations"
        CP_POSTGRES[checkpoint-postgres]
        CP_SQLITE[checkpoint-sqlite]
    end

    subgraph "Core & High-Level APIs"
        PREBUILT[prebuilt]
        LANGGRAPH[langgraph]
    end

    subgraph "SDKs & Tools"
        SDK_PY[sdk-py]
        CLI[cli]
        SDK_JS[sdk-js]
    end

    %% Dependencies
    CHECKPOINT --> CP_POSTGRES
    CHECKPOINT --> CP_SQLITE
    CHECKPOINT --> PREBUILT
    CHECKPOINT --> LANGGRAPH

    PREBUILT --> LANGGRAPH

    SDK_PY --> LANGGRAPH
    SDK_PY --> CLI

    style CHECKPOINT fill:#e1f5ff
    style LANGGRAPH fill:#fff9c4
    style PREBUILT fill:#f0f4c3
    style SDK_JS fill:#dcedc8
```

---

## 📦 Library Details

### 1. **checkpoint** - Persistence Layer
Base interfaces for LangGraph checkpointers that provide:
- State persistence across runs
- "Memory" between interactions
- Human-in-the-loop capabilities
- Thread-based state management

**Key Components:**
```
checkpoint/
├── langgraph/checkpoint/base/     # Base interfaces
├── langgraph/checkpoint/memory/   # In-memory implementation
├── langgraph/checkpoint/serde/    # Serialization/deserialization
├── langgraph/cache/base/          # Cache interfaces
└── langgraph/store/base/          # Store interfaces
```

---

### 2. **checkpoint-postgres** & **checkpoint-sqlite**
Concrete implementations of checkpoint savers:
- **Postgres**: Production-ready persistent storage
- **SQLite**: Lightweight, file-based storage
- Both support async operations
- Include cache and store implementations

---

### 3. **langgraph** - Core Framework ⚡
The heart of the system - a low-level orchestration framework.

**Architecture:**
```mermaid
graph TB
    subgraph "LangGraph Core"
        direction TB
        PREGEL["pregel/<br/>Execution Engine<br/>(inspired by Google Pregel)"]
        GRAPH["graph/<br/>Graph Construction<br/>(StateGraph, MessageGraph)"]
        CHANNELS["channels/<br/>State Management<br/>(LastValue, BinaryOp, Topic)"]
        MANAGED["managed/<br/>Managed Values<br/>(is_last_step, etc.)"]
        TYPES["types/<br/>Type Definitions"]
    end

    GRAPH --> PREGEL
    CHANNELS --> PREGEL
    MANAGED --> PREGEL

    style PREGEL fill:#ff9800
    style GRAPH fill:#4caf50
    style CHANNELS fill:#2196f3
```

**Key Modules:**
- **pregel/**: Core execution engine with async support
  - `_algo.py`: Graph algorithms
  - `_loop.py`: Execution loop
  - `_executor.py`: Task execution
  - `_checkpoint.py`: Checkpointing logic
  - `main.py`: Main Pregel class

- **graph/**: Graph building blocks
  - `state.py`: StateGraph (main graph type)
  - `message.py`: MessageGraph (for chat applications)
  - `_node.py`: Node definitions
  - `_branch.py`: Conditional branching

- **channels/**: State channels
  - Different update strategies (last value, reduce, etc.)
  - Topic channels for pub/sub patterns

---

### 4. **prebuilt** - High-Level APIs 🎯
Pre-configured components for common patterns:

```python
from langgraph.prebuilt import create_react_agent

# Quick ReAct agent setup
agent = create_react_agent(model, tools, prompt="...")
```

**Components:**
- `create_react_agent`: ReAct-style tool-calling agent
- `ToolNode`: Executes tool calls
- `ValidationNode`: Validates tool calls against schemas

---

### 5. **cli** - Command Line Interface 🛠️
Official CLI for LangGraph development and deployment:

**Commands:**
- `langgraph new`: Create new project from template
- `langgraph dev`: Development server with hot reload
- `langgraph up`: Launch in Docker
- `langgraph build`: Build Docker image
- `langgraph dockerfile`: Generate Dockerfile

**Configuration:** `langgraph.json`
```json
{
  "dependencies": ["langchain_openai", "./your_package"],
  "graphs": {
    "my_graph": "./your_package/file.py:graph"
  },
  "env": "./.env",
  "python_version": "3.11"
}
```

---

### 6. **sdk-py** - Python SDK 🐍
Client library for interacting with LangGraph Server API:

```python
from langgraph_sdk import get_client

client = get_client()
assistants = await client.assistants.search()
thread = await client.threads.create()

async for chunk in client.runs.stream(thread_id, assistant_id, input=input):
    print(chunk)
```

---

### 7. **sdk-js** - JavaScript/TypeScript SDK 📘
JS/TS SDK for interacting with LangGraph REST API (minimal in Python repo).

---

## 🎯 Core Concepts

### 1. State Graph
Central abstraction for building stateful workflows:
```python
from langgraph.graph import StateGraph

graph = StateGraph(StateSchema)
graph.add_node("node1", node1_function)
graph.add_node("node2", node2_function)
graph.add_edge("node1", "node2")
graph.set_entry_point("node1")
app = graph.compile(checkpointer=checkpointer)
```

### 2. Checkpointing
Automatic state persistence:
- State saved at every "superstep"
- Resume from any checkpoint
- Time-travel debugging
- Human-in-the-loop patterns

### 3. Channels
Different state update strategies:
- **LastValue**: Keep only the latest value
- **BinaryOperatorAggregate**: Reduce values (sum, concat, etc.)
- **Topic**: Pub/sub message passing
- **EphemeralValue**: Temporary state

### 4. Pregel-Inspired Execution
Based on Google's Pregel graph processing model:
- Superstep-based execution
- Message passing between nodes
- Parallel execution where possible
- Fault-tolerant with checkpointing

---

## 🚀 Key Features

```mermaid
mindmap
  root((LangGraph))
    Durable Execution
      Auto-resume from failures
      Long-running workflows
      Checkpoint recovery
    Human-in-the-Loop
      Inspect state
      Modify execution
      Approval workflows
    Memory
      Short-term state
      Long-term persistence
      Thread-based isolation
    Deployment
      LangGraph Studio
      LangSmith Integration
      Docker support
      Scalable infrastructure
```

---

## 📂 Directory Layout

```
langgraph/
├── libs/                          # All libraries
│   ├── checkpoint/               # ✅ Base checkpointer interfaces
│   ├── checkpoint-postgres/      # 🐘 Postgres implementation
│   ├── checkpoint-sqlite/        # 📦 SQLite implementation
│   ├── cli/                      # 🛠️ Command-line tools
│   ├── langgraph/                # ⚡ Core framework
│   │   ├── _internal/           # Internal utilities
│   │   ├── channels/            # State channels
│   │   ├── graph/               # Graph builders
│   │   ├── managed/             # Managed values
│   │   ├── pregel/              # Execution engine
│   │   └── utils/               # Helper utilities
│   ├── prebuilt/                # 🎯 High-level APIs
│   ├── sdk-js/                  # 📘 JS/TS SDK
│   └── sdk-py/                  # 🐍 Python SDK
├── docs/                         # Documentation
│   ├── docs/
│   │   ├── concepts/            # Core concepts
│   │   ├── tutorials/           # Getting started
│   │   ├── how-tos/             # Guides
│   │   └── reference/           # API reference
│   └── mkdocs.yml               # Docs configuration
├── examples/                     # Example implementations
│   ├── rag/                     # RAG examples
│   ├── multi_agent/             # Multi-agent systems
│   ├── memory/                  # Memory patterns
│   ├── reflection/              # Self-reflection agents
│   ├── rewoo/                   # ReWOO implementation
│   └── ...
├── AGENTS.md                     # Development guide
├── CONTRIBUTING.md               # Contribution guidelines
└── README.md                     # Main readme
```

---

## 🔄 Typical Workflow

```mermaid
sequenceDiagram
    participant Dev as Developer
    participant CLI as LangGraph CLI
    participant Core as LangGraph Core
    participant CP as Checkpointer
    participant Store as Database

    Dev->>CLI: langgraph new my-agent
    CLI->>Dev: Project created

    Dev->>CLI: langgraph dev
    CLI->>Core: Start development server
    
    Dev->>Core: Build StateGraph
    Core->>Core: Configure nodes & edges
    
    Dev->>Core: graph.compile(checkpointer)
    Core->>CP: Initialize checkpointer
    CP->>Store: Connect to database
    
    Dev->>Core: app.invoke(input, config)
    
    loop For each superstep
        Core->>Core: Execute nodes
        Core->>CP: Save checkpoint
        CP->>Store: Persist state
    end
    
    Core->>Dev: Return result
    
    Dev->>CLI: langgraph build
    CLI->>Dev: Docker image ready
    
    Dev->>CLI: langgraph up
    CLI->>Dev: Agent deployed
```

---

## 🧪 Development Commands

Each library supports:
```bash
cd libs/<library-name>
make format    # Run code formatters
make lint      # Run linter
make test      # Execute test suite
```

---

## 🌟 Key Benefits

1. **Durable Execution**: Persist through failures, resume automatically
2. **Human-in-the-Loop**: Inspect and modify state at any point
3. **Comprehensive Memory**: Short-term and long-term stateful memory
4. **Production Ready**: Scalable deployment with LangGraph Platform
5. **Debugging**: LangSmith integration for deep visibility
6. **Low-Level Control**: No abstraction of prompts or architecture

---

## 🔗 Integration Ecosystem

```mermaid
graph LR
    LG[LangGraph]
    LS[LangSmith<br/>Observability & Evals]
    LC[LangChain<br/>Components & Integrations]
    STUDIO[LangGraph Studio<br/>Visual Development]
    DEPLOY[LangSmith Deployment<br/>Production Platform]

    LG <--> LS
    LG <--> LC
    LG <--> STUDIO
    LG --> DEPLOY

    style LG fill:#ff9800
    style LS fill:#4caf50
    style STUDIO fill:#2196f3
    style DEPLOY fill:#9c27b0
```

---

## 📚 Additional Resources

- **Docs**: https://langchain-ai.github.io/langgraph/
- **API Reference**: https://reference.langchain.com/python/langgraph/
- **Examples**: 60+ examples in `/examples`
- **Tutorials**: Step-by-step guides in `/docs/docs/tutorials`
- **LangChain Academy**: Free course on LangGraph basics
- **Forum**: https://forum.langchain.com/

---

## 🎓 Learning Path

```mermaid
graph TD
    START([Start Here]) --> BASIC[Basic Chatbot Tutorial]
    BASIC --> PREBUILT[Use Prebuilt Agents]
    PREBUILT --> STATE[Build Custom StateGraph]
    STATE --> CP[Add Checkpointing]
    CP --> HITL[Human-in-the-Loop]
    HITL --> MULTI[Multi-Agent Systems]
    MULTI --> DEPLOY[Deploy to Production]

    style START fill:#4caf50
    style DEPLOY fill:#ff9800
```

---

## 📝 Summary

**LangGraph** is a comprehensive framework for building production-grade AI agents with:
- **8 libraries** organized as a monorepo
- **Pregel-inspired** execution engine for stateful workflows
- **Flexible persistence** via pluggable checkpointers
- **Developer-friendly** CLI and SDKs
- **Production-ready** deployment options
- **Rich ecosystem** with LangChain and LangSmith

The architecture separates concerns beautifully:
- **checkpoint**: Persistence abstraction
- **langgraph**: Core orchestration
- **prebuilt**: Quick-start components
- **cli/sdk**: Developer tooling

This makes LangGraph suitable for everything from simple chatbots to complex multi-agent systems running in production at scale.


