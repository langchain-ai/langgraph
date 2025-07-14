# LangGraph API Reference

**LangGraph** is a low-level orchestration framework for building, managing, and deploying long-running, stateful agents. It provides a graph-based approach to building complex AI workflows with persistence, human-in-the-loop capabilities, and streaming support.

## Installation

```bash
pip install langgraph

# Optional: Install with specific checkpoint backends
pip install langgraph[postgres]  # PostgreSQL checkpointer
pip install langgraph[sqlite]    # SQLite checkpointer

# CLI tools
pip install langgraph-cli
```

## Quick Start

```python
from langgraph.graph import StateGraph, MessagesState, START, END
from langgraph.prebuilt import create_react_agent
from langchain_openai import ChatOpenAI

# Simple message-based graph
def chatbot(state: MessagesState):
    return {"messages": [AIMessage(content="Hello!")]}

graph = StateGraph(MessagesState)
graph.add_node("chat", chatbot)
graph.add_edge(START, "chat")
graph.add_edge("chat", END)

app = graph.compile()
result = app.invoke({"messages": [HumanMessage(content="Hi")]})

# Or use prebuilt ReAct agent
agent = create_react_agent(
    model=ChatOpenAI(model="gpt-4"),
    tools=[my_tool1, my_tool2]
)
```


---

# Core Graph API

## Main Entry Points

### StateGraph
**Import**: `from langgraph.graph import StateGraph`

Main class for building stateful graphs with custom state schemas.

```python
class StateGraph(Generic[StateT, InputT, OutputT]):
    def __init__(
        self,
        state_schema: type[StateT],
        config_schema: type[Any] | None = None,
        *,
        input_schema: type[InputT] | None = None,
        output_schema: type[OutputT] | None = None,
    )
```

#### Key Methods:
- **`add_node(node, action=None, *, defer=False, metadata=None, input_schema=None, retry_policy=None, cache_policy=None, destinations=None)`**
  - Adds a node to the graph
  - Can accept node name + action or just a callable as node

- **`add_edge(start_key: str | list[str], end_key: str)`**
  - Adds directed edge(s) between nodes
  - Supports multiple start nodes (waits for ALL to complete)

- **`add_conditional_edges(source: str, path: Callable, path_map: dict | list | None = None)`**
  - Adds conditional routing based on path function output
  - Path function determines next node(s) based on state

- **`compile(checkpointer=None, *, cache=None, store=None, interrupt_before=None, interrupt_after=None, debug=False, name=None)`**
  - Compiles graph into executable `CompiledStateGraph`
  - Returns runnable graph with persistence and streaming capabilities

### MessageGraph
**Import**: `from langgraph.graph import MessageGraph`

Specialized graph for message-based workflows.

```python
class MessageGraph(StateGraph):
    # Inherits from StateGraph with MessagesState as default state
```

### MessagesState
**Import**: `from langgraph.graph import MessagesState`

```python
class MessagesState(TypedDict):
    messages: list[BaseMessage]
```

### CompiledStateGraph (Pregel)
Returned by `StateGraph.compile()`. Main execution interface.

#### Key Methods:
- **`invoke(input, config=None, *, stream_mode="values", output_keys=None, interrupt_before=None, interrupt_after=None)`**
  - Executes graph synchronously
  - Returns final state

- **`stream(input, config=None, *, stream_mode=None, output_keys=None, interrupt_before=None, interrupt_after=None)`**
  - Executes graph with streaming output
  - Yields intermediate states/values

- **`get_state(config, *, subgraphs=False)`**
  - Retrieves current state snapshot
  - Returns `StateSnapshot` with values, next steps, metadata

- **`update_state(config, values, as_node=None, task_id=None)`**
  - Updates graph state manually
  - Can simulate updates from specific nodes

- **`get_state_history(config, *, filter=None, before=None)`**
  - Returns iterator over historical states
  - Requires checkpointer for persistence

## Utility Functions

### add_messages
**Import**: `from langgraph.graph import add_messages`

```python
def add_messages(
    left: Messages,
    right: Messages,
    *,
    format: Literal["langchain-openai"] | None = None,
) -> Messages
```
Reducer function for combining message lists in state.

## Constants

### Graph Flow Control
**Import**: `from langgraph.constants import START, END`

- **`START`**: Entry point identifier (`"__start__"`)
- **`END`**: Exit point identifier (`"__end__"`)

## Channels

**Import**: `from langgraph.channels import LastValue, Topic, BinaryOperatorAggregate, UntrackedValue, EphemeralValue, AnyValue`

State management primitives:
- **`LastValue`**: Stores most recent value
- **`Topic`**: Pub/sub style channel
- **`BinaryOperatorAggregate`**: Combines values with binary operations
- **`UntrackedValue`**: Values not included in checkpoints
- **`EphemeralValue`**: Temporary values cleared after each step
- **`AnyValue`**: Accepts any value type

## Managed Values

**Import**: `from langgraph.managed import IsLastStep, RemainingSteps`

- **`IsLastStep`**: Boolean indicating if current step is final
- **`RemainingSteps`**: Number of remaining steps in execution

## Error Handling

**Import**: `from langgraph.errors import GraphRecursionError, InvalidUpdateError, GraphInterrupt, NodeInterrupt`

Key exceptions:
- **`GraphRecursionError`**: Raised when max steps exceeded
- **`InvalidUpdateError`**: Invalid state update attempt
- **`GraphInterrupt`**: Graph execution interruption
- **`NodeInterrupt`**: Node-level interruption

## Types

**Import**: `from langgraph.types import Send, Interrupt, Command`

- **`Send`**: Object for sending messages between nodes
- **`Interrupt`**: Interrupt execution with value
- **`Command`**: Base class for graph commands

## Configuration

**Import**: `from langgraph.config import RunnableConfig`

Configuration object for graph execution with:
- `thread_id`: Unique identifier for conversation thread
- `checkpoint_id`: Specific checkpoint to resume from
- `checkpoint_ns`: Namespace for nested graphs
- `configurable`: Additional configuration parameters


---

# Checkpoint System

## Overview
LangGraph checkpointers allow agents to persist their state within and across multiple interactions. The checkpoint system provides a unified interface for storing and retrieving graph execution state with support for multiple backends.

## Base Checkpoint Interfaces

### BaseCheckpointSaver
```python
from langgraph.checkpoint.base import BaseCheckpointSaver, Checkpoint, CheckpointMetadata, CheckpointTuple
from langchain_core.runnables import RunnableConfig

class BaseCheckpointSaver(Generic[V]):
    """Base class for creating a graph checkpointer.
    
    Checkpointers allow LangGraph agents to persist their state
    within and across multiple interactions.
    """
    
    serde: SerializerProtocol = JsonPlusSerializer()
    
    def get(self, config: RunnableConfig) -> Checkpoint | None:
        """Fetch a checkpoint using the given configuration."""
    
    def get_tuple(self, config: RunnableConfig) -> CheckpointTuple | None:
        """Fetch a checkpoint tuple with metadata."""
    
    def put(
        self,
        config: RunnableConfig,
        checkpoint: Checkpoint,
        metadata: CheckpointMetadata,
        new_versions: ChannelVersions,
    ) -> RunnableConfig:
        """Store a checkpoint with its configuration and metadata."""
    
    def list(
        self,
        config: RunnableConfig | None,
        *,
        filter: dict[str, Any] | None = None,
        before: RunnableConfig | None = None,
        limit: int | None = None,
    ) -> Iterator[CheckpointTuple]:
        """List checkpoints that match the given criteria."""
```

### Core Data Types
```python
from langgraph.checkpoint.base import Checkpoint, CheckpointMetadata, CheckpointTuple

class Checkpoint(TypedDict):
    """State snapshot at a given point in time."""
    v: int  # Version of the checkpoint format
    id: str  # Unique checkpoint identifier
    ts: str  # Timestamp in ISO format
    channel_values: dict[str, Any]  # Current channel values
    channel_versions: dict[str, Any]  # Channel version information
    versions_seen: dict[str, dict[str, Any]]  # Version tracking
    pending_sends: list[Any]  # Pending send operations

class CheckpointMetadata(TypedDict, total=False):
    """Metadata associated with a checkpoint."""
    source: Literal["input", "loop", "update", "fork"]  # Checkpoint source
    step: int  # Step number (-1 for input, 0+ for loop steps)
    parents: dict[str, str]  # Parent checkpoint references

class CheckpointTuple(NamedTuple):
    """A tuple containing a checkpoint and its associated data."""
    config: RunnableConfig
    checkpoint: Checkpoint
    metadata: CheckpointMetadata
    parent_config: RunnableConfig | None
    next_config: RunnableConfig | None
```

## Memory Checkpointer

### InMemorySaver
```python
from langgraph.checkpoint.memory import InMemorySaver

class InMemorySaver(BaseCheckpointSaver[str], AbstractContextManager, AbstractAsyncContextManager):
    """An in-memory checkpoint saver.
    
    This checkpoint saver stores checkpoints in memory using a defaultdict.
    
    Note:
        Only use InMemorySaver for debugging or testing purposes.
        For production use cases, use PostgresSaver or SqliteSaver.
    """
    
    def __init__(self, serde: SerializerProtocol | None = None) -> None:
        """Initialize the in-memory saver."""

# Usage
checkpointer = InMemorySaver()

# Context manager support
with InMemorySaver() as checkpointer:
    # Use checkpointer
    pass

# Async context manager support  
async with InMemorySaver() as checkpointer:
    # Use checkpointer
    pass
```

## PostgreSQL Checkpointer

### PostgresSaver
```python
from langgraph.checkpoint.postgres import PostgresSaver, AsyncPostgresSaver
from psycopg import Connection, Pipeline
from psycopg_pool import ConnectionPool

class PostgresSaver(BasePostgresSaver):
    """Checkpointer that stores checkpoints in a Postgres database."""
    
    def __init__(
        self,
        conn: Connection | ConnectionPool,
        pipe: Pipeline | None = None,
        serde: SerializerProtocol | None = None,
    ) -> None:
        """Initialize PostgreSQL checkpointer.
        
        Args:
            conn: PostgreSQL connection or connection pool
            pipe: Optional pipeline for batch operations
            serde: Custom serializer (defaults to JsonPlusSerializer)
        """

class AsyncPostgresSaver(BasePostgresSaver):
    """Asynchronous checkpointer that stores checkpoints in a Postgres database."""
    
    def __init__(
        self,
        conn: AsyncConnection | AsyncConnectionPool,
        pipe: AsyncPipeline | None = None,
        serde: SerializerProtocol | None = None,
    ) -> None:
        """Initialize async PostgreSQL checkpointer."""

# Usage Examples
import psycopg
from psycopg_pool import ConnectionPool

# Synchronous usage
conn = psycopg.connect("postgresql://user:pass@localhost/db")
checkpointer = PostgresSaver(conn)

# With connection pool
pool = ConnectionPool("postgresql://user:pass@localhost/db")
checkpointer = PostgresSaver(pool)

# Async usage
import asyncio
from psycopg import AsyncConnection

async def main():
    conn = await AsyncConnection.connect("postgresql://user:pass@localhost/db")
    checkpointer = AsyncPostgresSaver(conn)
```

### ShallowPostgresSaver
```python
from langgraph.checkpoint.postgres import ShallowPostgresSaver

class ShallowPostgresSaver(BasePostgresSaver):
    """PostgreSQL checkpointer that stores only checkpoint metadata, not full state.
    
    Useful for scenarios where you only need to track checkpoint history
    without storing the complete state data.
    """
```

## SQLite Checkpointer

### SqliteSaver
```python
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
import sqlite3

class SqliteSaver(BaseCheckpointSaver[str]):
    """A checkpoint saver that stores checkpoints in a SQLite database.
    
    Note:
        This class is meant for lightweight, synchronous use cases
        and does not scale to multiple threads.
    """
    
    def __init__(
        self,
        conn: sqlite3.Connection,
        serde: SerializerProtocol | None = None,
    ) -> None:
        """Initialize SQLite checkpointer.
        
        Args:
            conn: SQLite database connection
            serde: Custom serializer (defaults to JsonPlusSerializer)
        """

class AsyncSqliteSaver(BaseCheckpointSaver[str]):
    """An asynchronous checkpoint saver that stores checkpoints in a SQLite database.
    
    This class provides an asynchronous interface for saving and retrieving checkpoints
    using a SQLite database with aiosqlite.
    """
    
    def __init__(
        self,
        conn: aiosqlite.Connection,
        serde: SerializerProtocol | None = None,
    ) -> None:
        """Initialize async SQLite checkpointer."""

# Usage Examples
import sqlite3

# Synchronous usage
conn = sqlite3.connect("checkpoints.db")
checkpointer = SqliteSaver(conn)

# Async usage (requires aiosqlite)
import aiosqlite

async def main():
    conn = await aiosqlite.connect("checkpoints.db")
    checkpointer = AsyncSqliteSaver(conn)
```

## Serialization Classes

### JsonPlusSerializer
```python
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer

class JsonPlusSerializer(SerializerProtocol):
    """Serializer that uses ormsgpack, with a fallback to extended JSON serializer.
    
    Supports serialization of complex Python objects including:
    - LangChain Serializable objects
    - Pydantic models
    - Dataclasses and NamedTuples
    - Pathlib paths, UUIDs, decimals
    - Sets, frozensets, deques
    """
    
    def __init__(
        self,
        *,
        pickle_fallback: bool = False,
        __unpack_ext_hook__: Callable[[int, bytes], Any] | None = None,
    ) -> None:
        """Initialize JsonPlusSerializer.
        
        Args:
            pickle_fallback: Whether to use pickle as fallback for unsupported types
            __unpack_ext_hook__: Custom extension unpacking hook
        """
    
    def dumps(self, obj: Any) -> bytes:
        """Serialize object to bytes."""
    
    def loads(self, data: bytes) -> Any:
        """Deserialize object from bytes."""
    
    def dumps_typed(self, obj: Any) -> tuple[str, bytes]:
        """Serialize object to typed tuple."""
    
    def loads_typed(self, data: tuple[str, bytes]) -> Any:
        """Deserialize object from typed tuple."""
```

### EncryptedSerializer
```python
from langgraph.checkpoint.serde.encrypted import EncryptedSerializer

class EncryptedSerializer(SerializerProtocol):
    """Serializer that encrypts and decrypts data using an encryption protocol."""
    
    def __init__(
        self, 
        cipher: CipherProtocol, 
        serde: SerializerProtocol = JsonPlusSerializer()
    ) -> None:
        """Initialize encrypted serializer.
        
        Args:
            cipher: Encryption/decryption protocol implementation
            serde: Underlying serializer (defaults to JsonPlusSerializer)
        """
```


---

# Prebuilt Components

## Agent Executors

### create_react_agent
```python
from langgraph.prebuilt import create_react_agent

def create_react_agent(
    model: Union[str, LanguageModelLike],
    tools: Union[Sequence[Union[BaseTool, Callable, dict[str, Any]]], ToolNode],
    *,
    prompt: Optional[Prompt] = None,
    response_format: Optional[Union[StructuredResponseSchema, tuple[str, StructuredResponseSchema]]] = None,
    pre_model_hook: Optional[RunnableLike] = None,
    post_model_hook: Optional[RunnableLike] = None,
    state_schema: Optional[StateSchemaType] = None,
    config_schema: Optional[Type[Any]] = None,
    checkpointer: Optional[Checkpointer] = None,
    store: Optional[BaseStore] = None,
    interrupt_before: Optional[list[str]] = None,
    interrupt_after: Optional[list[str]] = None,
    debug: bool = False,
    version: Literal["v1", "v2"] = "v2",
    name: Optional[str] = None,
) -> CompiledStateGraph
```

**Description**: Creates a ReAct-style agent that calls tools in a loop until a stopping condition is met.

**Key Features**:
- Supports tool calling with LangChain chat models
- Optional structured response formatting
- Pre/post model hooks for message management and validation
- Built-in checkpointing and state persistence
- Human-in-the-loop interruption support

**Usage Pattern**:
```python
from langgraph.prebuilt import create_react_agent
from langchain_openai import ChatOpenAI

agent = create_react_agent(
    model=ChatOpenAI(model="gpt-4"),
    tools=[my_tool1, my_tool2],
    prompt="You are a helpful assistant"
)
```

## Tool Nodes

### ToolNode
```python
from langgraph.prebuilt import ToolNode

class ToolNode(RunnableCallable):
    def __init__(
        self,
        tools: Sequence[Union[BaseTool, Callable]],
        *,
        name: str = "tools",
        tags: Optional[list[str]] = None,
        handle_tool_errors: Union[bool, str, Callable[..., str], tuple[type[Exception], ...]] = True,
        messages_key: str = "messages",
    ) -> None
```

**Description**: A node that runs tools called in the last AIMessage. Executes multiple tool calls in parallel.

**Key Features**:
- Parallel tool execution
- Configurable error handling
- State and store injection support
- Works with StateGraph and MessageGraph

**Usage Pattern**:
```python
from langgraph.prebuilt import ToolNode

tool_node = ToolNode([search_tool, calculator_tool])
# Add to graph
graph.add_node("tools", tool_node)
```

### tools_condition
```python
from langgraph.prebuilt import tools_condition

def tools_condition(
    state: Union[list[AnyMessage], dict[str, Any], BaseModel],
    messages_key: str = "messages",
) -> Literal["tools", "__end__"]
```

**Description**: Conditional edge function that routes to ToolNode if the last message has tool calls, otherwise ends.

**Usage Pattern**:
```python
from langgraph.prebuilt import tools_condition

graph.add_conditional_edges(
    "agent",
    tools_condition,
    {"tools": "tools", "__end__": END}
)
```

## Tool Validation

### ValidationNode
```python
from langgraph.prebuilt import ValidationNode

class ValidationNode(RunnableCallable):
    def __init__(
        self,
        schemas: Sequence[Union[BaseTool, Type[BaseModel], Callable]],
        *,
        format_error: Optional[Callable[[BaseException, ToolCall, Type[BaseModel]], str]] = None,
        name: str = "validation",
        tags: Optional[list[str]] = None,
    ) -> None
```

**Description**: Validates tool calls against Pydantic schemas without executing them. Useful for structured output extraction.

**Key Features**:
- Schema validation for tool calls
- Custom error formatting
- Preserves original messages and tool IDs
- Parallel validation of multiple tool calls

**Usage Pattern**:
```python
from langgraph.prebuilt import ValidationNode
from pydantic import BaseModel

class MySchema(BaseModel):
    name: str
    age: int

validator = ValidationNode([MySchema])
```

## State Injection

### InjectedState
```python
from langgraph.prebuilt import InjectedState
from typing_extensions import Annotated

# Usage in tool definition
def my_tool(query: str, state: Annotated[dict, InjectedState]) -> str:
    # Tool implementation
    pass
```

**Description**: Annotation for tool arguments that should be populated with graph state.

### InjectedStore
```python
from langgraph.prebuilt import InjectedStore
from typing_extensions import Annotated

# Usage in tool definition  
def my_tool(query: str, store: Annotated[BaseStore, InjectedStore]) -> str:
    # Tool implementation
    pass
```

**Description**: Annotation for tool arguments that should be populated with LangGraph store.

## Human Interrupts

### HumanInterruptConfig
```python
from langgraph.prebuilt.interrupt import HumanInterruptConfig

class HumanInterruptConfig(TypedDict):
    allow_ignore: bool
    allow_respond: bool  
    allow_edit: bool
    allow_accept: bool
```

**Description**: Configuration for human interrupt behavior, defining available interaction options.

### ActionRequest
```python
from langgraph.prebuilt.interrupt import ActionRequest

class ActionRequest(TypedDict):
    action: str  # Action type or name
    args: dict   # Action arguments
```

**Description**: Represents a request for human action within graph execution.


---

# CLI Commands

## Project Management

### langgraph new
```bash
langgraph new [PATH] [--template TEMPLATE]
```

**Description**: Create a new LangGraph project from a template.

**Available Templates**:
- **New LangGraph Project**: Simple, minimal chatbot with memory
- **ReAct Agent**: Flexible agent extensible to many tools  
- **Memory Agent**: ReAct-style agent with memory storage across threads
- **Retrieval Agent**: Agent with retrieval-based QA system
- **Data Analyst**: Agent for data analysis tasks
- **Chatbot**: Basic conversational agent
- **Multi-Agent Collaboration**: Multiple agents working together

**Usage**:
```bash
langgraph new my-project --template "ReAct Agent"
langgraph new ./my-app  # Interactive template selection
```

## Development Server

### langgraph dev
```bash
langgraph dev [--host HOST] [--port PORT] [--no-reload]
```

**Description**: Run LangGraph API server in development mode with hot reloading and debugging support.

**Options**:
- `--host`: Host to bind to (default: localhost)
- `--port`: Port to run on (default: 8123)
- `--no-reload`: Disable auto-reload on file changes

**Usage**:
```bash
langgraph dev --port 8000
langgraph dev --host 0.0.0.0 --no-reload
```

## Production Deployment

### langgraph up
```bash
langgraph up [--config CONFIG] [--docker-compose COMPOSE_FILE] [--port PORT]
```

**Description**: Launch LangGraph API server in production mode.

**Options**:
- `--config, -c`: Path to configuration file (default: langgraph.json)
- `--docker-compose, -d`: Path to docker-compose.yml with additional services
- `--port`: Port to expose (default: 8123)

**Usage**:
```bash
langgraph up --config ./my-config.json
langgraph up --port 8080 --docker-compose ./docker-compose.yml
```

### langgraph build
```bash
langgraph build [--config CONFIG] [--base-image IMAGE] [BUILD_ARGS...]
```

**Description**: Build LangGraph API server Docker image.

**Options**:
- `--config, -c`: Path to configuration file
- `--base-image`: Base Docker image to use
- Additional Docker build arguments

**Usage**:
```bash
langgraph build --config ./langgraph.json
langgraph build --base-image python:3.11-slim
```

## Docker Integration

### langgraph dockerfile
```bash
langgraph dockerfile [--save-path PATH] [--config CONFIG] [--add-docker-compose]
```

**Description**: Generate Dockerfile for LangGraph API server with optional Docker Compose configuration.

**Options**:
- `--save-path`: Where to save the generated Dockerfile
- `--config, -c`: Configuration file to use
- `--add-docker-compose`: Also generate docker-compose.yml

**Usage**:
```bash
langgraph dockerfile --save-path ./Dockerfile
langgraph dockerfile --add-docker-compose --config ./langgraph.json
```

## Configuration

### Configuration File (langgraph.json)
```json
{
  "dependencies": [
    ".",
    "./local_package", 
    "package_name"
  ],
  "graphs": {
    "agent": "./my_package/agent.py:graph"
  },
  "env": {
    "OPENAI_API_KEY": "your-key"
  },
  "python_version": "3.11",
  "pip_config_file": "./pip.conf"
}
```

**Key Configuration Options**:
- `dependencies`: Python packages and local paths to install
- `graphs`: Mapping of graph IDs to module paths
- `env`: Environment variables
- `python_version`: Python version for Docker image
- `pip_config_file`: Custom pip configuration


---

# Python SDK

## Client Interface

### LangGraphClient
```python
from langgraph_sdk import get_client, get_sync_client

# Async client
client = get_client(url="http://localhost:2024")

# Sync client
sync_client = get_sync_client(url="http://localhost:2024")
```

**Description**: The LangGraph client implementations connect to the LangGraph API, providing both asynchronous and synchronous clients for interacting with core resources such as Assistants, Threads, Runs, and Cron jobs, as well as the persistent document Store.

### Authentication
```python
from langgraph_sdk.auth import Auth

# Initialize authentication
auth = Auth()
```

**Description**: Authentication module for LangGraph SDK providing secure access to LangGraph API endpoints.

---

# Usage Examples

## Basic StateGraph Example
```python
from langgraph.graph import StateGraph, START, END
from typing import TypedDict

class State(TypedDict):
    messages: list[str]
    count: int

def node_func(state: State) -> State:
    return {"count": state["count"] + 1}

graph = StateGraph(State)
graph.add_node("process", node_func)
graph.add_edge(START, "process")
graph.add_edge("process", END)

app = graph.compile()
result = app.invoke({"messages": [], "count": 0})
```

## MessageGraph with Streaming
```python
from langgraph.graph import MessageGraph, MessagesState, add_messages
from langchain_core.messages import AIMessage, HumanMessage

def chatbot(state: MessagesState):
    return {"messages": [AIMessage(content="Hello!")]}

graph = MessageGraph()
graph.add_node("chat", chatbot)
graph.add_edge(START, "chat")
graph.add_edge("chat", END)

app = graph.compile()
for chunk in app.stream({"messages": [HumanMessage(content="Hi")]}):
    print(chunk)
```

## Conditional Routing
```python
def route_condition(state: State) -> str:
    if state["count"] > 5:
        return "end_node"
    return "continue_node"

graph.add_conditional_edges(
    "process",
    route_condition,
    {"end_node": END, "continue_node": "process"}
)
```

## Persistence with Checkpointing
```python
from langgraph.checkpoint.memory import InMemorySaver
from langchain_core.runnables import RunnableConfig

# Initialize checkpointer
checkpointer = InMemorySaver()

# Compile graph with checkpointer
app = graph.compile(checkpointer=checkpointer)

# Create configuration with thread ID
config = RunnableConfig(configurable={"thread_id": "thread-1"})

# Run with persistence
result = app.invoke({"messages": [], "count": 0}, config=config)

# Continue from checkpoint
result2 = app.invoke({"messages": ["new message"]}, config=config)
```

## Production Database Setup
```python
# PostgreSQL production setup
from langgraph.checkpoint.postgres import PostgresSaver
from psycopg_pool import ConnectionPool

pool = ConnectionPool(
    "postgresql://user:password@localhost:5432/langgraph",
    min_size=1,
    max_size=10
)
checkpointer = PostgresSaver(pool)

# SQLite production setup  
from langgraph.checkpoint.sqlite import SqliteSaver
import sqlite3

conn = sqlite3.connect("production.db", check_same_thread=False)
checkpointer = SqliteSaver(conn)
```

## ReAct Agent with Tools
```python
from langgraph.prebuilt import create_react_agent
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool

@tool
def search_tool(query: str) -> str:
    """Search for information."""
    return f"Search results for: {query}"

@tool
def calculator(expression: str) -> str:
    """Calculate mathematical expressions."""
    try:
        return str(eval(expression))
    except:
        return "Invalid expression"

# Create agent
agent = create_react_agent(
    model=ChatOpenAI(model="gpt-4"),
    tools=[search_tool, calculator],
    checkpointer=InMemorySaver()
)

# Run agent
config = RunnableConfig(configurable={"thread_id": "agent-1"})
result = agent.invoke(
    {"messages": [HumanMessage(content="What is 2+2 and search for python")]},
    config=config
)
```

## Human-in-the-Loop
```python
from langgraph.graph import StateGraph, START, END
from langgraph.types import Interrupt

def human_approval_node(state):
    # Request human approval
    raise Interrupt({"question": "Do you approve this action?", "data": state})

graph = StateGraph(State)
graph.add_node("process", process_node)
graph.add_node("approval", human_approval_node)
graph.add_node("execute", execute_node)

graph.add_edge(START, "process")
graph.add_edge("process", "approval")
graph.add_edge("approval", "execute")
graph.add_edge("execute", END)

app = graph.compile(checkpointer=checkpointer)

# Run until interrupt
try:
    result = app.invoke(input_data, config=config)
except GraphInterrupt as e:
    print(f"Human input needed: {e.value}")
    
    # Resume after human input
    app.update_state(config, {"approved": True})
    result = app.invoke(None, config=config)
```

## CLI Quick Start
```bash
# Create new project
langgraph new my-agent --template "ReAct Agent"
cd my-agent

# Run in development
langgraph dev

# Build for production
langgraph build
langgraph up
```

---

# Key Features

- **Graph-based Architecture**: Build complex workflows as directed graphs
- **State Management**: Persistent state across graph execution steps
- **Streaming Support**: Real-time streaming of intermediate results
- **Checkpointing**: Built-in persistence with multiple backend options
- **Human-in-the-Loop**: Interrupt execution for human input and approval
- **Tool Integration**: Seamless integration with LangChain tools
- **Async Support**: Full async/await support throughout the framework
- **Production Ready**: CLI tools for deployment and Docker integration
- **Multi-Agent Support**: Coordinate multiple agents in complex workflows
- **Error Handling**: Robust error handling and retry mechanisms

This comprehensive API reference covers all the essential components needed to build, deploy, and manage LangGraph applications effectively.
