---
type: "Reference"
title: "Synchronous invocation"
openwiki_generated: true
verified:
  - by: openwiki/0.5.0
    at: 2026-09-03T15:19:56.419Z
sources:
  - id: openwiki-source-9b01a2a24dd7f7a0fb7e6f05
    resource: repo://libs/langgraph/langgraph/graph/message.py
  - id: openwiki-source-8b2910f50b9e167d64d2728e
    resource: repo://libs/prebuilt/langgraph/prebuilt/chat_agent_executor.py
  - id: openwiki-source-df24839a466a07d274e3777a
    resource: repo://libs/prebuilt/langgraph/prebuilt/tool_node.py
  - id: openwiki-source-2e9a7cdfb7a3c8dffbac1ef2
    resource: repo://libs/prebuilt/langgraph/prebuilt/tool_validator.py
  - id: openwiki-source-947a0c3d1c7e56087c7c9d5e
    resource: repo://libs/prebuilt/README.md
generated: { by: "openwiki/0.5.0", at: "2026-09-03T15:19:56.419Z" }
---


## Overview

The `langgraph.prebuilt` module provides factory functions and composable components for building tool-using agents quickly. Instead of manually constructing graphs with model nodes, tool selection logic, and tool execution, developers can use `create_react_agent()` to build a complete ReAct-style agent in one call, or use lower-level components like `ToolNode` and `tools_condition` for custom workflows.

**Core Exports:**

- **`create_react_agent()`**: Factory function that builds a compiled agent graph for tool-calling workflows.
- **`ToolNode`**: Runnable node that executes tool calls extracted from model outputs.
- **`tools_condition`**: Conditional routing function for standard tool-calling patterns.
- **`ValidationNode`**: Legacy component for validating tool call arguments against Pydantic schemas (deprecated).
- **`InjectedState`, `InjectedStore`**: Annotation helpers for injecting graph context into tools.
- **`ToolRuntime`**: Runtime context object automatically injected into tool parameters.

---

## Agent Loop Architecture

### Model Node → Tool Selection → Tool Execution → Loop

The prebuilt agent implements a standard ReAct loop:

1. **Agent Node (LLM Call)**: Model receives messages (including system prompt if provided), outputs an `AIMessage` with optional `tool_calls`.
2. **Should Continue**: Conditional edge examines the last `AIMessage` for tool calls.
   - If **tool calls present**: Route to the tool execution node.
   - If **no tool calls**: Proceed to post-model hook (if configured) or end.
3. **Tool Node**: Executes each tool call in parallel, producing `ToolMessage` objects with results or errors.
4. **Route Tool Responses**: If a tool has `return_direct=True`, end immediately. Otherwise, loop back to agent node.

This cycle continues until the model generates a response without tool calls or a tool with `return_direct=True` is executed.

### State Management

Agent state is built on a **messages list** with an `add_messages` reducer:

```python
from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph.message import add_messages
from langchain_core.messages import AnyMessage

class AgentState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]
    remaining_steps: int  # Tracks recursion limit progress
```

- **Messages List**: Accumulates all messages (user, assistant, tool results, system) in order.
- **Reducer**: The `add_messages` reducer deduplicates messages by ID, appending new messages and replacing older ones with matching IDs.
- **Remaining Steps**: Used to enforce iteration limits; agents check this value before continuing if tool calls are present.

---

## `create_react_agent()`: Agent Factory

### Function Signature & Key Parameters

```python
def create_react_agent(
    model: str | LanguageModelLike | Callable,
    tools: Sequence[BaseTool | Callable] | ToolNode,
    *,
    prompt: str | SystemMessage | Callable | Runnable | None = None,
    response_format: dict | type[BaseModel] | tuple[str, dict | type[BaseModel]] | None = None,
    pre_model_hook: RunnableLike | None = None,
    post_model_hook: RunnableLike | None = None,
    state_schema: type[AgentState] | None = None,
    context_schema: type[Any] | None = None,
    checkpointer: Checkpointer | None = None,
    store: BaseStore | None = None,
    interrupt_before: list[str] | None = None,
    interrupt_after: list[str] | None = None,
    version: Literal["v1", "v2"] = "v2",
    name: str | None = None,
    debug: bool = False,
) -> CompiledStateGraph
```

### Model Selection

The `model` parameter supports three patterns:

1. **Static Model String** (e.g., `"anthropic:claude-3-7-sonnet-latest"`):
   - Model identifier automatically resolved via `langchain.chat_models.init_chat_model()`.
   - Tools are automatically bound to the model via `.bind_tools()`.

2. **Static ChatModel Instance** (e.g., `ChatAnthropic(...)`):
   - Model used as-is; tools bound automatically if not already bound.
   - Pre-bound tools (via `.bind_tools()`) must be a subset of the `tools` parameter.

3. **Dynamic Model Callable** (e.g., `Callable[[state, runtime], BaseChatModel]`):
   - Called at runtime with graph state and runtime context.
   - Enables context-aware model selection (e.g., different models for different complexity levels).
   - Can be sync or async; async callables require `.ainvoke()` or `.astream()` on the graph.
   - Returned model can have tools pre-bound via `.bind_tools()`.

### Tools Configuration

- **Empty list**: Creates an agent without tool calling; graph runs model once and stops.
- **Tool list**: Functions or `BaseTool` instances converted to tools automatically.
- **ToolNode instance**: Pass a pre-configured `ToolNode` for fine-grained control over execution and error handling.

### Prompting

The `prompt` parameter accepts:

- **`None`**: Uses only messages from state without a system message.
- **`str`**: Converted to a `SystemMessage` prepended to messages during each LLM call.
- **`SystemMessage`**: Used as-is, prepended to messages.
- **`Callable` or `Runnable`**: Receives full state and outputs messages or `LanguageModelInput` for the LLM.

### Structured Output

If `response_format` is provided, the agent makes an additional LLM call after the main loop ends to extract structured output:

- **Schema Only**: Used as-is for structured output extraction.
- **Tuple** `(prompt, schema)`: Prepends the prompt message before extraction.
- Requires the model to support `.with_structured_output()`.

### Pre/Post Model Hooks

- **`pre_model_hook`**: Node called before each LLM invocation; can trim/summarize message history.
  - Must return `{"messages": [...]}` or `{"llm_input_messages": [...]}` plus any other state keys.
  - Useful for managing token limits or message history length.

- **`post_model_hook`**: Node called after each LLM invocation (v2 only); can validate, reject, or modify model output.
  - Returns state updates; can route to specialized nodes or back to agent.
  - Enables human-in-the-loop workflows.

### Version Differences (v1 vs v2)

| Aspect | v1 | v2 |
|--------|----|----|
| **Tool Execution** | Single node processes all tool calls in parallel from one message | Tool node receives individual tool calls via `Send` API |
| **Scalability** | All tools run in one batch | Each tool runs in separate task; better for human-in-the-loop |
| **Post-Model Hook** | Not supported | Supported |
| **Streaming** | Tools update state together | Tools update state independently |

**Recommendation**: Use v2 (default) for new agents, especially when using `post_model_hook` or expecting human interrupts.

### Interrupts & Checkpointing

- **`interrupt_before`**: List of node names (`["agent"]`, `["tools"]`) where execution pauses before the node runs.
- **`interrupt_after`**: List of node names where execution pauses after the node completes.
- **`checkpointer`**: Checkpoint saver for persisting state (e.g., for chat memory).
- **`store`**: Cross-thread persistent store for multi-turn context.

### Configuration Schema

- **`state_schema`**: Custom state TypedDict or Pydantic model. Must include `messages` and `remaining_steps` keys; can optionally include `structured_response` for response formatting.
- **`context_schema`**: Schema for runtime context shared across nodes (replaces deprecated `config_schema`).

### Return Value

Returns a **`CompiledStateGraph`** runnable:

```python
agent = create_react_agent(model, tools)
# Synchronous invocation
result = agent.invoke({"messages": [HumanMessage("What is 2+2?")]})

# Streaming
for chunk in agent.stream({"messages": [...]}, stream_mode="updates"):
    print(chunk)

# Async
result = await agent.ainvoke({"messages": [...]})
```

---

## `ToolNode`: Tool Execution Engine

`ToolNode` is a `RunnableCallable` that executes tool calls extracted from LLM outputs and returns `ToolMessage` objects with results or error messages.

### When to Use ToolNode

- **Standalone**: For custom graph workflows requiring fine-grained tool execution control.
- **Direct Invocation**: For testing tool execution or programmatic tool calling.
- **Error Handling**: To apply custom error handling strategies beyond defaults.
- **Existing Graphs**: To replace manual tool execution logic in hand-built graphs.

For standard agents, `create_react_agent()` configures `ToolNode` internally.

### Initialization

```python
from langgraph.prebuilt import ToolNode

tool_node = ToolNode(
    tools=[tool1, tool2],  # Functions or BaseTool instances
    name="tools",
    handle_tool_errors=True,  # or False, str, Callable, or Exception type(s)
    messages_key="messages",  # State key containing messages list
    wrap_tool_call=None,  # Optional sync wrapper for tool execution
    awrap_tool_call=None,  # Optional async wrapper for tool execution
)
```

### Input Formats

1. **Graph State (Dict)**: `{"messages": [AIMessage(..., tool_calls=[...])]}`
   - Extracts tool calls from the last `AIMessage` in the messages list.
   - Returns `{"messages": [ToolMessage(...)]}`

2. **Message List**: `[AIMessage(..., tool_calls=[...])]`
   - Direct message list; tool calls extracted from last message.
   - Returns `[ToolMessage(...)]`

3. **Direct Tool Calls**: `[{"name": "...", "args": {...}, "id": "...", "type": "tool_call"}]`
   - Programmatic tool invocation without message context.
   - Returns `[ToolMessage(...)]` or `[Command(...)]`

### Output Formats

- **Regular Tools**: Returns `ToolMessage` objects with results or error content.
- **Command Tools**: Returns `Command` objects for state updates, navigation, or specialized control flow.
- **Mixed**: List containing both `ToolMessage` and `Command` instances; LangGraph handles routing.

### Tool Execution Lifecycle

1. **Parse Input**: Extract tool calls from messages or direct format.
2. **Validate Availability**: Check that each tool exists in `tools_by_name`; raise error if not found.
3. **Inject Context**: Inject state, store, runtime, and other dependencies (see below).
4. **Execute in Parallel**: Run all tools concurrently using thread pool or async executor.
5. **Wrap Results**: Convert outputs to `ToolMessage` or `Command` with error handling.
6. **Format Output**: Return according to input type.

### Tool Validation & Error Handling

**Validation Errors** (invalid arguments provided by the model):
- Caught and returned as a `ToolMessage` with a user-friendly error.
- LLM receives error feedback and can retry with corrected arguments.
- Excludes injected arguments (state, store, runtime) from error messages; LLM has no control over those.

**Execution Errors** (tool logic failures):
- By default, re-raised; graph halts unless caught by higher-level error handler.
- Controlled by `handle_tool_errors` parameter:
  - `True`: Catch all errors, return default error template.
  - `str`: Catch all errors, return custom error message string.
  - `type[Exception]` or `tuple[type[Exception], ...]`: Catch only specified types.
  - `Callable[..., str]`: Catch types matching the callable's first parameter annotation; return callable result.
  - `False`: Disable error handling; let errors propagate.

### State and Context Injection

Tools can request graph state, persistent stores, or runtime information using annotations:

```python
from typing import Annotated
from langgraph.prebuilt import InjectedState, InjectedStore, ToolRuntime

@tool
def my_tool(
    query: str,
    messages: Annotated[list, InjectedState("messages")],  # Inject state["messages"]
    full_state: Annotated[dict, InjectedState()],           # Inject entire state
    store: Annotated[BaseStore, InjectedStore()],           # Inject persistent store
    runtime: ToolRuntime,                                    # Inject runtime context
) -> str:
    """A tool that uses state, store, and runtime."""
    return f"Query: {query}, Msg count: {len(messages)}"
```

- **`InjectedState(field_name)`**: Inject a specific state field (e.g., `"messages"`).
- **`InjectedState()`**: Inject entire state as a dict.
- **`InjectedStore()`**: Inject the persistent store object.
- **`ToolRuntime`**: Dataclass containing `state`, `tool_call_id`, `config`, `context`, `store`, `stream_writer`, `tools`, `execution_info`, `server_info`.

The `ToolNode` automatically injects these at tool invocation; the LLM never sees them in tool schemas.

### Tool Execution Middleware: `wrap_tool_call`

For advanced workflows, provide a `wrap_tool_call` callable to intercept and modify tool execution:

```python
def my_wrapper(request: ToolCallRequest, execute: Callable) -> ToolMessage | Command:
    # request.tool_call: {"name": ..., "args": ..., "id": ...}
    # request.tool: BaseTool instance
    # request.state: Current graph state
    # request.runtime: ToolRuntime
    
    # Modify request if needed
    modified_request = request.override(tool_call={...})
    
    # Execute tool (can be called multiple times for retries)
    return execute(modified_request)
```

**Use Cases:**
- **Retries**: Attempt tool execution multiple times with fallback logic.
- **Caching**: Short-circuit execution if result is cached.
- **Request Modification**: Adjust tool arguments before execution.
- **Validation**: Inspect and approve tool calls before running them.

---

## `tools_condition()`: Routing Function

A utility function for conditional edges in custom graphs; implements standard ReAct routing logic:

```python
def tools_condition(
    state: list | dict | BaseModel,
    messages_key: str = "messages",
) -> Literal["tools", "__end__"]:
    """Route to tools if last message has tool calls, else end."""
```

**Returns:**
- `"tools"`: If the last `AIMessage` contains tool calls.
- `"__end__"`: Otherwise.

**Usage in StateGraph:**

```python
from langgraph.graph import StateGraph, START
from langgraph.prebuilt import ToolNode, tools_condition

graph = StateGraph(State)
graph.add_node("model", call_model)
graph.add_node("tools", ToolNode([my_tool]))
graph.add_edge(START, "model")
graph.add_conditional_edges(
    "model",
    tools_condition,
    {"tools": "tools", "__end__": "__end__"},
)
```

---

## Component Relationships & Integration

### With create_react_agent()

`create_react_agent()` internally:
1. Creates a `ToolNode` with provided tools.
2. Builds a `StateGraph` with "agent" and "tools" nodes.
3. Uses conditional logic equivalent to `tools_condition` to route between nodes.
4. Optionally adds "pre_model_hook" and "post_model_hook" nodes.
5. Compiles the graph with checkpointer and store.

### With Custom Graphs

For non-standard agent architectures, combine components manually:

```python
from langgraph.graph import StateGraph, START
from langgraph.prebuilt import ToolNode, tools_condition

graph = StateGraph(CustomState)
graph.add_node("reasoning", reasoning_node)
graph.add_node("tools", ToolNode(tools))
graph.add_edge(START, "reasoning")
graph.add_conditional_edges("reasoning", tools_condition)
graph.add_edge("tools", "reasoning")
graph = graph.compile()
```

---

## Configuration & Operations

### Recursion Limits & Step Limiting

The `remaining_steps` field in state tracks steps taken:

- Decremented by 1 on each superstep.
- If `remaining_steps < 2` and tool calls exist, agent returns a "need more steps" message instead of executing tools.
- If `remaining_steps < 1` and all pending tools have `return_direct=True`, agent stops.

Configure via `recursion_limit` when compiling or via graph invoke:

```python
result = agent.invoke(
    {"messages": [...]},
    config={"recursion_limit": 50}
)
```

### Streaming Tool Execution

- **v1**: All tool calls from one message execute together; state updates once.
- **v2**: Each tool call is a separate `Send` task; state updates independently, enabling finer-grained streaming and interrupts.

### Message History Management

Use `pre_model_hook` for token budget management:

```python
def trim_messages(state: State) -> State:
    messages = state["messages"]
    # Keep only last N messages or recent messages within token limit
    trimmed = messages[-20:]  # Example: last 20 messages
    return {"messages": trimmed}

agent = create_react_agent(
    model, tools,
    pre_model_hook=trim_messages
)
```

---

## Key Invariants & Failure Modes

1. **Tool Availability**: All tools must be bound to the model (or model binding skipped). If a tool is in the graph but not bound, the model will never call it.

2. **Tool Call Validation**: If a model generates a tool call with a name not in `ToolNode.tools_by_name`, a `ToolMessage` with an error is returned; execution continues.

3. **Message History Continuity**: Every `AIMessage` with `tool_calls` must have corresponding `ToolMessage` objects in the history before subsequent LLM calls. LangGraph's message reducer maintains this.

4. **Remaining Steps Enforcement**: Setting `recursion_limit` too low can cause the agent to bail out early with "need more steps" message. Increase via config if graph halts unexpectedly.

5. **Store vs Checkpointer**: `store` persists across threads/conversations; `checkpointer` persists within a thread. Both can be used simultaneously.

---

## Extension Points

### Custom Error Handling

```python
def handle_value_errors(e: ValueError) -> str:
    return f"Invalid input: {str(e)}"

tool_node = ToolNode(
    [my_tool],
    handle_tool_errors=handle_value_errors
)
```

### Tool Call Wrapping & Middleware

```python
def my_wrapper(request, execute):
    print(f"About to execute {request.tool_call['name']}")
    result = execute(request)
    print(f"Tool execution complete")
    return result

tool_node = ToolNode(
    [my_tool],
    wrap_tool_call=my_wrapper
)
```

### Dynamic Model Selection

```python
def select_model(state: State, runtime) -> BaseChatModel:
    complexity = estimate_complexity(state["messages"])
    if complexity > 8:
        return gpt4.bind_tools(tools)
    else:
        return gpt35.bind_tools(tools)

agent = create_react_agent(
    model=select_model,
    tools=tools,
)
```

### Custom State Schemas

```python
class CustomAgentState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    remaining_steps: int
    user_context: dict  # Additional field
    conversation_id: str

agent = create_react_agent(
    model, tools,
    state_schema=CustomAgentState
)
```

---

## Deprecation Notes

The following are deprecated and moved to `langchain.agents`:

- `AgentState`, `AgentStatePydantic`
- `create_react_agent()` function itself (moved to `langchain.agents.create_agent`)
- `ValidationNode`
- `HumanInterrupt`, `HumanInterruptConfig`, `ActionRequest`, `HumanResponse` (moved to `langchain.agents.interrupt`)

Import from `langchain.agents` for new code. The `langgraph.prebuilt` versions remain for backward compatibility but will be removed in v2.0.

---

## Example: Complete Agent

```python
from langchain_anthropic import ChatAnthropic
from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent

@tool
def weather(location: str) -> str:
    """Get weather for a location."""
    return "Sunny and 72°F"

@tool
def search(query: str) -> str:
    """Search the web."""
    return f"Found results for {query}"

model = ChatAnthropic(model="claude-3-5-sonnet-latest")
tools = [weather, search]

agent = create_react_agent(
    model,
    tools,
    prompt="You are a helpful assistant. Use tools when needed.",
)

# Invoke
result = agent.invoke({
    "messages": [{"role": "user", "content": "What's the weather in SF?"}]
})
print(result["messages"][-1].content)

# Stream
for chunk in agent.stream(
    {"messages": [{"role": "user", "content": "What's trending today?"}]},
    stream_mode="updates"
):
    print(chunk)
```

---

## API Reference

| Component | Purpose | Key Methods |
|-----------|---------|------------|
| `create_react_agent()` | Build a complete agent graph | `.invoke()`, `.stream()`, `.ainvoke()` |
| `ToolNode` | Execute tools in graph | `.invoke()`, `.stream()` |
| `tools_condition()` | Route on tool calls | Returns `"tools"` or `"__end__"` |
| `InjectedState` | Annotate state dependencies | Used in tool signatures |
| `InjectedStore` | Annotate store dependencies | Used in tool signatures |
| `ToolRuntime` | Runtime context for tools | Injected automatically |

---

## See Also

- [Core Concepts](../architecture/core-concepts.md): StateGraph, nodes, edges, channels
- [Graph Execution Model](../architecture/graph-execution-model.md): Supersteps, concurrency, scheduling
- [State and Channels](../architecture/state-and-channels.md): State schemas, reducers, message lists
- [Command and Send](../concepts/command-and-send.md): Control flow primitives for advanced routing
