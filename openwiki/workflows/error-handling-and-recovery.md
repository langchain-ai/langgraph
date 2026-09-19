---
type: Workflow Patterns & Techniques
title: Error Handling and Recovery
description: Mechanisms for handling node failures, implementing retry policies with exponential backoff, enforcing per-node timeouts, and gracefully recovering from errors using node-level error handlers.
tags: [error-handling, retry-policy, timeouts, node-error-handler, recovery, resilience, graceful-degradation]
verified:
  - by: openwiki/0.5.0
    at: 2026-09-03T15:19:56.419Z
sources:
  - id: openwiki-source-22f40b109bcc8f002f430187
    resource: repo://libs/langgraph/langgraph/_internal/_retry.py
  - id: openwiki-source-38dc4e3fe1af9d8f3d241cc6
    resource: repo://libs/langgraph/langgraph/errors.py
  - id: openwiki-source-36058e684ab11833a28af83b
    resource: repo://libs/langgraph/langgraph/graph/state.py
  - id: openwiki-source-d30667fe1721764c2d67aebd
    resource: repo://libs/langgraph/langgraph/pregel/_algo.py
  - id: openwiki-source-b60fe5dbe973a0b51e8dc248
    resource: repo://libs/langgraph/langgraph/pregel/_retry.py
  - id: openwiki-source-32221bce0f3eead36a7c7e36
    resource: repo://libs/langgraph/langgraph/types.py
generated: { by: "openwiki/0.5.0", at: "2026-09-03T15:19:56.419Z" }
---

## Overview

LangGraph provides a robust error handling and recovery framework that enables graphs to tolerate transient failures, gracefully degrade when errors occur, and maintain deterministic execution even in the face of exceptions. The framework unifies:

- **Retry policies** (`RetryPolicy`): Configurable retry strategies with exponential, linear, or constant backoff, per-node exception filters, and maximum attempt limits.
- **Timeout policies** (`TimeoutPolicy`): Per-node wall-clock (`run_timeout`) and idle-time (`idle_timeout`) enforcement with progress-based refresh.
- **Node-level error handlers** (`error_handler` parameter): Functions that intercept node failures, modify state, and route execution (e.g., to fallback nodes or recovery paths).
- **Error context** (`NodeError`): Structured failure information passed to handlers, including node name, exception, and attempt count.
- **Subgraph error propagation** (`ParentCommand`): Allows parent graphs to decide recovery strategy for child graph failures.
- **Default retry rules** (`default_retry_on`): Sensible defaults that retry transient network and runtime errors while failing fast on programming errors.

These mechanisms operate at the **node level** within LangGraph's superstep execution model, so retries and error handling occur atomically with checkpointing and state advancement.

---

## Retry Policies

### Configuration and Backoff Strategies

Retry policies are configured per-node via `add_node(..., retry_policy=...)` or set as defaults via `set_node_defaults(retry_policy=...)`. A `RetryPolicy` is a named tuple defining:

```python
from langgraph.types import RetryPolicy

policy = RetryPolicy(
    max_attempts=3,                # Total attempts, including the first
    initial_interval=0.5,          # Seconds before first retry
    backoff_factor=2.0,            # Multiplier: interval *= factor each retry
    max_interval=128.0,            # Cap on interval between retries
    jitter=True,                   # Add random jitter to smooth thundering herd
    retry_on=ValueError,           # Exception class, tuple, or callable predicate
)
```

**Backoff Calculation:**

The interval grows exponentially by default: `interval = min(max_interval, initial_interval * (backoff_factor ^ (attempt - 1)))`, where `attempt` counts failures (1-indexed after the first failure). With jitter enabled, a random factor in `[0, 1)` is multiplied into the interval, reducing synchronized retry waves.

**Exception Filtering:**

The `retry_on` parameter determines which exceptions trigger a retry:

- **Exception class**: `retry_on=ConnectionError` — retry on that class and subclasses.
- **Tuple of classes**: `retry_on=(ConnectionError, TimeoutError)` — retry if the exception is an instance of any class.
- **Callable predicate**: `retry_on=lambda exc: isinstance(exc, requests.HTTPError) and exc.response.status_code >= 500` — custom logic.

### Default Retry Behavior

If `retry_on` is omitted, LangGraph uses `default_retry_on`, which:

- **Retries**: `ConnectionError`, HTTP 5xx responses (from `httpx.HTTPStatusError`, `requests.HTTPError`), and any exception not in the explicit "fail-fast" list below.
- **Never retries** (fails immediately): `ValueError`, `TypeError`, `ArithmeticError`, `ImportError`, `LookupError`, `NameError`, `SyntaxError`, `RuntimeError`, `ReferenceError`, `StopIteration`, `StopAsyncIteration`, `OSError`.

This default strategy assumes most exceptions are transient (network, resource contention) unless they are clearly programmer errors. `NodeTimeoutError` is notably **not** an `OSError`, so it is retried by default.

### Multiple Retry Policies per Node

A node can have multiple policies in sequence:

```python
from langgraph.graph import StateGraph

graph = (
    StateGraph(State)
    .add_node(
        "my_node",
        my_node_fn,
        retry_policy=[
            RetryPolicy(max_attempts=3, retry_on=ConnectionError),
            RetryPolicy(max_attempts=2, retry_on=TimeoutError),
        ]
    )
)
```

The first matching policy is applied; if no exception matches any policy, the exception propagates (unless an error handler catches it).

### Execution Tracking and Runtime Info

During retries, `runtime.execution_info` exposes:

- `node_attempt` (int, 1-indexed): The current attempt number.
- `node_first_attempt_time` (float, Unix timestamp): The wall-clock time of the first attempt for this node, constant across all retries.

Use these to log retry context, implement adaptive backoff, or skip expensive setup logic on retries:

```python
def my_node(state: State, runtime: Runtime) -> dict:
    if runtime.execution_info.node_attempt > 1:
        print(f"Retry #{runtime.execution_info.node_attempt}")
    # ... node logic
```

---

## Timeout Policies

### Configuration

Timeout policies are set per-node via `add_node(..., timeout=...)` using `TimeoutPolicy`:

```python
from langgraph.types import TimeoutPolicy
from datetime import timedelta

policy = TimeoutPolicy(
    run_timeout=timedelta(seconds=30),    # Hard wall-clock cap
    idle_timeout=10.0,                     # Max time without progress (seconds)
    refresh_on="auto",                     # Or "heartbeat" for manual refresh
)
```

Timeouts can also be a simple number (treated as `run_timeout`) or a `timedelta`:

```python
graph.add_node("my_node", my_fn, timeout=30)  # 30-second run timeout
graph.add_node("my_node", my_fn, timeout=timedelta(minutes=2))
```

### Timeout Types

**`run_timeout`** (hard deadline):

- Measures wall-clock time from task start.
- **Never refreshed** by progress signals.
- Ideal for preventing runaway operations.

**`idle_timeout`** (progress-based):

- Measures time since the last observable progress.
- Refreshed by:
  - Progress signals when `refresh_on="auto"` (default): channel writes, task scheduling, LangChain callbacks.
  - Explicit `runtime.heartbeat()` calls when `refresh_on="heartbeat"`.
- Ideal for detecting stalled work (e.g., hung API call, blocked thread).

### Triggering and Recovery

When a timeout fires, LangGraph raises `NodeTimeoutError` with fields:

```python
error.node          # Node name
error.kind          # "run" or "idle"
error.timeout       # The configured limit (seconds)
error.elapsed       # Time elapsed when timeout fired
error.run_timeout   # Configured run_timeout (or None)
error.idle_timeout  # Configured idle_timeout (or None)
```

Because `NodeTimeoutError` does not inherit from `OSError`, the default `RetryPolicy` treats it as **retryable**. Combine with `error_handler` for custom recovery:

```python
def timeout_handler(state: State, error: NodeError) -> Command:
    if isinstance(error.error, NodeTimeoutError):
        if error.error.kind == "idle":
            return Command(update={"status": "timeout_idle"}, goto="fallback")
        elif error.error.kind == "run":
            return Command(update={"status": "timeout_run"})
    # Re-raise if not a timeout
    raise error.error

graph.add_node("my_node", my_fn, timeout=30, error_handler=timeout_handler)
```

### Cooperative Cancellation

Timeouts rely on **asyncio cancellation**. For async code (coroutines), the timeout is enforced promptly. For sync code (blocking calls, `time.sleep()`), the timeout fires only after the event loop gains control (i.e., when the blocking operation yields). Heavy CPU work may delay timeout firing; use heartbeats or shorter idle timeouts for such workloads.

---

## Node-Level Error Handlers

### Definition and Injection

An error handler is a function that receives the node's state and a `NodeError` object describing the failure. It returns a `Command` to decide recovery:

```python
from langgraph.errors import NodeError
from langgraph.types import Command

def my_error_handler(state: State, error: NodeError) -> Command:
    """Handle errors from a specific node."""
    print(f"Node {error.node} failed with: {error.error}")
    # Option 1: Update state and continue
    return Command(update={"status": f"failed: {error.error}"})
    # Option 2: Route to a fallback node
    return Command(update={"fallback": True}, goto="fallback_node")
    # Option 3: Re-raise to propagate
    raise error.error

graph.add_node("my_node", my_fn, error_handler=my_error_handler)
```

The handler is injected as a special-case dependency injection: if the handler's signature includes a parameter typed `NodeError`, LangGraph automatically passes the failure context. The state parameter works as for normal nodes.

### Execution Semantics

1. **Trigger**: A node raises an exception that is not retryable (or retries are exhausted).
2. **Schedule**: LangGraph schedules the error handler as a new task in the same superstep boundary.
3. **Receive input**: The handler receives the same state snapshot the failed node saw (not the pre-node state).
4. **Decide recovery**: The handler returns a `Command` directing the graph:
   - `update`: Modify state keys (e.g., record the error).
   - `goto`: Navigate to a fallback node, recovery node, or END.
   - `resume`: Continue from an interrupt point.
5. **Persist**: The graph checkpoint includes the error context (node name, exception string) and handler result.

### Default Error Handlers

Set a default error handler via `set_node_defaults(error_handler=...)`:

```python
def default_handler(state: State, error: NodeError) -> Command:
    return Command(update={"error": str(error.error)})

graph = StateGraph(State).set_node_defaults(error_handler=default_handler)
```

Per-node handlers override the default. Error-handler nodes themselves are **excluded** from default error handlers (handlers never catch themselves), preventing infinite recursion.

### Handler Failures

If an error handler itself raises an exception, the entire run fails. This is by design: error handlers are trusted recovery logic. To ensure robustness, test handlers thoroughly or use a wrapper:

```python
def safe_error_handler(state: State, error: NodeError) -> Command:
    try:
        # Complex recovery logic
        return Command(update={"recovered": True})
    except Exception as handler_error:
        # Fallback: skip recovery
        return Command(update={"handler_failed": str(handler_error)})
```

---

## Subgraph Error Propagation and ParentCommand

### Handling Subgraph Failures

When a subgraph node fails, the parent graph can intercept and recover using an error handler:

```python
subgraph = StateGraph(SubState).add_node("worker", worker_fn).compile()

def parent_handler(state: ParentState, error: NodeError) -> Command:
    if error.node == "subgraph":
        # Subgraph failed; decide recovery
        return Command(update={"subgraph_retried": True}, goto="fallback")
    raise error.error

graph = (
    StateGraph(ParentState)
    .add_node("subgraph", subgraph, error_handler=parent_handler)
    .add_node("fallback", fallback_fn)
)
```

The parent sees the subgraph as an opaque node; if the subgraph raises, the parent's error handler (if set) can decide the recovery strategy.

### ParentCommand for Cross-Graph Control

A subgraph can issue a command to its parent using `ParentCommand`:

```python
from langgraph.errors import ParentCommand
from langgraph.types import Command

def subgraph_node(state: SubState):
    # Ask parent to do something
    cmd = Command(graph=Command.PARENT, update={"parent_flag": True})
    raise ParentCommand(cmd)

graph = (
    StateGraph(ParentState)
    .add_node("subgraph", subgraph, error_handler=parent_handler)
)
```

The parent's error handler can catch the `ParentCommand` (via its wrapped exception) and apply the command, allowing subgraphs to coordinate recovery with their parent.

---

## Error Class Hierarchy

Key error types in `langgraph.errors`:

### Execution Failures

**`NodeError` (dataclass)**:

Passed to error handlers, contains:

```python
@dataclass(frozen=True)
class NodeError:
    node: str                # Name of the failed node
    error: BaseException     # The exception that was raised
```

**`NodeTimeoutError`**:

Raised when `run_timeout` or `idle_timeout` fires. Not a subclass of `OSError`, so retried by default. Fields:

```python
error.node              # Node name
error.kind              # "idle" or "run"
error.timeout           # The configured limit (seconds)
error.elapsed           # Elapsed time at timeout
error.run_timeout       # Configured run_timeout (or None)
error.idle_timeout      # Configured idle_timeout (or None)
```

**`NodeCancelledError`**:

Raised when user code explicitly raises `asyncio.CancelledError` (not framework-initiated cancellation). Flows through retries and error handlers normally.

### Control Flow

**`GraphDrained`**:

Raised when `RunControl.request_drain()` is called (e.g., on SIGTERM). The checkpoint is saved; the run can be resumed later. Not an error in the sense of failure—it's cooperative shutdown.

**`GraphInterrupt`** / **`NodeInterrupt` (deprecated)**:

Raised by `interrupt()` or `NodeInterrupt` to pause graph execution at a human-in-the-loop checkpoint. Suppressed by the root graph; never surfaces to the user as an error.

**`ParentCommand`**:

A special exception wrapping a `Command` that a subgraph sends to its parent graph. The parent's error handler can catch and apply it.

### Validation Errors

**`GraphRecursionError`**:

Raised when the graph exceeds `recursion_limit` (default 25). Indicates an infinite loop (e.g., a conditional edge always sends back to itself). Increase the limit with `graph.invoke(..., {"recursion_limit": 100})` if truly needed.

**`InvalidUpdateError`**:

Raised when a node returns an update that conflicts with channel definitions (e.g., multiple nodes write incompatible values to a channel without a reducer, or an update uses a non-string key).

---

## Practical Examples

### Example 1: Graceful Degradation with LLM Timeouts

Handle LLM API timeouts by using a cached or fallback response:

```python
from langgraph.types import Command, TimeoutPolicy, RetryPolicy
from langgraph.errors import NodeError

class State(TypedDict):
    query: str
    response: str

def call_llm(state: State) -> dict:
    # LLM call with potential timeout
    return {"response": expensive_llm_call(state["query"])}

def llm_timeout_handler(state: State, error: NodeError) -> Command:
    # If LLM timed out, use a default response
    if isinstance(error.error, NodeTimeoutError):
        return Command(
            update={"response": "I'm busy right now. Please try again later."},
            goto="end"
        )
    raise error.error

graph = (
    StateGraph(State)
    .add_node(
        "llm",
        call_llm,
        timeout=TimeoutPolicy(run_timeout=10.0),
        error_handler=llm_timeout_handler
    )
    .compile()
)
```

### Example 2: Tool Failures with Retry and Fallback

Retry tool calls on transient errors, fall back to a manual step on persistent failure:

```python
class State(TypedDict):
    query: str
    tool_result: str
    needs_human_review: bool

def call_tool(state: State) -> dict:
    try:
        return {"tool_result": external_api_call(state["query"])}
    except Exception as e:
        raise e  # Let retry policy decide

def tool_fallback(state: State, error: NodeError) -> Command:
    # Log the error and request human review
    logger.error(f"Tool call failed: {error.error}")
    return Command(
        update={"needs_human_review": True, "tool_result": ""},
        goto="human_review"
    )

graph = (
    StateGraph(State)
    .add_node(
        "tool",
        call_tool,
        retry_policy=RetryPolicy(
            max_attempts=3,
            retry_on=ConnectionError,  # Only retry network errors
        ),
        error_handler=tool_fallback
    )
    .add_node("human_review", lambda state: state)
    .compile()
)
```

### Example 3: Subgraph Failure Recovery in Parent

Detect a subgraph failure and apply compensating logic:

```python
class SubState(TypedDict):
    input: str
    output: str

class ParentState(TypedDict):
    items: list[str]
    results: list[str]
    failed_items: list[str]

sub = StateGraph(SubState).add_node(
    "process",
    lambda s: process_item(s["input"]) if s["input"] else (_ for _ in ()).throw(ValueError("Empty input"))
).compile()

def parent_handler(state: ParentState, error: NodeError) -> Command:
    # Subgraph failed; mark the item as failed and continue
    # (Assumes the subgraph node was given input via Send)
    return Command(
        update={
            "failed_items": [error.node],  # Simplified; real code would track which item
            "results": []
        },
        goto="continue"
    )

graph = (
    StateGraph(ParentState)
    .add_node("process_all", lambda s: [Send("sub", {"input": item}) for item in s["items"]])
    .add_node("sub", sub, error_handler=parent_handler)
    .add_node("continue", lambda s: {"results": s["results"]})
    .compile()
)
```

### Example 4: Observing Attempt Count

Use `runtime.execution_info` to optimize retry behavior:

```python
class State(TypedDict):
    query: str
    response: str

def smart_llm_call(state: State, runtime: Runtime) -> dict:
    attempt = runtime.execution_info.node_attempt
    timeout_secs = 30 - (attempt - 1) * 5  # Tighter timeout on retries
    
    # Use shorter timeout on retries to fail fast
    return {
        "response": llm_call(
            state["query"],
            timeout=max(5, timeout_secs)
        )
    }

graph = StateGraph(State).add_node(
    "llm",
    smart_llm_call,
    timeout=TimeoutPolicy(run_timeout=30.0),
    retry_policy=RetryPolicy(max_attempts=3)
).compile()
```

---

## Durability and Checkpointing Integration

Error handlers and retries interact with checkpoint durability modes:

- **`"sync"`**: After the error handler completes (or error propagates), the checkpoint is persisted before the next superstep.
- **`"async"`**: Checkpoint writes occur in the background while the next superstep begins.
- **`"exit"`**: Checkpoint is written only when the entire graph completes or fails.

In all modes, the checkpoint includes:

- The failed node's name and exception (as a string).
- The state snapshot at the time of failure.
- Any updates applied by the error handler.

This ensures that resuming from a checkpoint after failure preserves full context for investigation and recovery.

---

## Best Practices

1. **Use `retry_policy` for transient failures**: ConnectionError, timeouts, rate limits. Avoid retrying logic errors (ValueError, TypeError).
2. **Set `error_handler` for expected failures**: LLM API unavailability, external service degradation, validation failures.
3. **Combine timeout and error handling**: Use `idle_timeout` to detect stalled work and error handlers to escalate gracefully.
4. **Test handler robustness**: Error handlers are critical recovery paths; ensure they never fail silently.
5. **Log context in handlers**: Capture `error.node`, `error.error` for observability.
6. **Use `set_node_defaults` for consistent policy**: Set retry/timeout/handler defaults at the graph level, override per-node as needed.
7. **Monitor `node_attempt` in Runtime**: Adjust retry behavior or logging based on the current attempt number.
