---
type: Advanced Control Flow
title: Command and Send
description: Primitives for dynamic routing, error recovery, and human-in-the-loop workflows—Send for fan-out task dispatch, Command for multi-directional control including resume and graph navigation.
tags: [command, send, dynamic-routing, fan-out, error-handling, resumption, task-dispatch, control-flow]
verified:
  - by: openwiki/0.5.0
    at: 2026-09-03T15:19:56.419Z
sources:
  - id: openwiki-source-38dc4e3fe1af9d8f3d241cc6
    resource: repo://libs/langgraph/langgraph/errors.py
  - id: openwiki-source-36058e684ab11833a28af83b
    resource: repo://libs/langgraph/langgraph/graph/state.py
  - id: openwiki-source-d30667fe1721764c2d67aebd
    resource: repo://libs/langgraph/langgraph/pregel/_algo.py
  - id: openwiki-source-cdb6a51f182f3b12d28a965a
    resource: repo://libs/langgraph/langgraph/pregel/_io.py
  - id: openwiki-source-73a7ae761087eaf044a3a24c
    resource: repo://libs/langgraph/langgraph/pregel/_runner.py
  - id: openwiki-source-32221bce0f3eead36a7c7e36
    resource: repo://libs/langgraph/langgraph/types.py
  - id: openwiki-source-cf9645f3c211482255b6ab55
    resource: repo://libs/langgraph/tests/test_parent_command.py
  - id: openwiki-source-359b1259bb077515edbf1b05
    resource: repo://libs/langgraph/tests/test_pregel.py
  - id: openwiki-source-6feaeb3da2a8a44b49e25d87
    resource: repo://libs/langgraph/tests/test_retry.py
generated: { by: "openwiki/0.5.0", at: "2026-09-03T15:19:56.419Z" }
---

## Overview

`Command` and `Send` are LangGraph's advanced control flow primitives that enable dynamic routing, fan-out execution, error recovery, and human-in-the-loop workflows. Unlike static edges, they allow nodes to make runtime decisions about where execution should flow and how to recover from failures.

- **Send** (repo://libs/langgraph/langgraph/types.py#L704-L793): Explicitly enqueues a task for a specific node with custom state in the same superstep. Used for map-reduce patterns and conditional fan-out.
- **Command** (repo://libs/langgraph/langgraph/types.py#L798-L849): A unified control directive that can simultaneously update state, navigate to nodes, resume interrupts, and route to parent graphs. Used for error recovery, resumption, and complex control flow.

---

## Send: Dynamic Task Dispatch

### Overview and Signature

`Send` explicitly enqueues a task for a specific node with a custom input state, enabling **dynamic fan-out** in the same superstep.

```python
Send(node="process_item", arg={"item": item}, timeout=None)
```

**Attributes:**
- **node** (str): Target node name.
- **arg** (Any): State or message to send to the target node. Can be a dict, scalar, or any value the node expects.
- **timeout** (TimeoutPolicy | None): Optional timeout for this specific task. If omitted, uses the node's default timeout.

### Use Cases: Map-Reduce and Fan-Out

The classic use case is **map-reduce**: a router conditionally sends multiple tasks to the same processing node with different inputs, then a reducing node aggregates results.

```python
from typing import Annotated, TypedDict
import operator
from langgraph.graph import StateGraph, START, END
from langgraph.types import Send

class OverallState(TypedDict):
    locations: list[str]
    results: Annotated[list[str], operator.add]

def get_weather(state: OverallState) -> OverallState:
    location = state["location"]
    weather = "sunny" if len(location) > 2 else "cloudy"
    return {"results": [f"It's {weather} in {location}"]}

def continue_to_weather(state: OverallState) -> list[Send]:
    """Fan out: create one task per location."""
    return [
        Send("get_weather", {"location": location})
        for location in state["locations"]
    ]

workflow = StateGraph(OverallState)
workflow.add_node("get_weather", get_weather)
workflow.set_conditional_entry_point(continue_to_weather)
workflow.add_edge("get_weather", END)

app = workflow.compile()
result = app.invoke({"locations": ["sf", "nyc"], "results": []})
# result = {"locations": ["sf", "nyc"], "results": ["It's cloudy in sf", "It's sunny in nyc"]}
```

**How it works:**
1. Entry point `continue_to_weather` returns a list of `Send` objects.
2. Each `Send` specifies the target node ("get_weather") and a custom state dict containing only the location field.
3. All tasks execute in the same superstep.
4. Results are aggregated into the "results" channel via the `operator.add` reducer.

### Send in Conditional Edges

A conditional edge function (repo://libs/langgraph/langgraph/graph/state.py#L700+) can return:
- A node name (string): Route to a single next node.
- A list of `Send` objects: Dispatch multiple tasks.
- A mixed list of strings and `Send` objects: Dispatch some tasks dynamically, route others statically.

```python
def router(state: State) -> list[Send | str]:
    sends = [Send("process", {"item": item}) for item in state["items"]]
    if should_aggregate:
        sends.append("aggregate")  # Static route to "aggregate" node
    return sends
```

### Send Semantics: PUSH Tasks

When a node returns `Send` objects (directly or via conditional edges), they are added to the `TASKS` channel and become **PUSH tasks** in the next superstep. The task scheduler (repo://libs/langgraph/langgraph/pregel/_algo.py#L442-L466) processes them immediately, enabling:

- **Explicit routing**: You decide which nodes run, not implicit edge traversal.
- **Parallel fan-out**: Multiple tasks to the same node run in parallel (no serialization).
- **Custom state per task**: Each `Send` encapsulates its own input state.

---

## Command: Unified Control Directives

### Overview and Structure

`Command` is a dataclass that wraps multiple control directives into a single return value, enabling simultaneous state updates, navigation, interrupt resumption, and graph-scoped control.

```python
from langgraph.types import Command

cmd = Command(
    update={"state_key": new_value},          # Update to apply
    goto="next_node",                         # Node to navigate to
    resume={"interrupt_id": resume_value},    # Resume an interrupt
    graph=None                                # "None" for current, Command.PARENT for parent
)
```

**Fields:**
- **update** (dict | Any): State changes to merge, like a regular node return.
- **goto** (str | Send | list[str | Send]): Override normal edge routing. Can be:
  - A node name: `goto="retry_node"`
  - A `Send` object: `goto=Send("task_node", {...})`
  - A list of names or `Send` objects for multiple routes.
- **resume** (dict[str, Any] | Any): Resume pending interrupts with values (see Interrupts below).
- **graph** (str | None): Route the command to a specific graph scope:
  - `None`: Apply to the current graph (default).
  - `Command.PARENT`: Route to the closest parent graph (in nested subgraph scenarios).

### Example: Error Recovery with Command.goto

Error handlers often use `Command` to decide whether to retry, skip, or fail.

```python
from langgraph.graph import StateGraph, START, END
from langgraph.errors import NodeError
from langgraph.types import Command

class State(TypedDict):
    foo: str

def always_failing_node(state: State) -> State:
    raise ValueError("Always fails")

def err_handler_node(state: State, error: NodeError) -> Command:
    """Receives the state and exception; returns a Command to recover."""
    if isinstance(error.error, ValueError):
        # Retry by updating state and jumping back
        return Command(update={"foo": "recovered"}, goto="always_failing")
    else:
        # Different error type: give up and move on
        return Command(update={"foo": "unrecoverable"}, goto="cleanup")

def cleanup(state: State) -> State:
    return state

graph = (
    StateGraph(State)
    .add_node("always_failing", always_failing_node, error_handler=err_handler_node)
    .add_node("cleanup", cleanup)
    .add_edge(START, "always_failing")
    .add_edge("cleanup", END)
    .compile()
)

result = graph.invoke({"foo": ""})  # Calls always_failing, error_handler, then cleanup
```

### Command in Error Handlers

When a node raises an exception, if it has an `error_handler` (repo://libs/langgraph/langgraph/pregel/_runner.py#L171-L174), the handler node receives:
- **state**: The state at the time of failure.
- **error**: A `NodeError` (repo://libs/langgraph/langgraph/errors.py) containing the exception and node name.

The handler can:
1. **Return a dict**: Normal state update (graph continues via normal edges).
2. **Return a Command**: Update state, navigate, or resume (full control flow).
3. **Raise an exception**: Propagate the failure and fail the graph.

```python
def error_handler(state: State, error: NodeError) -> Command | State:
    if error.node == "critical_node" and isinstance(error.error, TimeoutError):
        # Retry: go back to the same node with updated state
        return Command(update={"retry_count": state.get("retry_count", 0) + 1}, goto="critical_node")
    else:
        # Log and continue to next node
        return Command(update={"error_logged": True}, goto="post_error")
```

---

## Interrupt and Resume: Human-in-the-Loop

### The interrupt() Function

The `interrupt()` function (repo://libs/langgraph/langgraph/types.py#L851+) pauses graph execution from within a node, surfacing a value to the client. This enables workflows that require human approval, input, or verification.

```python
from langgraph.types import interrupt

def approval_node(state: State):
    # Pause execution and ask for human input
    approved = interrupt("Do you approve this action?")
    
    # If resumed, continue with the approved value
    state["approved"] = approved
    return state
```

**Mechanics:**
1. When `interrupt(value)` is called, it raises a `GraphInterrupt` exception internally (caught by the framework).
2. The graph pauses and the checkpoint is saved.
3. The value is communicated to the client via the exception.
4. The client calls `invoke(..., Command(resume={...}))` or `astream_events(..., command=Command(...))` to resume.

### Resuming with Command.resume

To resume a paused graph, use `Command(resume=...)`:

```python
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import interrupt, Command

class State(TypedDict):
    human_value: str | None

def node_with_interrupt(state: State) -> State:
    answer = interrupt("What is your age?")
    return {"human_value": answer}

graph = StateGraph(State).add_node("ask", node_with_interrupt).compile(checkpointer=InMemorySaver())

# First invocation: pauses at interrupt
try:
    result = graph.invoke({"human_value": None})
except GraphInterrupt as e:
    print(f"Interrupted: {e.value}")  # "What is your age?"

# Resume with Command
from langgraph.types import Command
result = graph.invoke(
    Command(resume="42"),  # Resume value
    config={"thread_id": thread_id}  # Same thread to get checkpoint
)
print(result["human_value"])  # "42"
```

### Multiple Interrupts

If a node contains multiple `interrupt()` calls, resume values are matched by order:

```python
def multi_interrupt_node(state: State) -> State:
    name = interrupt("What is your name?")      # First interrupt
    age = interrupt("What is your age?")         # Second interrupt
    return {"name": name, "age": age}
```

Resume by passing a dict mapping interrupt positions (or custom IDs):

```python
result = graph.invoke(
    Command(resume={"0": "Alice", "1": "30"}),
    config={"thread_id": thread_id}
)
```

---

## Graph Scoping: ParentCommand

### Command.PARENT and Nested Subgraphs

In nested graph scenarios (e.g., a parent graph that invokes a subgraph via `invoke()`), a subgraph node can communicate with the parent using `Command(graph=Command.PARENT, ...)`.

```python
class ChildState(TypedDict):
    value: int

class ParentState(TypedDict):
    parent_value: int

def child_node(state: ChildState) -> Command:
    if state["value"] > 10:
        # Break out of child graph and jump to parent's "finish" node
        return Command(graph=Command.PARENT, goto="finish")
    return state

child_builder = StateGraph(ChildState)
child_builder.add_node("child", child_node)
child_graph = child_builder.compile()

def parent_invoke_child(state: ParentState) -> ParentState:
    # Invoke child graph from within parent
    child_result = child_graph.invoke({"value": 5})
    return state

parent_builder = StateGraph(ParentState)
parent_builder.add_node("invoke_child", parent_invoke_child)
parent_builder.add_node("finish", lambda s: s)
parent_graph = parent_builder.compile()
```

**How it works:**
1. The child node returns `Command(graph=Command.PARENT, goto="finish")`.
2. The child graph recognizes the parent-scoped command and communicates it back.
3. When the child graph exits, the parent graph receives the command.
4. The parent graph routes to the "finish" node instead of continuing normally.

---

## Combining Send and Command

### Send with Command.goto

You can use `Send` within a `Command.goto` to dynamically dispatch with full control:

```python
def router(state: State) -> Command:
    if should_process:
        # Dispatch to "worker" with custom input
        return Command(
            update={"count": state["count"] + 1},
            goto=Send("worker", {"input": state["input"]})
        )
    else:
        return Command(update={"done": True}, goto="finish")
```

### Multiple Routes with Command.goto

`Command.goto` can target multiple nodes in one call:

```python
def process_node(state: State) -> Command:
    return Command(
        update={"processed": True},
        goto=[
            Send("analyze", state),
            Send("log", state),
            "finalize"  # Also route to finalize node
        ]
    )
```

---

## Implementation: How Command is Processed

### map_command Transformation

Commands are transformed into low-level writes via `map_command()` (repo://libs/langgraph/langgraph/pregel/_io.py#L56-L78). Each part of the command becomes a separate write:

1. **goto**: Converted to writes on the `TASKS` channel (for `Send`) or special branch channels (for node names).
2. **resume**: Written to the `RESUME` channel.
3. **update**: Written to the relevant state channels.

This unified transformation ensures all control directives are processed atomically in the same superstep.

---

## Key Semantics and Guarantees

### Atomicity

Both `Send` and `Command` preserve superstep atomicity:
- All writes from a `Command` are applied together.
- All `Send` tasks are scheduled and execute in parallel in the next superstep.
- State updates are never interleaved with other node executions.

### Determinism

Given the same input, checkpoint, and routing decisions, execution is deterministic. `Send` routing and `Command` navigation are explicit, so there is no hidden nondeterminism.

### Superstep Advancement

- **Send** enqueues tasks as PUSH operations, advancing the execution frontier in the same superstep cycle.
- **Command.goto** overrides normal edge traversal, letting nodes dictate the next execution step.
- Both respect channel versioning and trigger mechanisms for efficient execution.

### Error Recovery

Error handlers can return `Command` to:
- **Retry** by jumping back to the original node with updated state.
- **Skip** by jumping to a recovery node.
- **Fail** by re-raising or returning an error marker.

This enables sophisticated recovery patterns without explicit retry loops in node code.

---

## Practical Patterns

### Pattern: Conditional Retry in Error Handler

```python
def robust_node(state: State) -> State:
    result = call_external_api()
    return {"result": result}

def error_handler_with_retry(state: State, error: NodeError) -> Command:
    retry_count = state.get("retry_count", 0)
    if retry_count < 3 and isinstance(error.error, TimeoutError):
        return Command(
            update={"retry_count": retry_count + 1},
            goto="robust_node"  # Retry
        )
    else:
        return Command(
            update={"error": str(error.error)},
            goto="error_recovery"  # Give up, move to recovery
        )

graph.add_node("robust", robust_node, error_handler=error_handler_with_retry)
```

### Pattern: Map-Reduce Aggregation

```python
def fan_out(state: State) -> list[Send]:
    """Send each task to worker."""
    return [Send("worker", {"task_id": i, "data": task}) for i, task in enumerate(state["tasks"])]

def worker(state: State) -> State:
    """Process a single task."""
    result = process(state["data"])
    return {"results": [{"task_id": state["task_id"], "result": result}]}

def aggregate(state: State) -> State:
    """Merge results from all workers."""
    aggregated = {r["task_id"]: r["result"] for r in state["results"]}
    return {"aggregated": aggregated}

graph = StateGraph(State)
graph.add_node("fan_out", fan_out)
graph.add_node("worker", worker)
graph.add_node("aggregate", aggregate)
graph.add_edge(START, "fan_out")
graph.add_conditional_edges("fan_out", lambda s: [Send("worker", ...) ...])
graph.add_edge("worker", "aggregate")
```

### Pattern: Human-in-the-Loop Approval

```python
def propose_action(state: State) -> State:
    action = generate_action_proposal()
    return {"proposed_action": action}

def get_approval(state: State) -> State:
    approved = interrupt(f"Approve: {state['proposed_action']}?")
    return {"approved": approved}

def execute_if_approved(state: State) -> Command:
    if state["approved"]:
        return Command(update={"executed": True}, goto="apply")
    else:
        return Command(update={"executed": False}, goto="cancel")

def apply(state: State) -> State:
    apply_action(state["proposed_action"])
    return {"status": "applied"}

def cancel(state: State) -> State:
    return {"status": "cancelled"}

graph = StateGraph(State)
# Add nodes and edges...
```

---

## Extension and Advanced Usage

### Custom Timeout per Send

Each `Send` can override the target node's timeout:

```python
def router(state: State) -> list[Send]:
    sends = []
    for task in state["tasks"]:
        if task["priority"] == "high":
            sends.append(Send("process", task, timeout=5.0))  # Short timeout
        else:
            sends.append(Send("process", task, timeout=30.0))  # Longer timeout
    return sends
```

### Command with ParentCommand in Subgraph Chains

For deeply nested graphs, `Command` can target intermediate ancestors or siblings (though current LangGraph supports `PARENT` explicitly; more general ancestor targeting may require custom routing).

---

## See Also

- [Core Concepts](../architecture/core-concepts.md): Nodes, edges, and StateGraph API.
- [Graph Execution Model](../architecture/graph-execution-model.md): Supersteps, task scheduling, and state advancement.
- [Error Handling and Recovery](../workflows/error-handling-and-recovery.md): Detailed error handling patterns.
- Source files:
  - `langgraph/types.py`: `Send` and `Command` definitions.
  - `langgraph/pregel/_io.py`: `map_command` transformation.
  - `langgraph/errors.py`: `NodeError` and exception types.
  - `langgraph/pregel/_runner.py`: Error handler invocation and task routing.
  - `langgraph/pregel/_algo.py`: PUSH task scheduling and `Send` processing.
