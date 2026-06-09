# Human Approval Workflow Component

This component provides a production-ready Human-in-the-Loop (HITL) pattern for LangGraph.

## Architecture

The `ApprovalNode` is built on top of the LangGraph `interrupt` primitive. It simplifies the common requirement of pausing a graph for human intervention, specifically for approval, rejection, or modification of data.

### Key Components

- **`ApprovalNode`**: A reusable `Runnable` that can be added to any `StateGraph`. It raises a `GraphInterrupt` with a structured `ApprovalRequested` event.
- **Event Models**: Pydantic models for `ApprovalRequested`, `ApprovalGranted`, `ApprovalRejected`, and `ApprovalModified`.
- **Pregel Enhancements**: New `pause()` and `resume()` methods on the compiled graph for easier management of long-running threads.

## API Documentation

### `ApprovalNode`

```python
class ApprovalNode(RunnableCallable):
    def __init__(
        self,
        prompt: str,
        *,
        state_key: str = "approval_result",
        timeout: int | None = None,
        name: str | None = None,
        tags: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
    )
```

- `prompt`: The message displayed to the human.
- `state_key`: The key in the graph state where the outcome will be stored.
- `timeout`: Optional hint for the client about the approval deadline.

### `Pregel` Methods

- `graph.pause(config=None)`: Requests a cooperative drain. If `config` is provided with a `thread_id`, only that thread is paused.
- `graph.resume(config, value=None)`: Convenience method for `graph.invoke(Command(resume=value), config=config)`.

## Usage Guide

### Basic Approval

```python
from langgraph.prebuilt import ApprovalNode
from langgraph.graph import StateGraph, START

builder = StateGraph(State)
builder.add_node("approval", ApprovalNode(prompt="Approve payment?"))
builder.add_edge(START, "approval")
# ... add routing based on state["approval_result"]
```

### Resuming from Client

```python
from langgraph.types import Command

# To approve
graph.resume(config, value={"action": "approve", "data": {"optional": "data"}})

# To reject
graph.resume(config, value={"action": "reject", "reason": "Too expensive"})

# To modify
graph.resume(config, value={"action": "modify", "data": updated_data})
```

## Migration Notes

This is a new feature and does not introduce breaking changes. It replaces the deprecated `HumanInterrupt` patterns in `libs/prebuilt` with a modern, `Command`-based implementation.
