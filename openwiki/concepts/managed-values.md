---
type: Concept
title: Managed Values
description: Special state fields auto-populated by the LangGraph runtime before node execution. Managed values provide deterministic metadata (like execution position) without version tracking.
tags: [managed-values, state, runtime-metadata, is-last-step, execution-flow, annotations]
verified:
  - by: openwiki/0.5.0
    at: 2026-09-03T15:19:56.419Z
sources:
  - id: openwiki-source-29819e9efc57e77faf85f701
    resource: repo://libs/langgraph/langgraph/_internal/_scratchpad.py
  - id: openwiki-source-36058e684ab11833a28af83b
    resource: repo://libs/langgraph/langgraph/graph/state.py
  - id: openwiki-source-6cc6fc4ab2c58b65c5cbd03e
    resource: repo://libs/langgraph/langgraph/managed/base.py
  - id: openwiki-source-659a53a8304779f26c088f9d
    resource: repo://libs/langgraph/langgraph/managed/is_last_step.py
  - id: openwiki-source-d30667fe1721764c2d67aebd
    resource: repo://libs/langgraph/langgraph/pregel/_algo.py
  - id: openwiki-source-81a72de562b937c38cd9f60c
    resource: repo://libs/langgraph/tests/test_managed_values.py
  - id: openwiki-source-8b2910f50b9e167d64d2728e
    resource: repo://libs/prebuilt/langgraph/prebuilt/chat_agent_executor.py
generated: { by: "openwiki/0.5.0", at: "2026-09-03T15:19:56.419Z" }
---

## Overview

**Managed values** are special state fields that the LangGraph runtime automatically populates before executing a node. Unlike regular state channels, managed values:

- Are **never written by nodes**—they are read-only within node execution
- Are **not versioned** and do not trigger node execution via version changes
- Are **not stored in checkpoints**—they are recomputed from execution context at runtime
- Are **deterministic**—computed from the current execution step and stop count

Managed values enable conditional logic based on execution position without requiring explicit orchestration or state pollution.

---

## Core Concept: Annotation-Based Declaration

Managed values are declared using the `Annotated` type hint with a `ManagedValue` subclass:

```python
from typing import Annotated, TypedDict
from langgraph.managed import IsLastStep, RemainingSteps

class State(TypedDict):
    messages: list
    is_final: Annotated[bool, IsLastStep]
    steps_left: Annotated[int, RemainingSteps]
```

The annotation marker (e.g., `IsLastStep`) is a `ManagedValue` subclass that provides:
- **Type information** (`bool`, `int`, etc.) via the outer `Annotated` type parameter
- **Runtime computation** via a `get(scratchpad: PregelScratchpad)` method called before each node executes

---

## Built-in Managed Values

### IsLastStep

Boolean field indicating whether the current node is the final node before graph termination.

```python
from langgraph.managed import IsLastStep
from typing import Annotated, TypedDict

class State(TypedDict):
    messages: list
    is_last: Annotated[bool, IsLastStep]

def summarize_node(state: State) -> dict:
    if state["is_last"]:
        # Final node: perform cleanup or summary
        return {"messages": state["messages"] + ["Summary: ..."]}}
    else:
        # Non-final node: continue normally
        return {}
```

**Implementation:**
- Compares current execution step (`scratchpad.step`) with the stop count (`scratchpad.stop`)
- Returns `True` when `step == stop - 1`
- Set before the node receives input, allowing conditional branching

**Use cases:**
- Final cleanup or aggregation without explicit end-node wiring
- Dynamic routing based on execution position
- Conditional flushing of buffers before graph termination

### RemainingSteps

Integer field indicating how many more steps will execute in this run (including the current step).

```python
from langgraph.managed import RemainingSteps
from typing import Annotated, TypedDict

class State(TypedDict):
    depth: int
    remaining: Annotated[int, RemainingSteps]

def adaptive_node(state: State) -> dict:
    if state["remaining"] == 1:
        # Very last step: use fast path
        return {}
    elif state["remaining"] <= 3:
        # Approaching end: allocate resources carefully
        return {}
    else:
        # Plenty of steps left: be thorough
        return {}
```

**Implementation:**
- Computes `scratchpad.stop - scratchpad.step`
- Counts down by 1 on each superstep
- Allows adaptive behavior based on remaining execution budget

---

## Runtime Mechanics

### Scratchpad Integration

Managed values are computed from the **PregelScratchpad**, a runtime-maintained object created per execution context:

```python
@dataclass
class PregelScratchpad:
    step: int           # Current execution step (0-indexed)
    stop: int           # Total step limit for this run
    # ... other runtime state
```

When a node requests a managed value, the input preparation code (`_proc_input` in `_algo.py`) calls:

```python
if chan in channels:
    val[chan] = channels[chan].get()  # Regular channel
else:
    val[chan] = managed[chan].get(scratchpad)  # Managed value
```

The `ManagedValue` subclass's `get()` method receives the scratchpad and returns a fresh computed value.

### Read-Only Within Node

Once populated and passed to a node, managed values are read-only. The node cannot write to them:

```python
def my_node(state: State) -> dict:
    # This is allowed (reading):
    if state["is_last"]:
        print("Last node")
    
    # This is NOT allowed (cannot return updates to managed fields):
    # return {"is_last": False}  # ❌ Error: managed field is read-only
    
    # Only non-managed state can be returned:
    return {"messages": [...]}  # ✓ OK
```

---

## Schema Integration

### Detection and Validation

At graph compilation, LangGraph scans the state schema for `Annotated` fields with `ManagedValue` metadata:

```python
def _is_field_managed_value(name: str, typ: type[Any]) -> ManagedValueSpec | None:
    if hasattr(typ, "__metadata__"):
        meta = typ.__metadata__
        if len(meta) >= 1:
            decoration = get_origin(meta[-1]) or meta[-1]
            if is_managed_value(decoration):
                return decoration
    
    # Handle Required/NotRequired wrappers
    if get_origin(typ) is not None:
        # Recursively check inner type
        return _is_field_managed_value(name, get_args(typ)[0])
    
    return None
```

Recognized managed fields are stored separately in `StateGraph.managed` (a dict mapping field name to `ManagedValueSpec`).

### Input/Output Schema Restrictions

Managed values are **only allowed in the state schema**, not in input or output schemas:

```python
from langgraph.graph import StateGraph

class State(TypedDict):
    messages: list
    is_final: Annotated[bool, IsLastStep]

class Input(TypedDict):
    messages: list
    # is_final: Annotated[bool, IsLastStep]  # ❌ Error: not permitted in input

class Output(TypedDict):
    messages: list
    # is_final: Annotated[bool, IsLastStep]  # ❌ Error: not permitted in output

graph = StateGraph(State, input_schema=Input, output_schema=Output)
```

**Rationale:** Managed values are runtime-computed and should not be externally specified (input) or exposed to the user (output).

### Optional Managed Values

Managed values can be marked `NotRequired` or `Required` to control whether they must be present:

```python
from typing_extensions import NotRequired, Required
from langgraph.managed import RemainingSteps

class State(TypedDict):
    count: NotRequired[Annotated[int, RemainingSteps]]  # Optional managed value
    messages: Required[list]  # Required regular field
```

The `Required`/`NotRequired` wrappers are unwrapped during schema parsing, and the inner managed value is recognized.

---

## Use Cases and Patterns

### Conditional Final Aggregation

```python
def aggregator_node(state: State) -> dict:
    """Aggregate results only on the final step."""
    if state["is_last"]:
        return {
            "final_results": combine(state["intermediate_results"])
        }
    return {}
```

### Adaptive Depth-Based Pruning

```python
def research_node(state: State) -> dict:
    """Adjust search depth based on remaining execution budget."""
    max_depth = 5 if state["remaining"] > 10 else 2
    results = search(state["query"], depth=max_depth)
    return {"results": results}
```

### Dynamic Early Termination

```python
from langgraph.types import Command

def intelligent_router(state: State) -> Command | dict:
    """Exit early if approaching step limit or condition met."""
    if state["remaining"] <= 1:
        return Command(goto=END)  # Force termination
    elif state["confidence"] > 0.95:
        return Command(goto=END)  # Terminate early
    else:
        return Command(goto="next_node")
```

### Agent Loop with Bounded Iterations

```python
def agent_loop_step(state: State) -> dict:
    """Agent iteration that respects step budget."""
    if state["remaining"] == 1:
        # Last step: return final answer regardless
        return {"answer": state["final_answer"]}
    
    # Not last step: continue deliberation
    return {"thoughts": [...]}
```

---

## Type Safety and Validation

### Type Checking

The outer `Annotated` parameter specifies the field type:

```python
class State(TypedDict):
    is_final: Annotated[bool, IsLastStep]      # bool type
    remaining: Annotated[int, RemainingSteps]  # int type
    # position: Annotated[str, IsLastStep]     # ❌ Type mismatch
```

At runtime, `IsLastStepManager.get()` returns a `bool`, and `RemainingStepsManager.get()` returns an `int`. Type mismatches between the annotation and the manager's return type are caught during schema validation.

### Preventing Writes

Since managed values are stored in a separate `managed` dict (not `channels`), attempting to return updates for them produces an error:

```python
def bad_node(state: State) -> dict:
    # This fails at execution time or returns silently
    # (depends on graph implementation)
    return {"is_last": False}
```

The graph runtime ignores updates to managed field keys, preventing accidental writes.

---

## Lifecycle and Checkpointing

### Not Versioned

Managed values do not have versions. They do not participate in:
- **Version tracking**: Nodes don't watch managed value versions to decide execution
- **Change detection**: A change in a managed value doesn't trigger dependent nodes

### Not Checkpointed

Managed values are never saved to checkpoints because they are deterministic functions of `(step, stop)`:

```python
# During checkpoint:
checkpoint = {
    "messages": [...],  # Saved
    "is_last": ???      # NOT saved—recomputed from step/stop
}

# During resume:
state["is_last"] = IsLastStepManager.get(scratchpad)  # Recomputed
```

This ensures:
- Checkpoints remain minimal and resumeable at any step
- Managed values are always consistent with current execution position
- No stale metadata persists across checkpoint boundaries

### Recomputed Before Each Execution

When the graph resumes or executes a node, managed values are freshly computed:

1. Create or restore `PregelScratchpad` with current `step` and `stop`
2. Call `_proc_input(node, ..., scratchpad=...)`
3. For each managed field in the node's input:
   ```python
   val[field] = managed[field].get(scratchpad)
   ```
4. Pass `val` to the node for execution

---

## Extension Points

### Creating Custom Managed Values

To create a new managed value, subclass `ManagedValue` and implement `get()`:

```python
from langgraph.managed.base import ManagedValue
from langgraph._internal._scratchpad import PregelScratchpad
from typing import Annotated

class StepCountManager(ManagedValue[int]):
    """Return the current step number (0-indexed)."""
    @staticmethod
    def get(scratchpad: PregelScratchpad) -> int:
        return scratchpad.step

StepCount = Annotated[int, StepCountManager]

# Use it in a state:
class State(TypedDict):
    current_step: StepCount
```

**Requirements:**
- Subclass `ManagedValue[V]` where `V` is the return type
- Implement static method `get(scratchpad: PregelScratchpad) -> V`
- Create a type alias using `Annotated[type, YourManager]` for ergonomics
- The manager must be idempotent: `get(scratchpad)` must always return the same value for the same scratchpad state

### Introspection

Access managed values in a compiled graph:

```python
graph = StateGraph(State).compile()

# Inspect managed fields:
for name, manager in graph.managed.items():
    print(f"{name}: {manager}")
```

---

## Comparison with Regular Channels

| Aspect | Regular Channel | Managed Value |
|--------|-----------------|---------------|
| **Definition** | `Annotated[Type, reducer]` or plain type | `Annotated[Type, ManagedValue]` |
| **Written by** | Nodes (via returns) | Runtime only (read-only) |
| **Versioned** | Yes—changes trigger dependent nodes | No—never versioned |
| **Checkpointed** | Yes—value stored | No—recomputed from context |
| **Use** | Application state & data flow | Execution metadata & position |
| **Examples** | Messages, count, embeddings | `IsLastStep`, `RemainingSteps` |

---

## See Also

- **State and Channels** (`/openwiki/architecture/state-and-channels.md`): Comprehensive guide to state schema, channel types, and state lifecycle
- **Graph Building** (`/openwiki/workflows/graph-building.md`): How to construct and compile graphs with managed values
- **langgraph.managed** module: Built-in managed value implementations
- **langgraph.graph.StateGraph**: Main graph class that recognizes and manages managed value fields
