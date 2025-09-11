# Use Pydantic default values with reducer functions

This guide shows how to use Pydantic `Field(default=...)` and `Field(default_factory=...)` with `Annotated` reducer functions in LangGraph state schemas.

## Overview

When using Pydantic models as state schemas in LangGraph, you can now specify default values that work seamlessly with reducer functions like `operator.add`. This allows your state fields to start with meaningful default values instead of empty containers.

## Basic usage

### With default factories

Use `Field(default_factory=...)` to provide a callable that generates default values:

```python
import operator
from typing import Annotated
from pydantic import BaseModel, Field
from langgraph.graph import StateGraph, START, END

class State(BaseModel):
    messages: Annotated[list[str], operator.add] = Field(
        default_factory=lambda: ["Welcome!"]
    )
    counts: Annotated[list[int], operator.add] = Field(
        default_factory=list  # Empty list
    )

def my_node(state: State) -> dict:
    return {
        "messages": ["Hello, user!"],
        "counts": [1, 2, 3]
    }

graph = StateGraph(State)
graph.add_node("process", my_node)
graph.add_edge(START, "process")
graph.add_edge("process", END)

result = graph.compile().invoke({})
print(result["messages"])  # ["Welcome!", "Hello, user!"]
print(result["counts"])    # [1, 2, 3]
```

### With static defaults

Use `Field(default=...)` for static default values:

```python
class State(BaseModel):
    total: Annotated[int, operator.add] = Field(default=100)
    multiplier: Annotated[float, operator.mul] = Field(default=1.0)

def calculate(state: State) -> dict:
    return {
        "total": 50,      # 100 + 50 = 150
        "multiplier": 2.0  # 1.0 * 2.0 = 2.0
    }
```

## Custom reducer functions

Default values work with any reducer function:

```python
def merge_dicts(current: dict, update: dict) -> dict:
    """Custom reducer that merges dictionaries."""
    result = current.copy()
    result.update(update)
    return result

class State(BaseModel):
    metadata: Annotated[dict, merge_dicts] = Field(
        default_factory=lambda: {"version": "1.0", "created": "system"}
    )

def add_metadata(state: State) -> dict:
    return {
        "metadata": {"author": "user", "timestamp": "2024-01-01"}
    }

# Result: {"version": "1.0", "created": "system", "author": "user", "timestamp": "2024-01-01"}
```

## Mixed state schemas

You can mix reducer fields with defaults and regular fields:

```python
class State(BaseModel):
    # Reducer fields with defaults
    logs: Annotated[list[str], operator.add] = Field(
        default_factory=lambda: ["System started"]
    )
    score: Annotated[int, operator.add] = Field(default=0)
    
    # Regular fields with defaults (no reducer)
    name: str = Field(default="Unnamed")
    config: dict = Field(default_factory=dict)
    
    # Required field (no default)
    user_id: str

def process_request(state: State) -> dict:
    return {
        "logs": ["Processing request"],
        "score": 10,
        "name": f"User_{state.user_id}",
        "config": {"theme": "dark"}
    }

# Must provide user_id, others get defaults
result = graph.compile().invoke({"user_id": "123"})
```

## Behavior with initial values

When you provide initial values to `invoke()`, they are combined with defaults using the reducer function:

```python
class State(BaseModel):
    items: Annotated[list[str], operator.add] = Field(
        default_factory=lambda: ["default"]
    )

# No initial values - uses defaults
result1 = graph.compile().invoke({})
# items: ["default", "new_item"]

# With initial values - combines with defaults
result2 = graph.compile().invoke({"items": ["initial"]})
# items: ["default", "initial", "new_item"]
```

This behavior ensures that defaults work consistently whether values come from initial input or node updates.

## Backward compatibility

This feature is fully backward compatible:

- Existing code without defaults continues to work unchanged
- Non-Pydantic state schemas (TypedDict, dataclasses) are unaffected  
- Regular fields without reducer functions work as before

## Error handling

If you define defaults incorrectly, you'll get clear error messages:

```python
# ❌ This will fail - default_factory should return the expected type
class BadState(BaseModel):
    items: Annotated[list[str], operator.add] = Field(
        default_factory=lambda: "not a list"  # Wrong type
    )

# ✅ Correct usage
class GoodState(BaseModel):
    items: Annotated[list[str], operator.add] = Field(
        default_factory=lambda: []  # Returns list
    )
```

## Best practices

1. **Use `default_factory` for mutable defaults** (lists, dicts) to avoid shared mutable state
2. **Use `default` for immutable values** (strings, numbers, tuples)
3. **Keep default factories simple** - complex logic should be in nodes
4. **Document your defaults** - use Field descriptions to explain the purpose

```python
class State(BaseModel):
    messages: Annotated[list[str], operator.add] = Field(
        default_factory=list,
        description="Chat messages, starts empty"
    )
    system_info: Annotated[dict, merge_dicts] = Field(
        default_factory=lambda: {"version": "1.0"},
        description="System metadata, includes version info"
    )
```
