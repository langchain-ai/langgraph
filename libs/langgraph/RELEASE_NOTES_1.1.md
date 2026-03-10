# LangGraph 1.1.0 Release Notes

## Type-Safe Streaming & Invoke

LangGraph 1.1 introduces `version="v2"` — a new opt-in streaming format that brings full type safety to `stream()`, `astream()`, `invoke()`, and `ainvoke()`.

### What's changing

**v1 (default, unchanged):** `stream()` yields bare tuples like `(stream_mode, data)` or just `data`. `invoke()` returns a plain `dict`. Interrupts are mixed into the output dict under `"__interrupt__"`.

**v2 (opt-in):** `stream()` yields strongly-typed `StreamPart` dicts with `type`, `ns`, `data`, and (for values) `interrupts` fields. `invoke()` returns a `GraphOutput` object with `.value` and `.interrupts` attributes. When your state schema is a Pydantic model or dataclass, outputs are automatically coerced to the correct type.

### `invoke()` / `ainvoke()` with `version="v2"`

```python
from langgraph.types import GraphOutput

result = graph.invoke({"input": "hello"}, version="v2")

# result is a GraphOutput, not a dict
assert isinstance(result, GraphOutput)
result.value       # your output — dict, Pydantic model, or dataclass
result.interrupts  # tuple[Interrupt, ...], empty if none occurred
```

With a non-`"values"` stream mode, `invoke(..., stream_mode="updates", version="v2")` returns `list[StreamPart]` instead of `list[tuple]`.

### `stream()` / `astream()` with `version="v2"`

```python
for part in graph.stream({"input": "hello"}, version="v2"):
    if part["type"] == "values":
        part["data"]        # OutputT — full state
        part["interrupts"]  # tuple[Interrupt, ...]
    elif part["type"] == "updates":
        part["data"]        # dict[str, Any]
    elif part["type"] == "messages":
        part["data"]        # tuple[BaseMessage, dict]
    elif part["type"] == "custom":
        part["data"]        # Any
    elif part["type"] == "tasks":
        part["data"]        # TaskPayload | TaskResultPayload
    elif part["type"] == "debug":
        part["data"]        # DebugPayload
```

Each stream mode has its own `TypedDict` — `ValuesStreamPart`, `UpdatesStreamPart`, `MessagesStreamPart`, `CustomStreamPart`, `CheckpointStreamPart`, `TasksStreamPart`, `DebugStreamPart` — all importable from `langgraph.types`. The union type `StreamPart` is a discriminated union on `part["type"]`, enabling full type narrowing in editors and type checkers.

### Pydantic & dataclass output coercion

When your graph's state schema is a Pydantic model or dataclass, `version="v2"` automatically coerces outputs to the declared type:

```python
from pydantic import BaseModel

class MyState(BaseModel):
    answer: str
    count: int

graph = StateGraph(MyState)
# ... build graph ...
compiled = graph.compile()

result = compiled.invoke({"answer": "", "count": 0}, version="v2")
assert isinstance(result.value, MyState)  # not a dict!
```

### Backward compatibility

- **Default is still `version="v1"`** — existing code works without changes.
- To make migration easier, `GraphOutput` supports old-style best-effort access to graph values and interrupts. Dict-style access (`result["key"]`, `"key" in result`, `result["__interrupt__"]`) still works and delegates to `result.value` / `result.interrupts` under the hood. However, this is **deprecated** and emits a `LangGraphDeprecatedSinceV11` warning. It will be removed in v3.0 — migrate to `result.value` and `result.interrupts` at your convenience.

```python
result = graph.invoke({"input": "hello"}, version="v2")

# Old style — still works, but deprecated
result["input"]          # delegates to result.value["input"]
result["__interrupt__"]  # delegates to result.interrupts
"input" in result        # delegates to "input" in result.value

# New style — preferred
result.value["input"]
result.interrupts
```

## Migration Guide

1. **No action required** — `version="v1"` remains the default. All existing code continues to work.
2. **Adopt v2 incrementally** — Add `version="v2"` to individual `invoke()`/`stream()` calls to get typed outputs.
3. **Use typed imports** — Import `GraphOutput`, `StreamPart`, and individual part types from `langgraph.types` for type-safe code.
