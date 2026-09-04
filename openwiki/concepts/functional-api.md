---
type: API & Patterns
title: Functional API
description: The @entrypoint and @task decorators for function-based graph composition, supporting retry, cache, and timeout policies with streamlined task invocation.
tags: [functional-api, decorators, entrypoint, task, composition, async, retry, cache, timeout]
verified:
  - by: openwiki/0.5.0
    at: 2026-09-03T15:19:56.419Z
sources:
  - id: openwiki-source-1961d6a5fa3ab7c58fae0a68
    resource: repo://libs/langgraph/langgraph/func/__init__.py
  - id: openwiki-source-fcbca03f80a31d0e696f2a7a
    resource: repo://libs/langgraph/langgraph/pregel/_call.py
  - id: openwiki-source-32221bce0f3eead36a7c7e36
    resource: repo://libs/langgraph/langgraph/types.py
generated: { by: "openwiki/0.5.0", at: "2026-09-03T15:19:56.419Z" }
---

## Overview

LangGraph's functional API provides **`@entrypoint` and `@task` decorators** as an alternative to the procedural `StateGraph` builder. Instead of manually constructing nodes and edges, you write ordinary functions that automatically compile into executable graphs. This approach is intuitive for simple workflows and enables natural composition through task invocation, while preserving the full power of LangGraph's execution model: retry policies, caching, timeouts, checkpointing, and state persistence.

**Key Design Principles:**

- **Functions as nodes**: Decorate regular sync or async functions to turn them into graph nodes.
- **Implicit state context**: Tasks operate on graph state without explicit passing.
- **Futures for parallelism**: Task calls return `SyncAsyncFuture` objects, enabling easy concurrent execution.
- **Composition**: Entrypoints call tasks, tasks call other tasks, enabling nested execution.
- **Policy-driven resilience**: Attach retry, cache, and timeout policies declaratively at decoration time.

---

## @entrypoint Decorator

The `@entrypoint` decorator marks a function as the **entry point and outermost boundary of a LangGraph workflow**. It wraps the function into a `Pregel` graph that can be invoked, streamed, and checkpointed like any compiled LangGraph graph.

### Signature and Parameters

```python
@entrypoint(
    checkpointer: BaseCheckpointSaver | None = None,
    store: BaseStore | None = None,
    cache: BaseCache | None = None,
    context_schema: type[ContextT] | None = None,
    cache_policy: CachePolicy | None = None,
    retry_policy: RetryPolicy | Sequence[RetryPolicy] | None = None,
    timeout: float | timedelta | TimeoutPolicy | None = None,
)
def my_workflow(input_value: InputType) -> OutputType:
    ...
```

**Parameters:**

- **`checkpointer`**: Optional checkpoint saver for persistence across runs. Enables resuming workflows from the same `thread_id`.
- **`store`**: Optional key-value store for long-lived, multi-run data access.
- **`cache`**: Cache instance for caching task and workflow results. Tasks within the entrypoint can opt into caching via `@task(cache_policy=...)`.
- **`context_schema`**: Optional TypedDict or dataclass defining the schema for run-scoped context passed via `config["context"]`.
- **`cache_policy`**: Cache policy for the entrypoint's own result. Rarely used; task-level caching is more common.
- **`retry_policy`**: Retry policy (or sequence of policies) applied to the entrypoint on failure.
- **`timeout`**: Timeout enforcement. A float or `timedelta` becomes a `run_timeout` (hard wall-clock cap). Use `TimeoutPolicy` for fine-grained control with both `run_timeout` and `idle_timeout`.

### Function Signature Constraints

The decorated function **must accept a single input parameter**. To pass multiple values, use a dictionary or dataclass:

```python
@entrypoint()
def single_input(state: dict[str, Any]) -> dict[str, Any]:
    # ✓ Single parameter
    return state

# For multiple logical inputs, pass a dict:
@entrypoint()
def multi_input(data: dict) -> str:
    topic = data["topic"]
    style = data["style"]
    return f"Essay on {topic} in {style} style"
```

### Injectable Parameters

The function can request **optional injected parameters** that are filled at runtime:

| Parameter | Type | Description |
|-----------|------|-------------|
| **`config`** | `RunnableConfig` | Configuration object holding `thread_id`, `configurable`, and other runtime settings. |
| **`previous`** | Any | The return value from the previous invocation on the same `thread_id` (only when checkpointer is provided). |
| **`runtime`** | `Runtime` | Access to the current run context, including `context`, `store`, and `heartbeat()` for timeout progress signaling. |

**Example:**

```python
@entrypoint(checkpointer=InMemorySaver())
def stateful_workflow(
    input_value: str,
    *,
    previous: str | None = None,
    runtime: Runtime | None = None,
) -> str:
    """Workflow with optional injected parameters."""
    if previous is None:
        previous = "start"
    result = f"{previous} -> {input_value}"
    if runtime:
        runtime.heartbeat()  # Reset idle_timeout
    return result
```

### Return Types and `entrypoint.final`

By default, the entrypoint's return value is both the output to the caller **and** the value saved to the checkpoint for the `previous` parameter in the next invocation.

To **decouple return and checkpoint values**, use `entrypoint.final[ReturnType, SaveType]`:

```python
@entrypoint(checkpointer=InMemorySaver())
def my_workflow(
    number: int,
    *,
    previous: int | None = None,
) -> entrypoint.final[int, int]:
    """Return previous value; save doubled value for next invocation."""
    prev = previous or 0
    return entrypoint.final(value=prev, save=2 * number)

config = {"configurable": {"thread_id": "t1"}}
my_workflow.invoke(3, config)  # Returns: 0 (previous was None)
my_workflow.invoke(5, config)  # Returns: 6 (previous was 3 * 2)
```

---

## @task Decorator

The `@task` decorator marks a function as a **task that can be called from within an entrypoint or StateGraph node**. Tasks enable:

- **Composition**: Call tasks from entrypoints or other tasks.
- **Parallelism**: Task invocation returns a future, enabling concurrent execution.
- **Resilience**: Attach retry, cache, and timeout policies.
- **State isolation**: Each task maintains its own scope; state context is managed automatically.

### Signature and Parameters

```python
@task(
    name: str | None = None,
    retry_policy: RetryPolicy | Sequence[RetryPolicy] | None = None,
    cache_policy: CachePolicy[Callable[P, str | bytes]] | None = None,
    timeout: float | timedelta | TimeoutPolicy | None = None,
)
def my_task(arg1: Type1, arg2: Type2) -> ResultType:
    ...
```

**Parameters:**

- **`name`**: Optional name for the task. If omitted, the function's `__name__` is used.
- **`retry_policy`**: A `RetryPolicy` or list of policies. Applied sequentially; each policy defines `initial_interval`, `backoff_factor`, `max_interval`, `max_attempts`, and an exception filter (`retry_on`).
- **`cache_policy`**: A `CachePolicy` with optional `key_func` and `ttl`. Caches the task result by input; results are returned without re-executing the task.
- **`timeout`**: Timeout enforcement. **Only supported for async tasks.** Sync tasks cannot be safely cancelled in-process. A float becomes `run_timeout`; use `TimeoutPolicy` for `run_timeout` + `idle_timeout`.

### Invocation and Futures

When called, a task **does not execute immediately**. Instead, it returns a `SyncAsyncFuture[ResultType]` that acts as both a standard `concurrent.futures.Future` and an awaitable:

```python
@task
def add_one(x: int) -> int:
    return x + 1

@entrypoint()
def workflow(values: list[int]) -> list[int]:
    # Call tasks in parallel
    futures = [add_one(x) for x in values]
    
    # Retrieve results
    results = [f.result() for f in futures]
    return results

workflow.invoke([1, 2, 3])  # Returns: [2, 3, 4]
```

In async contexts, you can `await` futures:

```python
@task
async def async_add_one(x: int) -> int:
    return x + 1

@entrypoint()
async def async_workflow(values: list[int]) -> list[int]:
    futures = [async_add_one(x) for x in values]
    results = await asyncio.gather(*futures)
    return results

await async_workflow.ainvoke([1, 2, 3])  # Returns: [2, 3, 4]
```

### Task Return Type Inference

The task's return type annotation is used to infer the output schema for serialization and introspection:

```python
@task
def process_item(item: dict) -> list[str]:
    """Return type hint informs schema inference."""
    return item["tags"]
```

---

## Retry Policies

A `RetryPolicy` (repo://libs/langgraph/langgraph/types.py#L418-L438) defines **exponential backoff and exception filtering** for task and entrypoint retries.

### Structure

```python
@dataclass
class RetryPolicy:
    initial_interval: float = 0.5
    """Delay before first retry (seconds)."""
    
    backoff_factor: float = 2.0
    """Multiplier for interval after each retry."""
    
    max_interval: float = 128.0
    """Maximum delay between retries (seconds)."""
    
    max_attempts: int = 3
    """Total attempts including the first (max_attempts=3 means 2 retries)."""
    
    jitter: bool = True
    """Add random variance to delay."""
    
    retry_on: type[Exception] | Sequence[type[Exception]] | Callable[[Exception], bool]
    """Exception class(es) or predicate determining retryable failures."""
```

### Usage

```python
from langgraph.types import RetryPolicy

# Single policy: retry on TimeoutError up to 5 times
policy = RetryPolicy(
    max_attempts=5,
    initial_interval=1.0,
    backoff_factor=2.0,
    retry_on=TimeoutError,
)

@task(retry_policy=policy)
async def fetch_data(url: str) -> str:
    """Retry on timeout."""
    return await client.get(url)

# Multiple policies (attempted in sequence)
policies = [
    RetryPolicy(max_attempts=3, retry_on=TimeoutError),
    RetryPolicy(max_attempts=2, retry_on=ValueError),
]

@task(retry_policy=policies)
async def multi_retry_task(value: int) -> int:
    if value < 0:
        raise ValueError("negative")
    return value + 1
```

---

## Cache Policies

A `CachePolicy` (repo://libs/langgraph/langgraph/types.py#L521-L530) enables **memoization of task results based on input**.

### Structure

```python
@dataclass
class CachePolicy(Generic[KeyFuncT]):
    key_func: KeyFuncT = default_cache_key
    """Callable mapping function input to a string or bytes cache key.
    Defaults to hashing the input with pickle."""
    
    ttl: int | None = None
    """Time-to-live in seconds. None means never expire."""
```

### Usage

```python
from langgraph.types import CachePolicy
from langgraph.cache import InMemoryCache

cache = InMemoryCache()

# Default cache key (input hash)
@task(cache_policy=CachePolicy())
def expensive_computation(x: int) -> int:
    print(f"Computing {x}")
    return x ** 2

@entrypoint(cache=cache)
def workflow() -> None:
    result1 = expensive_computation(5).result()  # Computed
    result2 = expensive_computation(5).result()  # Cached
    # Output: "Computing 5" printed only once

# Custom cache key function
def custom_key(x: int) -> str:
    """Cache by range, not exact value."""
    return f"bucket_{x // 10}"

@task(cache_policy=CachePolicy(key_func=custom_key, ttl=60))
def bucketed_task(x: int) -> int:
    return x * 2

@entrypoint(cache=cache)
def workflow2() -> None:
    result1 = bucketed_task(5).result()   # Cached under "bucket_0"
    result2 = bucketed_task(8).result()   # Same cache key "bucket_0" → reused
    result3 = bucketed_task(15).result()  # New cache key "bucket_1" → computed
```

### Cache Isolation by Namespace

Each task's cache is isolated using a namespace derived from the task's identifier (module and function name). This prevents cache collisions across different tasks.

### Clearing Cache

```python
cache = InMemoryCache()

@task(cache_policy=CachePolicy())
def my_task(x: int) -> int:
    return x + 1

# Clear cache for a specific task
my_task.clear_cache(cache)

# Or async
await my_task.aclear_cache(cache)
```

---

## Timeout Policies

Timeouts enforce **maximum execution duration** for tasks and entrypoints. **Timeouts are only supported for async functions**; sync functions cannot be safely cancelled in-process.

### TimeoutPolicy Structure

```python
@dataclass
class TimeoutPolicy:
    run_timeout: float | timedelta | None = None
    """Hard wall-clock cap (seconds) for the entire node attempt.
    Never refreshed by progress signals."""
    
    idle_timeout: float | timedelta | None = None
    """Maximum time without observable progress (seconds).
    Refreshed by callbacks, standard graph signals, and runtime.heartbeat()."""
    
    refresh_on: Literal["auto", "heartbeat"] = "auto"
    """"auto": refresh on any standard progress signal.
    "heartbeat": refresh only on explicit runtime.heartbeat() calls."""
```

### Usage

```python
from langgraph.types import TimeoutPolicy
import asyncio

# Simple wall-clock timeout
@task(timeout=5.0)
async def timeout_task() -> str:
    await asyncio.sleep(10)
    return "done"

# Separate run and idle timeouts
@entrypoint(timeout=TimeoutPolicy(run_timeout=30, idle_timeout=5))
async def workflow() -> str:
    # Entire workflow must finish in 30 seconds
    # If no progress for 5 seconds, abort
    return "result"

# Heartbeat-based progress signaling
@task(timeout=TimeoutPolicy(idle_timeout=2, refresh_on="heartbeat"))
async def long_task() -> str:
    runtime = get_runtime()  # Injected at runtime
    for i in range(10):
        await asyncio.sleep(1)
        runtime.heartbeat()  # Reset idle timeout
    return "done"
```

When a timeout fires, a `NodeTimeoutError` is raised, and the retry policy (if configured) decides whether to retry.

---

## State Context and Implicit Passing

**Tasks implicitly operate on the current graph state.** Unlike `StateGraph` nodes that explicitly receive and return state, tasks work with state automatically through the execution context.

This means:

- **No explicit state parameter**: Tasks don't take the full state as an argument.
- **Closure variables and parameters**: Tasks access their regular parameters and any closure variables defined in the entrypoint.
- **State updates via interrupts and commands**: Advanced state mutation uses `interrupt()` and `Command` primitives.

**Example:**

```python
@task
def summarize(topic: str) -> str:
    """Regular function: no state parameter."""
    return f"Summary of {topic}"

@entrypoint()
def workflow(input_dict: dict) -> str:
    topic = input_dict["topic"]
    
    # Task receives regular parameters, not state
    future = summarize(topic)
    summary = future.result()
    
    return summary

workflow.invoke({"topic": "AI"})
```

---

## Composition Patterns

### Entrypoint Calling Tasks

```python
@task
def fetch(url: str) -> str:
    return requests.get(url).text

@task
def process(text: str) -> dict:
    return {"length": len(text), "words": text.split()}

@entrypoint()
def pipeline(url: str) -> dict:
    text_future = fetch(url)
    text = text_future.result()
    
    result_future = process(text)
    result = result_future.result()
    
    return result
```

### Task Calling Other Tasks

```python
@task
def double(x: int) -> int:
    return x * 2

@task
def add_one(x: int) -> int:
    return x + 1

@task
def compose_operations(x: int) -> int:
    """Task calling other tasks."""
    future1 = double(x)
    result1 = future1.result()
    
    future2 = add_one(result1)
    result2 = future2.result()
    
    return result2

@entrypoint()
def workflow(x: int) -> int:
    return compose_operations(x).result()
```

### Parallel Task Execution

```python
@task
async def fetch_user(user_id: int) -> dict:
    await asyncio.sleep(0.1)
    return {"id": user_id, "name": f"User{user_id}"}

@task
async def fetch_posts(user_id: int) -> list[str]:
    await asyncio.sleep(0.2)
    return [f"Post from user {user_id}"]

@entrypoint()
async def fetch_all(user_id: int) -> dict:
    """Fetch user and posts in parallel."""
    user_future = fetch_user(user_id)
    posts_future = fetch_posts(user_id)
    
    # Both run concurrently
    user = await user_future
    posts = await posts_future
    
    return {"user": user, "posts": posts}

await fetch_all.ainvoke(123)
```

---

## Checkpointing and Resumption

When an entrypoint has a `checkpointer`, you can **pause, interrupt, and resume workflows** using the same `thread_id`:

```python
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import interrupt, Command

@task
def ask_user(question: str) -> str:
    return interrupt({"question": question})

@entrypoint(checkpointer=InMemorySaver())
def approval_workflow(request: str) -> dict:
    """Workflow that interrupts for human approval."""
    result = ask_user(f"Approve: {request}?").result()
    
    if result == "approved":
        return {"status": "approved", "request": request}
    else:
        return {"status": "rejected", "request": request}

config = {"configurable": {"thread_id": "workflow_1"}}

# Start the workflow
for event in approval_workflow.stream({"request": "deploy v2"}, config):
    print(event)
    # Output: interrupted, awaiting user input

# Resume with approval
for event in approval_workflow.stream(
    Command(resume="approved"),
    config,
):
    print(event)
    # Workflow resumes, completes with approved status
```

---

## Error Handling and Resilience

Retry and timeout policies work together to handle failures gracefully:

```python
import random
from langgraph.types import RetryPolicy, TimeoutPolicy

@task(
    retry_policy=RetryPolicy(
        max_attempts=3,
        initial_interval=0.5,
        retry_on=Exception,
    ),
    timeout=TimeoutPolicy(idle_timeout=10),
)
async def unreliable_task() -> str:
    if random.random() < 0.7:
        raise ValueError("Random failure")
    return "success"

@entrypoint()
async def resilient_workflow() -> str:
    try:
        result = await unreliable_task()
        return result
    except Exception as e:
        return f"Failed after retries: {e}"
```

---

## Stream Modes and Introspection

Entrypoints are `Pregel` instances with full streaming support:

```python
@entrypoint()
def workflow(x: int) -> int:
    return x * 2

# invoke: single execution
result = workflow.invoke(5)  # Returns: 10

# stream with different modes
for update in workflow.stream(5, stream_mode="updates"):
    print(update)

for value in workflow.stream(5, stream_mode="values"):
    print(value)

# Async
async for event in workflow.astream(5, stream_mode="debug"):
    print(event)
```

---

## Type Hints and Schema Inference

Return type annotations are used for schema inference and serialization:

```python
from typing import TypedDict

class OutputSchema(TypedDict):
    score: float
    label: str

@task
def classify(text: str) -> OutputSchema:
    """Return type drives schema inference."""
    return {"score": 0.95, "label": "positive"}

# Schema introspected from return annotation
print(classify.output_schema)
```

---

## Async Support

Both `@task` and `@entrypoint` support **async functions**. The decorator automatically handles async execution:

```python
@task
async def async_task(x: int) -> int:
    await asyncio.sleep(0.1)
    return x + 1

@entrypoint()
async def async_entrypoint(x: int) -> int:
    future = async_task(x)
    result = await future  # Await the future
    return result + 1

# Invoke async
result = await async_entrypoint.ainvoke(5)  # Returns: 7

# Stream async
async for event in async_entrypoint.astream(5):
    print(event)
```

**Important:** Timeouts are only available for async functions. Sync functions cannot be safely cancelled in-process.

---

## Comparison with StateGraph

| Aspect | `@entrypoint/@task` | `StateGraph` |
|--------|---------------------|------------|
| **Style** | Function-based, declarative | Graph-based, procedural |
| **Complexity** | Simple workflows, composition | Complex topologies, DAGs |
| **State Management** | Implicit via context | Explicit dict/TypedDict |
| **Node Setup** | `@task` decorator | `add_node()` method |
| **Edges** | Via task invocation | `add_edge()`, `add_conditional_edges()` |
| **Parallelism** | Natural (futures) | Via parallel node execution |
| **When to Use** | Sequential logic, composition | Branching, loops, complex routing |

---

## Implementation Notes

- **`SyncAsyncFuture`** (repo://libs/langgraph/langgraph/pregel/_call.py#L253): A custom future type that is both a standard `concurrent.futures.Future` and an awaitable, enabling seamless sync/async composition.
- **`_TaskFunction`** (repo://libs/langgraph/langgraph/func/__init__.py#L59-L107): Wraps decorated task functions with retry, cache, and timeout logic. Calls are intercepted via `_call_with_options()`.
- **Entrypoint as Pregel** (repo://libs/langgraph/langgraph/func/__init__.py#L516-L620): The `entrypoint.__call__()` method converts the decorated function into a `Pregel` graph with a single node, automatically managing channels, writers, and reducers.
- **Config Resolution** (repo://libs/langgraph/config.py): The `get_config()` context manager retrieves the current `RunnableConfig`, enabling task access to runtime, checkpointer, and store.

---

## Examples

### Simple Retry and Cache

```python
from langgraph.func import entrypoint, task
from langgraph.types import RetryPolicy, CachePolicy
from langgraph.cache import InMemoryCache

cache = InMemoryCache()

call_count = 0

@task(
    retry_policy=RetryPolicy(max_attempts=2),
    cache_policy=CachePolicy(),
)
def expensive_op(x: int) -> int:
    global call_count
    call_count += 1
    if call_count == 1:
        raise ValueError("First attempt fails")
    return x ** 2

@entrypoint(cache=cache)
def workflow(x: int) -> int:
    return expensive_op(x).result()

# First call: fails, retries, succeeds
assert workflow.invoke(5) == 25

# Second call with same input: returned from cache
assert workflow.invoke(5) == 25
assert call_count == 1  # Not recomputed
```

### Async Parallelism

```python
import asyncio
from langgraph.func import entrypoint, task

@task
async def fetch(url: str) -> str:
    await asyncio.sleep(0.1)
    return f"Data from {url}"

@entrypoint()
async def fetch_all(urls: list[str]) -> list[str]:
    futures = [fetch(url) for url in urls]
    return await asyncio.gather(*futures)

result = await fetch_all.ainvoke([
    "http://a.com",
    "http://b.com",
    "http://c.com",
])
# All 3 fetches run concurrently (total ~0.1s, not 0.3s)
```

### Stateful Workflow with Checkpointing

```python
from langgraph.func import entrypoint
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command, interrupt

@entrypoint(checkpointer=InMemorySaver())
def count_workflow(
    increment: int,
    *,
    previous: int | None = None,
) -> entrypoint.final[int, int]:
    """Accumulate a count, resumable via Command."""
    count = (previous or 0) + increment
    
    if count < 10:
        # Pause for approval
        approval = interrupt({"count": count, "message": "Continue?"})
        if approval == "yes":
            return entrypoint.final(value=count, save=count)
        else:
            return entrypoint.final(value=count, save=0)
    else:
        return entrypoint.final(value=count, save=count)

config = {"configurable": {"thread_id": "counter"}}

# First run: accumulate to 5, ask for approval
for event in count_workflow.stream({"increment": 5}, config):
    print(event)

# Resume with approval
for event in count_workflow.stream(Command(resume="yes"), config):
    print(event)
    # count workflow resumes with count=5 saved

# Next invocation: add 7 to previous 5 = 12 (halts as count >= 10)
for event in count_workflow.stream({"increment": 7}, config):
    print(event)
```
