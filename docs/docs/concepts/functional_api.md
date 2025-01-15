# Functional API

## Overview

LangGraph's functional API allows building applications that persist state in between successive invocations.

The applications are able to manage state, work with LLMs, support HIL patterns and stream results.

The functional API is built using two key primitives: `entrypoint` and `task`.

You can intermix the Functional and the Graph API in the same application by invoking
`@entrypoint` or `@task` from within a node in a state graph or vice versa invoking a stqte graph from
inside a task or entrypoint.


## Example

Here's a quick example to get the gist of the Functional API:

=== "Map-reduce"

    ```python
    from langgraph.func import entrypoint, task
    from langgraph.types import interrupt

    @task()
    def mapper(i: int) -> str:
        return str(i) * 2

    @entrypoint()
    def graph(input: list[int]) -> list[str]:
        futures = [mapper(i) for i in input]
        mapped = [f.result() for f in futures]
        answer = interrupt("question")
        return [m + answer for m in mapped]
    ```




The Functional API is composed of two key primitives: entrypoint and task. These primitives work together to define modular, reusable workflows. While entrypoint serves as the starting point for a functional graph, task encapsulates units of work that can be executed independently or in sequence. This API is designed for flexibility and scalability, supporting features like checkpointing, streaming, and deterministic execution.


### Difference from the Graph API

- **Entrypoint**: Focuses on executing a sequence of tasks with a functional, linear workflow.
- **StateGraph**: Phrases the control flow in terms of a state machine that allows branching and cycles. This makes checkpointing and time-travel easier, but comes at the cost of more boilerplate code.

## Entrypoint

### What is an entrypoint?

An entrypoint is a decorated function that serves as the starting point for executing a functional workflow. It defines the input and output boundaries for a graph of tasks and provides mechanisms for checkpointing and streaming results.


### How to define an entrypoint

An entrypoint is defined using the `@entrypoint` decorator. It accepts optional parameters such as `checkpointer` for checkpointing purposes.

#### Example:

```python
@entrypoint(checkpointer=checkpointer)
def graph(input: list[int]) -> list[str]:
    futures = [mapper(i) for i in input]
    mapped = [f.result() for f in futures]
    answer = interrupt("question")
    return [m + answer for m in mapped]
```

### How to invoke an entrypoint

- Call the function with appropriate inputs.
- Use `.stream()` for streaming results if supported.

#### Example:

```python
thread1 = {"configurable": {"thread_id": "1"}}
assert [*graph.stream([0, 1], thread1)] == [
    {"mapper": "00"},
    {"mapper": "11"},
    {"__interrupt__": (Interrupt(value="question", resumable=True, ns=[AnyStr("graph:")], when="during"))},
]
```

### What happens when one yields from an entrypoint?

Yielding from an entrypoint streams intermediate results to the consumer. This can be useful for long-running workflows or providing partial results in real-time.

## Task

### When to use a task

Use a task when you need to encapsulate a unit of work that can be executed independently. Tasks are useful for modularizing workflows, enabling parallel execution, and supporting checkpointing.

### How to define a task

Tasks are defined using the `@task` decorator. The function should adhere to the serialization requirements for inputs and outputs.

#### Example:

```python
@task()
def foo(state: dict) -> tuple:
    return state["a"] + "foo", "bar"

@task
def bar(a: str, b: str, c: Optional[str] = None) -> dict:
    return {"a": a + b, "c": (c or "") + "bark"}
```

### How to call a task

Call the task function like a regular Python function. The result is typically a future object that can be awaited or resolved.

#### Example:

```python
fut_foo = foo(state)
fut_bar = bar(*fut_foo.result())
```

### Serialization requirements for `task`

- Inputs and outputs must be serializable to ensure compatibility with checkpointing and distributed execution.
- Avoid complex, non-serializable objects unless custom serialization logic is implemented.

### Eager map-reduce

The `@task` decorator supports eager evaluation of map-reduce operations, allowing partial results to be processed immediately as they become available.

### Snapshot information

The results and states of tasks are stored in a snapshot, which tracks the progression of the workflow and aids in debugging or recovery.

### Using tasks from within a stategraph

Tasks can be called from within a stategraph, enabling fine-grained control over execution while benefiting from the modularity of tasks.

### Deterministic execution and side-effects

- Tasks must produce deterministic results given the same inputs.
- Side-effects should be minimized or managed explicitly to ensure reproducibility and consistent checkpointing.

### Verifying `task` decorator on methods

The `@task` decorator works on both standalone functions and class methods. Ensure proper use of `self` or `cls` when decorating methods.

#### Example:

```python
class MyClass:
    @task
    def my_method(self, value: int) -> int:
        return value * 2
```
