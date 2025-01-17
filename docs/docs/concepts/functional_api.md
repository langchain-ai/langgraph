# Functional API

## Overview

The Functional API is designed to provide an imperative programming model for building LangGraph workflows.

LangGraph workflows support parallel execution of tasks, [persistence](persistence.md), recovery from failures, [human-in-the-loop](human_in_the_loop.md), and real-time [streaming](streaming.md) from anywhere in the workflow.

## Interface

The Functional API consists of two main primitives: `entrypoint` and `task`.

- **Entrypoint**: Defines the starting point for a LangGraph workflow.
- **Task**: A unit of work that can be executed. Tasks can be executed in parallel or sequentially within the scope of an entrypoint.

## Functional vs. Graph API

The functional and graph APIs are two ways to define workflows in LangGraph. They are both built on top of the same LangGraph runtime, allowing you to intermix the two APIs in the same application.

The main difference between the two APIs is the programming model they provide:

* **Functional API**: Provides an *imperative* programming model for building workflows. It allows you to define workflows using standard Python functions and decorators.
* **Graph API**: Provides a *declarative* programming model for defining control flow in a workflow using a state machine (graph).

The primary advantage of using the **Graph API** is that it:

* 
* Define explicit states for separate parts of the workflow which can make it easiest to reason about what 


## Example

Here's a quick example to demonstrate the Functional API in action:

```python
import time
from langgraph.func import entrypoint, task
from langgraph.types import interrupt, Command
from langgraph.checkpoint.memory import MemorySaver

@task
def write_essay(topic: str) -> str:
    """Write an essay about the given topic."""
    time.sleep(1) # Simulate a long-running task; e.g., due to an LLM call.
    return f"An essay about {topic}"

@entrypoint(checkpointer=MemorySaver())
def workflow(inputs: dict) -> dict:
    tasks = [write_essay(topic) for topic in inputs['topics']]
    essays = [t.result() for t in tasks]
    # Human-in-the-loop interrupt
    value = interrupt({
        "question": "Are these good?",
        "essays": essays
    })
    return {
        "essays": essays,
        "review_result": value,
    }

config = {
    "configurable": {
        "thread_id": "some_thread"
    }
}

for item in workflow.stream({"topics": ["cats", "dogs"]}, config):
    print(item)


for item in workflow.stream(Command(resume="yes"), config):
    print(item)
```

```pycon
{'write_essay': 'An essay about dogs'}
{'write_essay': 'An essay about cats'}
{'__interrupt__': (Interrupt(value={'question': 'Are these good?', 'essays': ['An essay about cats', 'An essay about dogs']}, resumable=True, ns=['workflow:8f5c1b6e-a5b5-9107-30d8-c163528a35f8'], when='during'),)}
{'workflow': {'essays': ['An essay about cats', 'An essay about dogs'], 'review_result': 'yes'}}
```


The Functional API enables streamlined, Pythonic workflows. The entrypoint serves as the starting point for a functional graph, while task defines units of work that can execute independently or in sequence. The API supports features like checkpointing, streaming, and deterministic execution, making it both flexible and scalable.

## Workflows must be deterministic

To leverage features like **human-in-the-loop** and **recovery from failures**, workflows must be **deterministic**.
This means that the workflow should be written in a way such that a given set of inputs always produces the same set of outputs.


### Difference from the Graph API

LangGraph's Graph API uses an explicit state machine to define control flow, enabling precise control over branching and cyclical behavior.

The Functional API, in contrast, does not require explicit control flow declarations. Instead, tasks and entrypoints can be called directly, just like standard Python functions. This approach simplifies development while maintaining the flexibility to handle complex workflows.

## Entrypoint

An entrypoint is a decorator that you can apply

### Interface

```python

```

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

### Running tasks in parallel

Tasks can be executed in parallel by invoking them concurrently and waiting for the results. This can improve performance by leveraging multiple cores or distributed resources.

```python
import time
from langgraph.func import task, entrypoint

    
@task
def slow_add_one(x: int) -> int:
    """A slow task that takes 1 second to complete."""
    time.sleep(1)
    return x

@entrypoint

```





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
