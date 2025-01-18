# Functional API

## Overview

LangGraph's **Functional API** allows creating workflows using Python decorators and standard control 
flow constructs (e.g., loops, conditionals) to define complex workflows needing to define a graph.

The Functional API and the Graph API use the same LangGraph runtime, which manages the execution of workflows and provides [persistence](persistence.md), [human-in-the-loop](human_in_the_loop.md), and [streaming](streaming.md).

The Functional and Graph APIs can be used together in the same application, allowing you to choose the best approach for each part of your workflow.

## See it in action

```python
import time

from langgraph.func import entrypoint, task
from langgraph.types import interrupt
from langgraph.checkpoint.memory import MemorySaver

@task
def write_essay(topic: str) -> str:
    """Write an essay about the given topic."""
    time.sleep(1) # This is a placeholder for a long-running task.
    return f"An essay about topic: {topic}"

@entrypoint(checkpointer=MemorySaver())
def workflow(topic: str) -> dict:
    """A simple workflow that write an essay and asks for a review."""
    essay = write_essay("cat").result()
    value = interrupt({
        # Any json-serializable value can be used here.
        "action": "review",
        "essays": essay
    })
    return {
        "essay": essay,
        "review": value,
    }

config = {
    "configurable": {
        "thread_id": "some_thread"
    }
}

for item in workflow.stream("cat", config):
    print(item)
```

```python
{'write_essay': 'An essay about topic: cat'}
{'__interrupt__': (Interrupt(value={'action': 'review', 'essays': 'An essay about topic: cat'}, resumable=True, ns=['workflow:875155a0-97c1-7dc8-c216-f37c8d6b83c1'], when='during'),)}
```

An essay has been written and is ready for review. Once the review is provided, we can resume the workflow:

```python
from langgraph.types import Command

for item in workflow.stream(Command(resume="This essay is great."), config):
    print(item)
```

```pycon
{'workflow': {'essay': 'An essay about topic: cat', 'review': 'This essay is great.'}}
```

When checkpointing is enabled, LangGraph automatically saves the state of the workflow including
the results of tasks and interrupts. 

!!! tip

    When the workflow was resumed, it executes from the very start, but because 
    the result of the `write_essay` task was already saved, it doesn't need to be recomputed
    and instead is loaded from the checkpoint.


## Primitives

The Functional API consists of two primitives:

- **[Entrypoint](#entrypoint)**: Defines the starting point for a LangGraph workflow.
- **[Task](#task)**: A unit of work that can be executed. Tasks can be executed in parallel or sequentially within the scope of an entrypoint.


## Functional API vs. Graph API

The Functional API and the Graph APIs provide two different paradaigms to create workflows in LangGraph.

These APIs can be thought of as two different paradigms for defining workflows, with the following differences:

- **Resuming graph execution**: The execution of a LangGraph application can be interrupted (e.g., for human in the loop or due to an error). When the application is resumed, the 
- **Control flow**: The Functional API does not require thinking about graph structure or state machines. You can use standard Python constructs to define workflows.
- **GraphState** and **reducers**: In the functional API, the different functions can 
- **Visualization**: The Graph API makes it easy to visualize the workflow as a graph which can be useful for debugging, understanding the workflow, and sharing with others.
* **Functional API**: Utilizes an *imperative* programming model for constructing **workflows**. The control flow leverages standard Python primitives, such as conditionals (if/else), loops (for/while), and function calls. This approach allows for a more traditional, step-by-step execution of tasks.
* **Graph API**: Offers a *declarative* programming model for specifying control flow within a **workflow** through the use of a state machine (graph). This approach defines the workflow as a series of nodes and edges, where each node represents a task or, and edges define the flow of execution between them.

## Workflows must be deterministic

To leverage features like **human-in-the-loop** and **recovery from failures**, workflows must be **deterministic**.
This means that the workflow should be written in a way such that a given set of inputs always produces the same set of outputs.

## Entrypoint

An entrypoint is a decorator that you can apply

For more information please see the API reference for the [entrypoint][langgraph.func.entrypoint] decorator.

## Task

For more information please see the API reference for the [task][langgraph.func.task] decorator.

### Interface

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
