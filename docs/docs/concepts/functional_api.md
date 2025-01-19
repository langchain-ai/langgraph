# Functional API

## Overview

LangGraph's **Functional API** allows creating workflows using Python decorators and standard control flow constructs (e.g., loops, conditionals) to define complex workflows without having to define a graph.

The Functional API uses the same LangGraph runtime as the Graph API, allowing you to benefit from features such as [persistence](persistence.md), [human-in-the-loop](human_in_the_loop.md), and [streaming](streaming.md) capabilities using an imperative programming approach. 

The Functional and Graph APIs can be used together in the same application, allowing you to intermix the two paradigms if needed.

??? example "Example: Write an essay and ask for a review"

    This workflow will write an essay about the topic "cat" and then pause to get a review from a human. The workflow can be interrupted for an indefinite amount of time until a review is provided.

    When the workflow is resumed, it executes from the very start, but because the result of the `write_essay` task was already saved, the task will result will be loaded from the checkpoint instead of being recomputed.

    ```python
    import time
    import uuid

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

    thread_id = str(uuid.uuid4())

    config = {
        "configurable": {
            "thread_id": thread_id
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

    The workflow has been completed and the review has been added to the essay.


## Workflow

A workflow defines the flow of an application. In LangGraph workflows can be built using the Functional API (using `entrypoint` and `task` decorators) or the Graph API (using a state machine) or a combination of both.

LangGraph workflows are backed by a **persistence layer** that allows interrupting for human-in-the-loop, recovering from failures, and resuming from checkpoints.


## Building Blocks

The functional API introduces two building blocks for defining workflows:

- **[Entrypoint](#entrypoint)**: Defines a starting point for a workflow. It can include other workflows, tasks or graphs.
- **[Task](#task)**: A unit of work that can be executed like a future. Tasks can be used to encapsulate logic, parallelize execution, and support checkpointing.

### Entrypoint

An entrypoint is a decorator that you can apply to a function to define a LangGraph workflow. An entrypoint will usually be created
with a checkpointer so that the state of the workflow is persisted.

Here's a simple example of an entrypoint that adds one to an input:

```python
from langgraph.func import entrypoint

@entrypoint(checkpointer=checkpointer)
def my_workflow(input: int) -> int:
    # some logic that may involve long-running tasks like API calls,
    # and may be interrupted for human-in-the-loop.
    ...
    return result
```

For more information please see the API reference for the [entrypoint][langgraph.func.entrypoint] decorator.

### Task

Tasks are useful to define a unit of work that can be executed and whose result can be saved and restored.

Use a task when you need to encapsulate a unit of work that can be executed independently. Tasks are useful for modularizing workflows, enabling parallel execution, and supporting checkpointing.


Tasks are defined using the `@task` decorator. The function should adhere to the serialization requirements for inputs and outputs.


```python
from langgraph.func import task

@task()
def slow(some_input):
    # Some long-running task (e.g., API call to an LLM)
    ...
    return result
```

A task can be called from within an entrypoint, another task or a state graph node.

Tasks are called like a regular Python function, but the result is typically a future object that can be awaited or resolved.

=== "Sync"

    ```python
    @entrypoint(checkpointer=checkpointer)
    def my_workflow(input: int) -> int:
        fut = slow(input)
        return fut.result() # Sync
    ```

=== "Async"

    ```python
    @entrypoint(checkpointer=checkpointer)
    async def my_workflow(input: int) -> int:
        return await async_slow(input)
    ```

## Determinism


To leverage features like **human-in-the-loop** and **recovery from failures**, workflows must be **deterministic**.
This means that the workflow should be written in a way such that a given set of inputs always produces the same set of outputs.

For applications to work correctly with LangGraph's persistence layer, they should follow the same sequence of steps when
resuming the workflow following an interruption.

=== "Non-deterministic control flow (incorrect)"

    In this example, the workflow uses the current time to determine which task to execute. This is non-deterministic because the result of the workflow depends on the time at which it is executed.

    ```python
    from langgraph.func import entrypoint

    @entrypoint(checkpointer=checkpointer)
    def my_workflow(inputs: dict) -> int:
        t0 = inputs["t0"]
        t1 = time.time()
        
        delta_t = t1 - t0
        
        if delta_t > 1:
            result = slow_task(1).result()
            value = interrupt("question")
        else:
            result = slow_task(2).result()
            value = interrupt("question")
            
        return {
            "result": result,
            "value": value
        }
    ```

=== "Deterministic control flow (correct)"

    In this example, the workflow uses the input `t0` to determine which task to execute. This is deterministic because the result of the workflow depends only on the input.

    ```python

    import time
    from langgraph.func import task

    @task
    def get_time() -> float:
        return time.time()

    @entrypoint(checkpointer=checkpointer)
    def my_workflow(inputs: dict) -> int:
        t0 = inputs["t0"]
        t1 = time.time()
        
        delta_t = t1 - t0
        
        if delta_t > 1:
            result = slow_task(1).result()
            value = interrupt("question")
        else:
            result = slow_task(2).result()
            value = interrupt("question")
            
        return {
            "result": result,
            "value": value
        }
    ```

Code that yields non determinism results should be placed inside of tasks whose results are saved and restore, without
being recomputed when resuming a given workflow execution.

## Patterns

### Parallel Execution

Tasks can be executed in parallel by invoking them concurrently and waiting for the results. This is useful for improving performance in IO bound tasks (e.g., calling APIs for LLMs).

```python
from langgraph.func import task, entrypoint

```


## Calling Subgraphs

The functional API and the graph API can be used together in the same application as they share the same underlying runtime.

```python
from langgraph.func import entrypoint

@entrypoint()
def some_workflow(x: int) -> int:
    result_1 = graph.invoke(...)
    # Another graph call
    result_2 = some_other_graph.invoke(...)
    return {
        "result_1": result_1,
        "result_2": result_2
    }
```

## Checkpointing

The Functional API supports checkpointing, which allows the state of the workflow to be saved and restored. This is useful for recovering from failures, pausing and resuming workflows, and improving performance by avoiding recomputation.

## Functional API vs. Graph API

The Functional API and the Graph APIs provide two different paradaigms to create workflows in LangGraph.

These APIs can be thought of as two different paradigms for defining workflows, with the following differences:

- **Resuming graph execution**: The execution of a LangGraph application can be interrupted (e.g., for human in the loop or due to an error). When the application is resumed, the
- **Control flow**: The Functional API does not require thinking about graph structure or state machines. You can use standard Python constructs to define workflows.
- **GraphState** and **reducers**: In the functional API, the different functions can
- **Visualization**: The Graph API makes it easy to visualize the workflow as a graph which can be useful for debugging, understanding the workflow, and sharing with others.
* **Functional API**: Utilizes an *imperative* programming model for constructing **workflows**. The control flow leverages standard Python primitives, such as conditionals (if/else), loops (for/while), and function calls. This approach allows for a more traditional, step-by-step execution of tasks.
* **Graph API**: Offers a *declarative* programming model for specifying control flow within a **workflow** through the use of a state machine (graph). This approach defines the workflow as a series of nodes and edges, where each node represents a task or, and edges define the flow of execution between them.


## Handling side-effects

If you're using `interrupt` to pause the workflow and wait for a response from a human, you should generally place side-effects inside tasks. This ensures that 

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


## Considerations

### Serialization

When `checkpointing` is enabled, the inputs and outputs of tasks, entrypoints, and other functions

### Serialization requirements for `task`

- Inputs and outputs must be serializable to ensure compatibility with checkpointing and distributed execution.
- Avoid complex, non-serializable objects unless custom serialization logic is implemented.

### Side-effects

### Idempotency


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

@entrypoint()
def parallel_workflow(x: int) -> int:
    """A workflow that runs two tasks in parallel."""
    fut1 = slow_add_one(x) # This task will run in parallel with the next one.
    fut2 = slow_add_one(x)
    return fut1.result() + fut2.result() # Wait for both tasks to complete and return the sum.
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
