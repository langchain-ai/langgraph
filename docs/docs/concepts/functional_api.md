# Functional API

!!! warning "Experimental"
    The Functional API is currently in **beta** and is subject to change. Please report any issues or feedback to the LangGraph team.

Things that need to be clarified

1. Thread id and persistence
2. thread id vs. run!
3. serialization
4. determinism
5. idempotency & side-effects
6. parallelization


## Overview

A **workflow** is a sequence of logic that defines the flow of an application. In LangGraph, workflows can be built using the Functional API or the Graph API.

LangGraph's **Functional API** allows using standard control flow constructs (e.g., loops, conditionals) to define complex [workflows](#workflow) without having to define a graph.

The Functional API runs on the same LangGraph runtime as the Graph API, providing access to features like [persistence](persistence.md), [human-in-the-loop](human_in_the_loop.md), and [streaming](streaming.md).

The Functional and Graph APIs can be used together in the same application, allowing you to intermix the two paradigms if needed.


```python
@task
def write_essay(topic: str) -> str:
    """Write an essay about the given topic."""
    time.sleep(1) # A placeholder for a long-running task.
    return f"An essay about topic: {topic}"

@entrypoint(checkpointer=MemorySaver())
def workflow(topic: str) -> dict:
    """A simple workflow that write an essay and asks for a review."""
    essay = write_essay("cat").result()
    # Interrupt the workflow to get a review from a human.
    value = interrupt({ 
        # Any json-serializable value to surface to the client.
        "action": "review",
        "essays": essay
    })
    return {
        "essay": essay,
        "review": value,
    }
```

??? example "Full Code"

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

## Building Blocks

The Functional API provides two building blocks for defining workflows:

- **[Entrypoint](#entrypoint)**: Define a workflow which can include calls to tasks, other entrypoints, or state graph nodes. Entrypoints configured with a checkpointer enable workflow interruption and resumption, allowing human-in-the-loop interactions.

- **[Task](#task)**: Represents a discrete unit of work, such as an API call or data processing step, that can be executed asynchronously. Invoking a task returns a future-like object, which can be awaited to obtain the result or resolved synchronously.

These building blocks provide flexibility to structure workflows while ensuring modularity, reusability, and persistence support.

## Entrypoint

An entrypoint is a decorator that designates a function as the starting point of a LangGraph workflow. It encapsulates workflow logic and manages execution flow, including handling long-running tasks and interruptions.

Entrypoints typically include a checkpointer to persist workflow state, enabling resumption from where it was paused.

### Defining an Entrypoint 

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


### Executing

The `@entrypoint` yields a Pregel Runnable object.

=== "Invoke"

    ```python
    my_workflow.invoke(42).result()  # Wait for the result synchronously
    ```

=== "Async Invoke"

    ```python
    await my_workflow.invoke(42)  # Await result asynchronously
    ```

=== "Stream"
    
    ```python
    for item in my_workflow.stream(42):
        print(item)
    ```

=== "Async Stream"

    ```python
    async for item in my_workflow.stream(42):
        print(item)
    ```

### Resuming

When an entrypoint is interrupted, it can be resumed by providing the appropriate command.

=== "Resume"

    ```python

    from langgraph.types import Command
    ```


## Task

A task is a distinct unit of work in a workflow, such as making an API call, processing data, or performing computations. Tasks support persistence with checkpointing, allowing workflow execution to resume without recomputing results. They can also run asynchronously, enabling parallel execution for better performance.

### Defining a Task

Tasks are defined using the `@task` decorator, which wraps a regular Python function.

```python
from langgraph.func import task

@task()
def slow_computation(input_value):
    # Simulate a long-running operation
    ...
    return result
```

!!! important "Serialization"

    The inputs and outputs of tasks must be JSON-serializable to support checkpointing.

### Invoking a Task

Tasks can only be called from within an entrypoint, another task, or a state graph node. They **cannot** be called directly from the main application code. Calling a task produces a future-like object that can be awaited or resolved to obtain the result.

=== "Synchronous Invocation"

    ```python
    @entrypoint(checkpointer=checkpointer)
    def my_workflow(input: int) -> int:
        future = slow_computation(input)
        return future.result()  # Wait for the result synchronously
    ```

=== "Asynchronous Invocation"

    ```python
    @entrypoint(checkpointer=checkpointer)
    async def my_workflow(input: int) -> int:
        return await slow_computation(input)  # Await result asynchronously
    ```

### Lifecycle

1. **Invocation:** Calling a task returns a future-like object.
2. **Checkpointing:** Results are saved to the persistence layer.
3. **Interruption Handling:** Workflows can be paused and resumed without re-executing completed tasks.

### When to Use a Task


Use tasks when:

- Work needs to be retried to handle failures or inconsistencies.
- Parallel execution is beneficial for I/O-bound tasks, allowing multiple operations to run concurrently without blocking, but it does not improve performance for CPU-bound tasks.
- Results must be persisted to resume workflows without repeating computations or API calls.

For more details, refer to the [API reference](#task-api).

## Serialization

## Determinism

To leverage features like human-in-the-loop and failure recovery, workflows must encapsulate randomness within tasks. This guarantees that upon resumption, the workflow follows the same sequence of steps, even if task results are non-deterministic.

LangGraph accomplishes this by persisting task and sub-graph results as they execute. A well-designed workflow ensures that resuming execution follows the same sequence of steps, allowing previously computed results to be retrieved without re-execution.

### Importance of Capturing Randomness

Ensuring proper state capture is crucial for:

* Correct Resumption: After an interruption, the workflow should pick up exactly where it left off without re-executing tasks unnecessarily.

* Reliable Checkpointing: Persisted results should align with the workflow's logical flow, ensuring data integrity.

* Predictable Resumption: Even if workflows include randomness, resuming execution should always follow the same recorded sequence of events.

### Key Takeaways

Place operations involving randomness inside tasks to ensure results are saved and reused when resuming.

Avoid using dynamic values like timestamps or random numbers directly in workflow logic unless they are captured within tasks.

While different runs of a workflow can produce different results, resuming a specific run should always follow the same sequence of recorded steps.


## Common Pitfalls

### Non-deterministic Control Flow

=== "Non-deterministic control flow (incorrect)"

    In this example, the workflow uses the current time to determine which task to execute. This is non-deterministic because the result of the workflow depends on the time at which it is executed.

    ```python
    from langgraph.func import entrypoint

    @entrypoint(checkpointer=checkpointer)
    def my_workflow(inputs: dict) -> int:
        t0 = inputs["t0"]
        # highlight-next-line
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

    # highlight-next-line
    @task
    # highlight-next-line
    def get_time() -> float:
        return time.time()

    @entrypoint(checkpointer=checkpointer)
    def my_workflow(inputs: dict) -> int:
        t0 = inputs["t0"]
        # highlight-next-line
        t1 = get_time().result()
        
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

### Side-effects not in Tasks

=== "Incorrect Example"

    In this example, a side effect (writing to a file) is directly included in the workflow, making resumption inconsistent.

    ```python
    @entrypoint(checkpointer=checkpointer)
    def my_workflow(inputs: dict) -> int:
        # This code will be executed a second time when resuming the workflow.
        # Which is likely not what you want.
        # highlight-next-line
        with open("output.txt", "w") as f:
            # highlight-next-line
            f.write("Side effect executed")
        value = interrupt("question")
        return value
    ```

=== "Correct Example"

    In this example, the side effect is encapsulated in a task, ensuring consistent execution upon resumption.

    ```python
    from langgraph.func import task

    # Define a task 
    # highlight-next-line
    @task
    # highlight-next-line
    def write_to_file():
        with open("output.txt", "w") as f:
            f.write("Side effect executed")

    @entrypoint(checkpointer=checkpointer)
    def my_workflow(inputs: dict) -> int:
        # The side effect is now encapsulated in a task.
        write_to_file().result()
        value = interrupt("question")
        return value
    ```


## Patterns

### Parallel Execution

Tasks can be executed in parallel by invoking them concurrently and waiting for the results. This is useful for improving performance in IO bound tasks (e.g., calling APIs for LLMs).

```python
@entrypoint(checkpointer=checkpointer)
def graph(input: list[int]) -> list[str]:
    futures = [mapper(i) for i in input]
    mapped = [f.result() for f in futures]
    answer = interrupt("question")
    return [m + answer for m in mapped]
```


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
```


### Calling Subgraphs

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


## Functional API vs. Graph API

The Functional API and the Graph APIs provide two different paradaigms to create workflows in LangGraph.

These APIs can be thought of as two different paradigms for defining workflows, with the following differences:

- **Resuming graph execution**: The execution of a LangGraph application can be interrupted (e.g., for human in the loop or due to an error). When the application is resumed, the
- **Control flow**: The Functional API does not require thinking about graph structure or state machines. You can use standard Python constructs to define workflows.
- **GraphState** and **reducers**: In the functional API, the different functions can
- **Visualization**: The Graph API makes it easy to visualize the workflow as a graph which can be useful for debugging, understanding the workflow, and sharing with others.
* **Functional API**: Utilizes an *imperative* programming model for constructing **workflows**. The control flow leverages standard Python primitives, such as conditionals (if/else), loops (for/while), and function calls. This approach allows for a more traditional, step-by-step execution of tasks.
* **Graph API**: Offers a *declarative* programming model for specifying control flow within a **workflow** through the use of a state machine (graph). This approach defines the workflow as a series of nodes and edges, where each node represents a task or, and edges define the flow of execution between them.



## What happens when one yields from an entrypoint?

Yielding from an entrypoint streams intermediate results to the consumer. This can be useful for long-running workflows or providing partial results in real-time.

