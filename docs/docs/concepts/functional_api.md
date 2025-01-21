# Functional API

!!! warning "Experimental"
    The Functional API is currently in **beta** and is subject to change. Please report any issues or feedback to the LangGraph team.

## Overview

A **workflow** is a sequence of logic that defines the flow of an application. In LangGraph, workflows can be built using the Functional API or the Graph API.

LangGraph's **Functional API** allows using standard control flow constructs (e.g., loops, conditionals) to define complex [workflows](#workflow) without having to define a graph.

The Functional API runs on the same LangGraph runtime as the Graph API, providing access to features like [persistence](persistence.md), [human-in-the-loop](human_in_the_loop.md), and [streaming](streaming.md).

The Functional and Graph APIs can be used together in the same application, allowing you to intermix the two paradigms if needed.

## Example

```python
from langgraph.func import entrypoint, task

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

The Functional API provides two building blocks for building workflows:

- **[Entrypoint](#entrypoint)**: Defines a workflow that can include calls to tasks, other entrypoints, or state graph nodes. Entrypoints configured with a **checkpointer** enable workflow interruption and *resumption*, allowing human-in-the-loop interactions.

- **[Task](#task)**: Represents a discrete unit of work, such as an API call or data processing step, that can be executed asynchronously. Invoking a task returns a future-like object, which can be awaited to obtain the result or resolved synchronously.

## Entrypoint

An **entrypoint** is a decorator that designates a function as the starting point of a LangGraph workflow. It encapsulates workflow logic and manages execution flow, including handling *long-running tasks* and **interrupts**.

Entrypoints typically include a **checkpointer** to persist workflow state, enabling *resumption* from where it was *paused*.

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

### Executing

The [`@entrypoint`](#entrypoint) yields a [Runnable](https://python.langchain.com/docs/concepts/runnables/) object that can be executed using the 
standard `invoke`, `ainvoke`, `stream`, and `astream` methods.

=== "Invoke"

    ```python
    my_workflow.invoke(some_input)  # Wait for the result synchronously
    ```

=== "Async Invoke"

    ```python
    await my_workflow.ainvoke(some_input)  # Await result asynchronously
    ```

=== "Stream"
    
    ```python
    for chunk in my_workflow.stream(some_input):
        print(chunk)
    ```

=== "Async Stream"

    ```python
    async for chunk in my_workflow.astream(some_input):
        print(chunk)
    ```

### Resuming

When an entrypoint is interrupted, it can be resumed by providing the appropriate command.

=== "Invoke"

    ```python
    from langgraph.types import Command
    
    my_workflow.invoke(Command(resume=some_resume_value))
    ```

=== "Async Invoke"

    ```python
    from langgraph.types import Command
    
    await my_workflow.ainvoke(Command(resume=some_resume_value))
    ```

=== "Stream"

    ```python
    from langgraph.types import Command
    
    for chunk in my_workflow.stream(Command(resume=some_resume_value)):
        print(chunk)
    ```

=== "Async Stream"

    ```python
    from langgraph.types import Command

    async for chunk in my_workflow.astream(Command(resume=some_resume_value)):
        print(chunk)
    ```


## Task

A **task** represents a discrete unit of work, such as an API call or data processing step, that can be executed asynchronously. Invoking a task returns a future, which can be waited on to obtain the result.

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

- To encapsulate any source of non-determinism, such as API calls, database queries, or random number generation. This ensures that the workflow can be **resumed** after being interrupted, which is critical for **human-in-the-loop** interactions.
- Work needs to be retried to handle failures or inconsistencies.
- Parallel execution is beneficial for I/O-bound tasks, allowing multiple operations to run concurrently without blocking (e.g., calling multiple APIs).

## Serialization

The inputs and outputs of `@task` and `@entrypoint` must be JSON-serializable to enable checkpointing and workflow resumption. Supported data types include dictionaries, lists, strings, numbers, and booleans.

Serialization ensures that workflow state, such as task results and intermediate values, can be reliably saved and restored. This is critical for enabling human-in-the-loop interactions, fault tolerance, and parallel execution.

Providing non-serializable inputs or outputs will result in a runtime error when a workflow is configured with a checkpointer.

## Determinism

To utilize features like **human-in-the-loop**, any randomness should be encapsulated inside of tasks. This guarantees that when a workflow is halted (e.g., for human in the loop) and then resumed, it will follow the same **sequence of steps**, even if task results are non-deterministic.

LangGraph achieves this behavior by persisting task and sub-graph results as they execute. A well-designed workflow ensures that resuming execution follows the same sequence of steps, allowing previously computed results to be retrieved correctly without having to re-execute them. This is particularly useful for long-running tasks or tasks with non-deterministic results, as it avoids repeating previously done work and allows resuming from essentially the same 

While different runs of a workflow can produce different results, resuming a **specific** run should always follow the same sequence of recorded steps. This allows LangGraph to efficiently look up task and sub-graph results that were executed prior to the graph being interrupted and avoid recomputing them.

## Idempotency

Idempotency ensures that running the same operation multiple times produces the same
result. This helps prevent duplicate API calls and redundant processing if a step is
rerun due to a failure. Always place API calls inside `@task` functions for
checkpointing, and design them to be idempotent in case of re-execution. Re-execution
can occur if a task completes but its successful execution is not persisted due to a
failure, causing the task to run again when the workflow is resumed. Use idempotency
keys or verify existing results to avoid duplication.

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

=== "Incorrect"

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

=== "Correct"

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


### Calling subgraphs

The functional API and the graph API can be used together in the same application as they share the same underlying runtime.

```python
from langgraph.func import entrypoint

@entrypoint()
def some_workflow(x: int) -> int:
    # Call a graph defined using the graph API
    result_1 = some_graph.invoke(...)
    # Call another graph defined using the graph API
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