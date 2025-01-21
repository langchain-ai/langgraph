# Functional API

!!! warning "Beta"
    The Functional API is currently in **beta** and is subject to change. Please [report any issues](https://github.com/langchain-ai/langgraph/issues) or feedback to the LangGraph team. Use the label `#functional-api` when creating issues in Github.

## Overview

LangGraph's **Functional API** allows using standard control flow constructs (e.g., loops, conditionals) to define complex workflows without having to define a graph.

The **Functional API** runs on the same LangGraph runtime as the [**Graph API**](./low_level.md), providing access to features like [persistence](persistence.md), [human-in-the-loop](human_in_the_loop.md), and [streaming](streaming.md).

The **Functional API** and the **Graph API** can be used together in the same application, allowing you to intermix the two paradigms if needed.

## Example

Below we demonstrate a simple application that writes an essay and [interrupts](human_in_the_loop.md) to request human review.
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

??? example "Detailed Explanation"

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

The **Functional API** provides two primitives for building workflows:

- **[Entrypoint](#entrypoint)**: An **entrypoint** is a decorator that designates a function as the starting point of a LangGraph workflow. It encapsulates workflow logic and manages execution flow, including handling *long-running tasks* and [interrupts](human_in_the_loop.md).
- **[Task](#task)**: Represents a discrete unit of work, such as an API call or data processing step, that can be executed asynchronously from within an **entrypoint**. Invoking a **task** returns a future-like object, which can be awaited to obtain the result or resolved synchronously.

## Entrypoint

The `@entrypoint` decorator can be used to create a LangGraph workflow from a function. It encapsulates workflow logic and manages execution flow, including handling *long-running tasks* and [interrupts](./low_level.md#interrupt)

Entrypoints typically include a **checkpointer** to persist workflow state, enabling *resumption* from where it was *interrupted*.

### Definition

An entrypoint is defined by decorating a function with the `@entrypoint` decorator. The function should accept a **single** input argument. If you need to pass multiple arguments, you can use a dictionary.

You will usually want to pass a **checkpointer** to the `@entrypoint` decorator to enable persistence and use features like **human-in-the-loop**.

```python
from langgraph.func import entrypoint

@entrypoint(checkpointer=checkpointer)
def my_workflow(some_input: dict) -> int:
    # some logic that may involve long-running tasks like API calls,
    # and may be interrupted for human-in-the-loop.
    ...
    return result
```

### Executing

Using the [`@entrypoint`](#entrypoint) yields a [EntrypointPregel][langgraph.func.EntrypointPregel] object that can be executed using the `invoke`, `ainvoke`, `stream`, and `astream` methods.

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

Execution can be **resumed** using the [Command][langgraph.types.Command] primitive.

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

A **task** represents a discrete unit of work, such as an API call or data processing step, that can be executed asynchronously. Invoking a **task** returns a future, which can be waited on to obtain the result.

### Definition

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

    The outputs of tasks must be JSON-serializable to support checkpointing.

### Execution

**Tasks** can only be called from within an **entrypoint**, another **task**, or a [state graph node](./low_level.md#nodes). They **cannot** be called directly from the main application code. Calling a **task** produces a future-like object that can be awaited or resolved to obtain the result.

=== "Synchronous Invocation"

    ```python
    @entrypoint(checkpointer=checkpointer)
    def my_workflow(some_input: int) -> int:
        future = slow_computation(some_input)
        return future.result()  # Wait for the result synchronously
    ```

=== "Asynchronous Invocation"

    ```python
    @entrypoint(checkpointer=checkpointer)
    async def my_workflow(some_input: int) -> int:
        return await slow_computation(some_input)  # Await result asynchronously
    ```

## When to use a task

**Tasks** are useful in the following scenarios:

- **Resumable Graph Execution**: When graph execution may need to be **resumed** after being **interrupted** (e.g., for **human-in-the-loop**), **tasks** can encapsulate any source of non-determinism, such as API calls, database queries, or random number generation. See the [determinism](#determinism) for more details.
- **Retriable Work**: When work needs to be retried to handle failures or inconsistencies, **tasks** provide a way to encapsulate and manage the retry logic.
- **Parallel Execution**: For I/O-bound tasks, **tasks** enable parallel execution, allowing multiple operations to run concurrently without blocking (e.g., calling multiple APIs).
 
## Patterns

Below are a few simple patterns that show examples of how to use the **Functional API**.

### Multiple inputs

When defining an `entrypoint`, input is restricted to the first argument of the function. To pass multiple inputs, you can use a dictionary.

```python
@entrypoint(checkpointer=checkpointer)
def my_workflow(inputs: dict) -> int:
    value = inputs["value"]
    another_value = inputs["another_value"]
    return value

my_workflow.invoke({"value": 1, "another_value": 2})  
```

### Parallel execution

Tasks can be executed in parallel by invoking them concurrently and waiting for the results. This is useful for improving performance in IO bound tasks (e.g., calling APIs for LLMs).

```python
@task
def add_one(number: int) -> int:
    return number + 1

@entrypoint(checkpointer=checkpointer)
def graph(numbers: list[int]) -> list[str]:
    futures = [add_one(i) for i in numbers]
    return [f.result() for f in futures]
```

### Calling subgraphs

The **Functional API** and the [**Graph API**](./low_level.md) can be used together in the same application as they share the same underlying runtime.

```python
from langgraph.func import entrypoint
from langgraph.graph import StateGraph

builder = StateGraph()
...
some_graph = builder.compile()

@entrypoint()
def some_workflow(some_input: dict) -> int:
    # Call a graph defined using the graph API
    result_1 = some_graph.invoke(...)
    # Call another graph defined using the graph API
    result_2 = another_graph.invoke(...)
    return {
        "result_1": result_1,
        "result_2": result_2
    }
```

## Serialization

There are two key aspects to serialization in LangGraph:

1. `@entrypoint` inputs and outputs must be JSON-serializable.
2. `@task` outputs must be JSON-serializable.

These requirements are necessary for enabling checkpointing and workflow resumption. Use python primitives
like dictionaries, lists, strings, numbers, and booleans to ensure that your inputs and outputs are serializable.

Serialization ensures that workflow state, such as task results and intermediate values, can be reliably saved and restored. This is critical for enabling human-in-the-loop interactions, fault tolerance, and parallel execution.

Providing non-serializable inputs or outputs will result in a runtime error when a workflow is configured with a checkpointer.

## Determinism

To utilize features like **human-in-the-loop**, any randomness should be encapsulated inside of **tasks**. This guarantees that when execution is halted (e.g., for human in the loop) and then resumed, it will follow the same *sequence of steps*, even if **task** results are non-deterministic.

LangGraph achieves this behavior by persisting **task** and [**subgraph**](./low_level.md#subgraphs) results as they execute. A well-designed workflow ensures that resuming execution follows the *same sequence of steps*, allowing previously computed results to be retrieved correctly without having to re-execute them. This is particularly useful for long-running **tasks** or **tasks** with non-deterministic results, as it avoids repeating previously done work and allows resuming from essentially the same 

While different runs of a workflow can produce different results, resuming a **specific** run should always follow the same sequence of recorded steps. This allows LangGraph to efficiently look up **task** and **subgraph** results that were executed prior to the graph being interrupted and avoid recomputing them.

## Idempotency

Idempotency ensures that running the same operation multiple times produces the same result. This helps prevent duplicate API calls and redundant processing if a step is rerun due to a failure. Always place API calls inside **tasks** functions for checkpointing, and design them to be idempotent in case of re-execution. Re-execution can occur if a **task** starts, but does not complete successfully. Then, if the workflow is resumed, the **task** will run again. Use idempotency keys or verify existing results to avoid duplication.


## Functional API vs. Graph API

The **Functional API** and the **Graph APIs** provide two different paradigms to create workflows in LangGraph. Here are some key differences:

- **Control flow**: The Functional API does not require thinking about graph structure. You can use standard Python constructs to define workflows. Please note that you can as of Lan
- **State management**: The GraphAPI requires declaring a [**GraphState**](./low_level.md#state) and may require defining [**reducers**](./low_level.md#reducers) to manage updates to the graph state.
- **Visualization**: The Graph API makes it easy to visualize the workflow as a graph which can be useful for debugging, understanding the workflow, and sharing with others.

## Common Pitfalls

### Handling side effects

Side effects, such as writing to a file or sending an email, should be encapsulated in tasks to ensure consistent execution upon resumption.

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

### Non-deterministic control flow

Non-deterministic control flow can lead to inconsistent results when resuming a workflow. To ensure correct behavior, encapsulate non-deterministic operations (e.g., random number generation, time-based logic) inside **tasks**.

=== "Incorrect"

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

=== "Correct"

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

Please see the [determinism](#determinism) section for more details.
