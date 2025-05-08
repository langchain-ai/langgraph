# Use the functional API

## Patterns

Below are a few simple patterns that show examples of **how to** use the **Functional API**.

When defining an `entrypoint`, input is restricted to the first argument of the function. To pass multiple inputs, you can use a dictionary.

```python
@entrypoint(checkpointer=checkpointer)
def my_workflow(inputs: dict) -> int:
    value = inputs["value"]
    another_value = inputs["another_value"]
    ...

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

The **Functional API** and the [**Graph API**](../concepts/low_level.md) can be used together in the same application as they share the same underlying runtime.

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

### Call other entrypoints

You can call other **entrypoints** from within an **entrypoint** or a **task**.

```python
@entrypoint() # Will automatically use the checkpointer from the parent entrypoint
def some_other_workflow(inputs: dict) -> int:
    return inputs["value"]

@entrypoint(checkpointer=checkpointer)
def my_workflow(inputs: dict) -> int:
    value = some_other_workflow.invoke({"value": 1})
    return value
```

### Stream custom data

You can stream custom data from an **entrypoint** by using the `StreamWriter` type. This allows you to write custom data to the `custom` stream.

```python
from langgraph.checkpoint.memory import MemorySaver
from langgraph.func import entrypoint, task
from langgraph.types import StreamWriter

@task
def add_one(x):
    return x + 1

@task
def add_two(x):
    return x + 2

checkpointer = MemorySaver()

@entrypoint(checkpointer=checkpointer)
def main(inputs, writer: StreamWriter) -> int:
    """A simple workflow that adds one and two to a number."""
    writer("hello") # Write some data to the `custom` stream
    add_one(inputs['number']).result() # Will write data to the `updates` stream
    writer("world") # Write some more data to the `custom` stream
    add_two(inputs['number']).result() # Will write data to the `updates` stream
    return 5 

config = {
    "configurable": {
        "thread_id": "1"
    }
}

for chunk in main.stream({"number": 1}, stream_mode=["custom", "updates"], config=config):
    print(chunk)
```

```pycon
('updates', {'add_one': 2})
('updates', {'add_two': 3})
('custom', 'hello')
('custom', 'world')
('updates', {'main': 5})
```

!!! important

    The `writer` parameter is automatically injected at run time. It will only be injected if the parameter name appears in the function signature with that *exact* name.


### Retry policy

```python
from langgraph.checkpoint.memory import MemorySaver
from langgraph.func import entrypoint, task
from langgraph.types import RetryPolicy

attempts = 0

# Let's configure the RetryPolicy to retry on ValueError.
# The default RetryPolicy is optimized for retrying specific network errors.
retry_policy = RetryPolicy(retry_on=ValueError)

@task(retry=retry_policy) 
def get_info():
    global attempts
    attempts += 1

    if attempts < 2:
        raise ValueError('Failure')
    return "OK"

checkpointer = MemorySaver()

@entrypoint(checkpointer=checkpointer)
def main(inputs, writer):
    return get_info().result()

config = {
    "configurable": {
        "thread_id": "1"
    }
}

main.invoke({'any_input': 'foobar'}, config=config)
```

```pycon
'OK'
```

### Resuming after an error

```python
import time
from langgraph.checkpoint.memory import MemorySaver
from langgraph.func import entrypoint, task
from langgraph.types import StreamWriter

# This variable is just used for demonstration purposes to simulate a network failure.
# It's not something you will have in your actual code.
attempts = 0

@task()
def get_info():
    """
    Simulates a task that fails once before succeeding.
    Raises an exception on the first attempt, then returns "OK" on subsequent tries.
    """
    global attempts
    attempts += 1

    if attempts < 2:
        raise ValueError("Failure")  # Simulate a failure on the first attempt
    return "OK"

# Initialize an in-memory checkpointer for persistence
checkpointer = MemorySaver()

@task
def slow_task():
    """
    Simulates a slow-running task by introducing a 1-second delay.
    """
    time.sleep(1)
    return "Ran slow task."

@entrypoint(checkpointer=checkpointer)
def main(inputs, writer: StreamWriter):
    """
    Main workflow function that runs the slow_task and get_info tasks sequentially.

    Parameters:
    - inputs: Dictionary containing workflow input values.
    - writer: StreamWriter for streaming custom data.

    The workflow first executes `slow_task` and then attempts to execute `get_info`,
    which will fail on the first invocation.
    """
    slow_task_result = slow_task().result()  # Blocking call to slow_task
    get_info().result()  # Exception will be raised here on the first attempt
    return slow_task_result

# Workflow execution configuration with a unique thread identifier
config = {
    "configurable": {
        "thread_id": "1"  # Unique identifier to track workflow execution
    }
}

# This invocation will take ~1 second due to the slow_task execution
try:
    # First invocation will raise an exception due to the `get_info` task failing
    main.invoke({'any_input': 'foobar'}, config=config)
except ValueError:
    pass  # Handle the failure gracefully
```

When we resume execution, we won't need to re-run the `slow_task` as its result is already saved in the checkpoint.

```python
main.invoke(None, config=config)
```

```pycon
'Ran slow task.'
```

### Human-in-the-loop

The functional API supports [human-in-the-loop](../concepts/human_in_the_loop.md) workflows using the `interrupt` function and the `Command` primitive.

Please see the following examples for more details:

* [How to wait for user input (Functional API)](./wait-user-input-functional.ipynb): Shows how to implement a simple human-in-the-loop workflow using the functional API.
* [How to review tool calls (Functional API)](./review-tool-calls-functional.ipynb): Guide demonstrates how to implement human-in-the-loop workflows in a ReAct agent using the LangGraph Functional API.

### Short-term memory

[State management](#) using the **previous** parameter and optionally using the `entrypoint.final` primitive can be used to implement [short term memory](../concepts/memory.md).

Please see the following how-to guides for more details:

*  [How to add thread-level persistence (functional API)](./persistence-functional.ipynb): Shows how to add thread-level persistence to a functional API workflow and implements a simple chatbot.

### Long-term memory

[long-term memory](../concepts/memory.md#long-term-memory) allows storing information across different **thread ids**. This could be useful for learning information
about a given user in one conversation and using it in another.

Please see the following how-to guides for more details:

* [How to add cross-thread persistence (functional API)](./cross-thread-persistence-functional.ipynb): Shows how to add cross-thread persistence to a functional API workflow and implements a simple chatbot.

### Workflows

* [Workflows and agent](../tutorials/workflows.md) guide for more examples of how to build workflows using the Functional API.

### Agents

* [How to create an agent from scratch (Functional API)](./react-agent-from-scratch-functional.ipynb): Shows how to create a simple agent from scratch using the functional API.
* [How to build a multi-agent network](./multi-agent-network-functional.ipynb): Shows how to build a multi-agent network using the functional API.
* [How to add multi-turn conversation in a multi-agent application (functional API)](./multi-agent-multi-turn-convo-functional.ipynb): allow an end-user to engage in a multi-turn conversation with one or more agents.  

### Integrate with other libraries

* [Add LangGraph's features to other frameworks using the functional API](./autogen-integration-functional.ipynb): Add LangGraph features like persistence, memory and streaming to other agent frameworks that do not provide them out of the box.