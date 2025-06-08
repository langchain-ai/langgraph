# Use the functional API

## Creating a simple workflow

When defining an `entrypoint`, input is restricted to the first argument of the function. To pass multiple inputs, you can use a dictionary.

```python
@entrypoint(checkpointer=checkpointer)
def my_workflow(inputs: dict) -> int:
    value = inputs["value"]
    another_value = inputs["another_value"]
    ...

my_workflow.invoke({"value": 1, "another_value": 2})  
```

??? example "Extended example: simple workflow" 

    ```python
    import uuid
    from langgraph.func import entrypoint, task
    from langgraph.checkpoint.memory import MemorySaver

    # Task that checks if a number is even
    @task
    def is_even(number: int) -> bool:
        return number % 2 == 0

    # Task that formats a message
    @task
    def format_message(is_even: bool) -> str:
        return "The number is even." if is_even else "The number is odd."

    # Create a checkpointer for persistence
    checkpointer = MemorySaver()

    @entrypoint(checkpointer=checkpointer)
    def workflow(inputs: dict) -> str:
        """Simple workflow to classify a number."""
        even = is_even(inputs["number"]).result()
        return format_message(even).result()

    # Run the workflow with a unique thread ID
    config = {"configurable": {"thread_id": str(uuid.uuid4())}}
    result = workflow.invoke({"number": 7}, config=config)
    print(result)
    ```

??? example "Extended example: Compose an essay with an LLM"

    This example demonstrates how to use the `@task` and `@entrypoint` decorators
    syntactically. Given that a checkpointer is provided, the workflow results will
    be persisted in the checkpointer.

    ```python
    import uuid
    from langchain.chat_models import init_chat_model
    from langgraph.func import entrypoint, task
    from langgraph.checkpoint.memory import MemorySaver

    llm = init_chat_model('openai:gpt-3.5-turbo')

    # Task: generate essay using an LLM
    @task
    def compose_essay(topic: str) -> str:
        """Generate an essay about the given topic."""
        return llm.invoke([
            {"role": "system", "content": "You are a helpful assistant that writes essays."},
            {"role": "user", "content": f"Write an essay about {topic}."}
        ]).content

    # Create a checkpointer for persistence
    checkpointer = MemorySaver()

    @entrypoint(checkpointer=checkpointer)
    def workflow(topic: str) -> str:
        """Simple workflow that generates an essay with an LLM."""
        return compose_essay(topic).result()

    # Execute the workflow
    config = {"configurable": {"thread_id": str(uuid.uuid4())}}
    result = workflow.invoke("the history of flight", config=config)
    print(result)
    ```

## Parallel execution

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


??? example "Extended example: parallel LLM calls"

    This example demonstrates how to run multiple LLM calls in parallel using `@task`. Each call generates a paragraph on a different topic, and results are joined into a single text output.

    ```python
    import uuid
    from langchain.chat_models import init_chat_model
    from langgraph.func import entrypoint, task
    from langgraph.checkpoint.memory import MemorySaver

    # Initialize the LLM model
    llm = init_chat_model("openai:gpt-3.5-turbo")

    # Task that generates a paragraph about a given topic
    @task
    def generate_paragraph(topic: str) -> str:
        response = llm.invoke([
            {"role": "system", "content": "You are a helpful assistant that writes educational paragraphs."},
            {"role": "user", "content": f"Write a paragraph about {topic}."}
        ])
        return response.content

    # Create a checkpointer for persistence
    checkpointer = MemorySaver()

    @entrypoint(checkpointer=checkpointer)
    def workflow(topics: list[str]) -> str:
        """Generates multiple paragraphs in parallel and combines them."""
        futures = [generate_paragraph(topic) for topic in topics]
        paragraphs = [f.result() for f in futures]
        return "\n\n".join(paragraphs)

    # Run the workflow
    config = {"configurable": {"thread_id": str(uuid.uuid4())}}
    result = workflow.invoke(["quantum computing", "climate change", "history of aviation"], config=config)
    print(result)
    ```

    This example uses LangGraph's concurrency model to improve execution time, especially when tasks involve I/O like LLM completions.

## Calling graphs 

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

??? example "Extended example: calling a simple graph from the functional API"

    ```python
    import uuid
    from typing import TypedDict
    from langgraph.func import entrypoint
    from langgraph.checkpoint.memory import MemorySaver
    from langgraph.graph import StateGraph

    # Define the shared state type
    class State(TypedDict):
        foo: int

    # Define a simple transformation node
    def double(state: State) -> State:
        return {"foo": state["foo"] * 2}

    # Build the graph using the Graph API
    builder = StateGraph(State)
    builder.add_node("double", double)
    builder.set_entry_point("double")
    graph = builder.compile()

    # Define the functional API workflow
    checkpointer = MemorySaver()

    @entrypoint(checkpointer=checkpointer)
    def workflow(x: int) -> dict:
        result = graph.invoke({"foo": x})
        return {"bar": result["foo"]}

    # Execute the workflow
    config = {"configurable": {"thread_id": str(uuid.uuid4())}}
    print(workflow.invoke(5, config=config))  # Output: {'bar': 10}
    ```


## Call other entrypoints

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

??? example "Extended example: calling another entrypoint"

    ```python
    import uuid
    from langgraph.func import entrypoint
    from langgraph.checkpoint.memory import MemorySaver

    # Initialize a checkpointer
    checkpointer = MemorySaver()

    # A reusable sub-workflow that multiplies a number
    @entrypoint()
    def multiply(inputs: dict) -> int:
        return inputs["a"] * inputs["b"]

    # Main workflow that invokes the sub-workflow
    @entrypoint(checkpointer=checkpointer)
    def main(inputs: dict) -> dict:
        result = multiply.invoke({"a": inputs["x"], "b": inputs["y"]})
        return {"product": result}

    # Execute the main workflow
    config = {"configurable": {"thread_id": str(uuid.uuid4())}}
    print(main.invoke({"x": 6, "y": 7}, config=config))  # Output: {'product': 42}
    ```
        

## Streaming

The **Functional API** uses the same streaming mechanism as the **Graph API**. Please
read the [**streaming guide**](../concepts/streaming.md) section for more details.

Example of using the streaming API to stream both updates and custom data.

```python
from langgraph.func import entrypoint
from langgraph.checkpoint.memory import MemorySaver
from langgraph.config import get_stream_writer # (1)!

checkpointer = MemorySaver()

@entrypoint(checkpointer=checkpointer)
def main(inputs: dict) -> int:
    writer = get_stream_writer() # (2)!
    writer("Started processing") # (3)!
    result = inputs["x"] * 2
    writer(f"Result is {result}") # (4)!
    return result

config = {"configurable": {"thread_id": "abc"}}

# highlight-next-line
for mode, chunk in main.stream( # (5)!
    {"x": 5},
    stream_mode=["custom", "updates"], # (6)!
    config=config
):
    print(f"{mode}: {chunk}")
```

1. Import `get_stream_writer` from `langgraph.config`.
2. Obtain a stream writer instance within the entrypoint.
3. Emit custom data before computation begins.
4. Emit another custom message after computing the result.
5. Use `.stream()` to process streamed output.
6. Specify which streaming modes to use.

```pycon
('updates', {'add_one': 2})
('updates', {'add_two': 3})
('custom', 'hello')
('custom', 'world')
('updates', {'main': 5})
```



!!! important "Async with Python < 3.11"

    If using Python < 3.11 and writing async code, using `get_stream_writer()` will not work. Instead please 
    use the `StreamWriter` class directly. See [Async with Python < 3.11](../how-tos/streaming.md#async) for more details.

    ```python
    from langgraph.types import StreamWriter

    @entrypoint(checkpointer=checkpointer)
    # highlight-next-line
    async def main(inputs: dict, writer: StreamWriter) -> int:
        ...
    ```

## Retry policy

```python
from langgraph.checkpoint.memory import MemorySaver
from langgraph.func import entrypoint, task
from langgraph.types import RetryPolicy

# This variable is just used for demonstration purposes to simulate a network failure.
# It's not something you will have in your actual code.
attempts = 0

# Let's configure the RetryPolicy to retry on ValueError.
# The default RetryPolicy is optimized for retrying specific network errors.
retry_policy = RetryPolicy(retry_on=ValueError)

@task(retry_policy=retry_policy) 
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

## Caching Tasks

```python
import time
from langgraph.cache.memory import InMemoryCache
from langgraph.func import entrypoint, task
from langgraph.types import CachePolicy


@task(cache_policy=CachePolicy(ttl=120))  # (1)!
def slow_add(x: int) -> int:
    time.sleep(1)
    return x * 2


@entrypoint(cache=InMemoryCache())
def main(inputs: dict) -> dict[str, int]:
    result1 = slow_add(inputs["x"]).result()
    result2 = slow_add(inputs["x"]).result()
    return {"result1": result1, "result2": result2}


for chunk in main.stream({"x": 5}, stream_mode="updates"):
    print(chunk)

#> {'slow_add': 10}
#> {'slow_add': 10, '__metadata__': {'cached': True}}
#> {'main': {'result1': 10, 'result2': 10}}
```

1. `ttl` is specified in seconds. The cache will be invalidated after this time.

## Resuming after an error

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

## Human-in-the-loop

The functional API supports [human-in-the-loop](../concepts/human_in_the_loop.md) workflows using the `interrupt` function and the `Command` primitive.

Please see the following examples for more details:

* [How to wait for user input (Functional API)](./wait-user-input-functional.ipynb): Shows how to implement a simple human-in-the-loop workflow using the functional API.
* [How to review tool calls (Functional API)](./review-tool-calls-functional.ipynb): Guide demonstrates how to implement human-in-the-loop workflows in a ReAct agent using the LangGraph Functional API.

## Short-term memory

Short-term memory allows storing information across different **invocations** of the same **thread id**. See [short-term memory](../concepts/functional_api.md#short-term-memory) for more details.

### Decouple return value from saved value

Use `entrypoint.final` to decouple what is returned to the caller from what is persisted in the checkpoint. This is useful when:

* You want to return a computed result (e.g., a summary or status), but save a different internal value for use on the next invocation.
* You need to control what gets passed to the previous parameter on the next run.

```python
from typing import Optional
from langgraph.func import entrypoint
from langgraph.checkpoint.memory import MemorySaver

checkpointer = MemorySaver()

@entrypoint(checkpointer=checkpointer)
def accumulate(n: int, *, previous: Optional[int]) -> entrypoint.final[int, int]:
    previous = previous or 0
    total = previous + n
    # Return the *previous* value to the caller but save the *new* total to the checkpoint.
    return entrypoint.final(value=previous, save=total)

config = {"configurable": {"thread_id": "my-thread"}}

print(accumulate.invoke(1, config=config))  # 0
print(accumulate.invoke(2, config=config))  # 1
print(accumulate.invoke(3, config=config))  # 3
```

### Chatbot example

An example of a simple chatbot using the functional API and the `MemorySaver` checkpointer.
The bot is able to remember the previous conversation and continue from where it left off.

```python
from langchain_core.messages import BaseMessage
from langgraph.graph import add_messages
from langgraph.func import entrypoint, task
from langgraph.checkpoint.memory import MemorySaver
from langchain_anthropic import ChatAnthropic

model = ChatAnthropic(model="claude-3-5-sonnet-latest")

@task
def call_model(messages: list[BaseMessage]):
    response = model.invoke(messages)
    return response

checkpointer = MemorySaver()

@entrypoint(checkpointer=checkpointer)
def workflow(inputs: list[BaseMessage], *, previous: list[BaseMessage]):
    if previous:
        inputs = add_messages(previous, inputs)

    response = call_model(inputs).result()
    return entrypoint.final(value=response, save=add_messages(inputs, response))

config = {"configurable": {"thread_id": "1"}}
input_message = {"role": "user", "content": "hi! I'm bob"}
for chunk in workflow.stream([input_message], config, stream_mode="values"):
    chunk.pretty_print()

input_message = {"role": "user", "content": "what's my name?"}
for chunk in workflow.stream([input_message], config, stream_mode="values"):
    chunk.pretty_print()
```

??? example "Extended example: build a simple chatbot"

     [How to add thread-level persistence (functional API)](./persistence-functional.ipynb): Shows how to add thread-level persistence to a functional API workflow and implements a simple chatbot.

## Long-term memory

[long-term memory](../concepts/memory.md#long-term-memory) allows storing information across different **thread ids**. This could be useful for learning information about a given user in one conversation and using it in another.


??? example "Extended example: add long-term memory"

    [How to add cross-thread persistence (functional API)](./cross-thread-persistence-functional.ipynb): Shows how to add cross-thread persistence to a functional API workflow and implements a simple chatbot.

## Workflows

* [Workflows and agent](../tutorials/workflows.md) guide for more examples of how to build workflows using the Functional API.

## Agents

* [How to create an agent from scratch (Functional API)](./react-agent-from-scratch-functional.ipynb): Shows how to create a simple agent from scratch using the functional API.
* [How to build a multi-agent network](./multi-agent-network-functional.ipynb): Shows how to build a multi-agent network using the functional API.
* [How to add multi-turn conversation in a multi-agent application (functional API)](./multi-agent-multi-turn-convo-functional.ipynb): allow an end-user to engage in a multi-turn conversation with one or more agents.  

## Integrate with other libraries

* [Add LangGraph's features to other frameworks using the functional API](./autogen-integration-functional.ipynb): Add LangGraph features like persistence, memory and streaming to other agent frameworks that do not provide them out of the box.