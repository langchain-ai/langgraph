# Durable Execution

**Durable execution** is a technique in which a process or workflow saves its progress at key points, allowing it to pause and later resume exactly where it left off. This is particularly useful in scenarios that require [human-in-the-loop](./human_in_the_loop.md), where users can inspect, validate, or modify the process before continuing, and in long-running tasks that might encounter interruptions or errors (e.g., calls to an LLM timing out). By preserving completed work, durable execution enables a process to resume without reprocessing previous steps—even after a significant delay (e.g., a week later). 

LangGraph's built-in [persistence](./persistence.md) layer provides durable execution for workflows, ensuring that the state of each execution step is saved to a durable store. This capability guarantees that if a workflow is interrupted -- whether by a system failure or for [human-in-the-loop](./human_in_the_loop.md) interactions -- it can be resumed from its last recorded state.

!!! tip

    If you are using LangGraph with a checkpointer, you already have durable execution enabled. You can pause and resume workflows at any point, even after interruptions or failures.
    To make the most of durable execution, ensure that your workflow is designed to be [deterministic](#determinism-and-consistent-replay) and [idempotent](#idempotency) and warp any side effects or non-deterministic operations inside [tasks](./functional_api.md#task). You can use [tasks](./functional_api.md#task) from both the [Graph API](./low_level.md) and the [Functional API](./functional_api.md).

## Requirements

To leverage durable execution in LangGraph, you need to:

1. Enable [persistence](./persistence.md) in your workflow by specifying a [checkpointer](./persistence.md#checkpointer-libraries) that will save workflow progress.
2. Specify a [thread identifier](./persistence.md#threads) when executing a workflow. This will track the execution history for a particular instance of the workflow.
3. Wrap any non-deterministic operations (e.g., random number generation) or operations with side effects (e.g., file writes, API calls) inside [tasks][langgraph.func.task] to ensure that when a workflow is resumed, these operations are not repeated for the particular run, and instead their results are retrieved from the persistence layer. For more information, see [Determinism and Consistent Replay](#determinism-and-consistent-replay).

## Determinism and Consistent Replay

When you resume a workflow run, the code does **NOT** resume from the **same line of code** where execution stopped; instead, it will identify an appropriate [starting point](#starting-points-for-resuming-workflows) from which to pick up where it left off. This means that the workflow will replay all steps from the [starting point](#starting-points-for-resuming-workflows) until it reaches the point where it was stopped.

As a result, when you are writing a workflow for durable execution, you must wrap any non-deterministic operations (e.g., random number generation) and any operations with side effects (e.g., file writes, API calls) inside [tasks](./functional_api.md#task) or [nodes](./low_level.md#nodes).

!!! tip "Using Tasks with StateGraph API"

    If a node contains multiple operations, it is easier to convert each operation into a **task** rather than refactor the operations into individual nodes. See the [example below](#__tabbed_1_2) for more details.

- **Avoid Repeating Work**:  Place operations that produce side effects (such as logging, file writes, or network calls) after an interrupt in StateGraph (Graph API) **nodes** or encapsulate them within [tasks](./functional_api.md#task). This prevents their unintended repetition when the workflow is resumed. LangGraph will look up information stored in the persistence layer previously executed **nodes** or **tasks** for the specific run and will swap in the results of those tasks instead of re-executing them.
- **Encapsulate Non-Deterministic Operations:**  Wrap any code that might yield non-deterministic results (e.g., random number generation) inside **tasks** or **nodes**. This ensures that, upon resumption, the workflow follows the exact recorded sequence of steps with the same outcomes.


??? example "Using Tasks in a StateGraph (Graph API)"

    If a node contains multiple operations, it is easier to wrap each operation in a **task** rather than separate each operation into its own node. See
    example below for more details.

    === "Original"

        ```python
        from typing import NotRequired
        from typing_extensions import TypedDict
        import uuid

        from langgraph.graph import StateGraph, START, END
        from langgraph.checkpoint.memory import MemorySaver
        import requests

        # Define a TypedDict to represent the state
        class State(TypedDict):
            url: str
            result: NotRequired[str]

        def call_api(state: State):
            """Example node that makes an API request."""
            # highlight-next-line
            result = requests.get(state['url']).text[:100]  # Side-effect
            return {
                "result": result
            }

        # Create a StateGraph builder and add a node for the call_api function
        builder = StateGraph(State)
        builder.add_node("call_api", call_api)

        # Connect the start and end nodes to the call_api node
        builder.add_edge(START, "call_api")
        builder.add_edge("call_api", END)

        # Specify a checkpointer
        checkpointer = MemorySaver()

        # Compile the graph with the checkpointer
        graph = builder.compile(checkpointer=checkpointer)

        # Define a config with a thread ID.
        thread_id = uuid.uuid4()
        config = {"configurable": {"thread_id": thread_id}}

        # Invoke the graph
        graph.invoke({"url": "https://www.example.com"}, config)
        ```

    === "With task"

        ```python
        from typing import NotRequired
        from typing_extensions import TypedDict
        import uuid

        from langgraph.graph import StateGraph, START, END
        from langgraph.func import task
        import requests

        # Define a TypedDict to represent the state
        class State(TypedDict):
            urls: list[str]
            result: NotRequired[list[str]]


        @task
        def _make_request(url: str):
            """Make a request."""
            # highlight-next-line
            return requests.get(url).text[:100]

        def call_api(state: State):
            """Example node that makes an API request."""
            # highlight-next-line
            requests = [_make_request(url) for url in state['urls']]
            results = [request.result() for request in requests]
            return {
                "results": results
            }

        # Create a StateGraph builder and add a node for the call_api function
        builder = StateGraph(State)
        builder.add_node("call_api", call_api)

        # Connect the start and end nodes to the call_api node
        builder.add_edge(START, "call_api")
        builder.add_edge("call_api", END)

        # Specify a checkpointer
        checkpointer = MemorySaver()

        # Compile the graph with the checkpointer
        graph = builder.compile(checkpointer=checkpointer)

        # Define a config with a thread ID.
        thread_id = uuid.uuid4()
        config = {"configurable": {"thread_id": thread_id}}

        # Invoke the graph
        graph.invoke({"urls": ["https://www.example.com"]}, config)
        ```

For some examples of pitfalls to avoid, see the [Common Pitfalls](./functional_api.md#common-pitfalls) section in the functional API, which shows
how to structure your code using **tasks** to avoid these issues. The same principles apply to the [StateGraph (Graph API)][langgraph.graph.state.StateGraph].

## Idempotency

Idempotency ensures that running the same operation multiple times produces the same result. This helps prevent duplicate API calls and redundant processing if a step is rerun due to a failure. Always place API calls inside **tasks** functions for checkpointing, and design them to be idempotent in case of re-execution. Re-execution can occur if a **task** starts, but does not complete successfully. Then, if the workflow is resumed, the **task** will run again. Use idempotency keys or verify existing results to avoid duplication.

## Resuming Workflows

Once you have enabled durable execution in your workflow, you can resume execution for the following scenarios:

- **Pausing and Resuming Workflows:** Use the [interrupt][langgraph.types.interrupt] function to pause a workflow at specific points and the [Command][langgraph.types.Command] primitive to resume it with updated state. See [**Human-in-the-Loop**](./human_in_the_loop.md) for more details.
- **Recovering from Failures:** Automatically resume workflows from the last successful checkpoint after an exception (e.g., LLM provider outage). This involves executing the workflow with the same thread identifier by providing it with a `None` as the input value (see this [example](./functional_api.md#resuming-after-an-error) with the functional API).

## Starting Points for Resuming Workflows

* If you're using a [StateGraph (Graph API)][langgraph.graph.state.StateGraph], the starting point is the beginning of the [**node**](./low_level.md#nodes) where execution stopped. 
* If you're making a subgraph call inside a node, the starting point will be the **parent** node that called the subgraph that was halted.
Inside the subgraph, the starting point will be the specific [**node**](./low_level.md#nodes) where execution stopped.
* If you're using the Functional API, the starting point is the beginning of the [**entrypoint**](./functional_api.md#entrypoint) where execution stopped.