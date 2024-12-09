# Breakpoints

Breakpoints pause graph execution at specific points, enabling [**human-in-the-loop**](./human_in_the_loop.md) workflows and debugging. Breakpoints are powered by  LangGraph's [**persistence layer**](./persistence.md), which saves the state after each graph step. 

## Requirements

To use breakpoints, you will need to:

1. [**Specify a checkpointer**](persistence.md#checkpoints) to save the graph state after each step.
2. [**Set breakpoints**](#setting-breakpoints) to specify where execution should pause.
3. Run the graph with a [**thread ID**](./persistence.md#threads) to pause execution at the breakpoint.
4. [**Resume execution**](#resuming) from the paused state.

## Setting breakpoints

There are two places where you can set breakpoints:

1. **Inside** a node using the [`interrupt` function](#the-interrupt-function) (or the older [`NodeInterrupt` exception](#nodeinterrupt-exception)).
2. **Before** or **after** a node executes by setting breakpoints at **compile time** or **run time**. We call these [**static breakpoints**](#static-breakpoints).
 
The **recommended** way to set breakpoints is using the [`interrupt` function](#the-interrupt-function). This method is easier to use and more flexible than the older methods.

### The `interrupt` function

Use the [interrupt](../reference/types.md/#langgraph.types.interrupt) function to **pause** the graph at specific points to collect user input. The `interrupt` function surfaces interrupt information to the client, allowing the developer to collect user input, validate the graph state, or make decisions before resuming execution.

```python
from langgraph.types import interrupt

def human_approval(state: State):
    ...
    answer = interrupt(
        # Interrupt information to surface to the client.
        # Can be any JSON serializable value.
        {
            "question": "Can we proceed?",
            "llm_output": state["llm_output"]
        }
    )
    
    if answer['approved']:
        # Proceed with the action
        ...
    else:
        # Do something else
    ...


# Add the node to the graph
graph_builder.add_node("human_approval", human_approval)
# Compile the graph with a checkpointer
graph = graph_builder.compile(checkpointer=checkpointer)

# Run the graph until the breakpoint
thread_config = {"configurable": {"thread_id": "some_id"}}
for event in graph.stream(inputs, thread_config, stream_mode="values"):
    print(event)
```

```pycon
{'__interrupt__': (
        Interrupt(
            value={'question': 'Can we proceed?', "llm_output": "..."}, 
            resumable=True, 
            ns=['node:5df255f7-d683-1a99-b7c8-00dd534aed8e'], 
            when='during'
        ),
    )
}
```

Graph execution can be resumed using the [Command](../reference/types.md#langgraph.types.Command) primitive. The `Command` primitive provides several options to control and modify the graph's state during resumption:

```python
# Resume the graph with the user's input
for event in graph.stream(Command(resume={"approved": True}), config=thread_config):
    print(event)
```

### Static breakpoints

Static breakpoints are triggered either **before** or **after** a node executes. You can set static breakpoints by specifying `interrupt_before` and `interrupt_after` at **"compile" time** or **run time**.

=== "Compile time"

    ```python
    graph = graph_builder.compile(
        interrupt_before=["node_a"], 
        interrupt_after=["node_b", "node_c"],
        checkpointer=..., # Specify a checkpointer
    )

    thread_config = {
        "configurable": {
            "thread_id": "some_thread"
        }
    }

    # Run the graph until the breakpoint
    graph.invoke(inputs, config=thread_config)

    # Optionally update the graph state based on user input
    graph.update_state(update, config=thread_config)

    # Resume the graph
    graph.invoke(None, config=thread_config)
    ```

=== "Run time"

    ```python
    graph.invoke(
        inputs, 
        config={"configurable": {"thread_id": "some_thread"}}, 
        interrupt_before=["node_a"], 
        interrupt_after=["node_b", "node_c"]
    )

    thread_config = {
        "configurable": {
            "thread_id": "some_thread"
        }
    }

    # Run the graph until the breakpoint
    graph.invoke(inputs, config=thread_config)

    # Optionally update the graph state based on user input
    graph.update_state(update, config=thread_config)

    # Resume the graph
    graph.invoke(None, config=thread_config)
    ```

    !!! note

        You cannot set static breakpoints at runtime for **sub-graphs**.
        If you have a sub-graph, you must set the breakpoints at compilation time.

Static breakpoints can be especially useful for debugging if you want to step through the graph execution one
node at a time or if you want to pause the graph execution at specific nodes.

### `NodeInterrupt` exception

We recommend that you [**use the `interrupt` function instead**](#the-interrupt-function) of the `NodeInterrupt` exception. The `interrupt` function is easier to use and more flexible.

??? node "`NodeInterrupt` exception"

    The developer can define some *condition* that must be met for a breakpoint to be triggered. This concept of [dynamic breakpoints](./low_level.md#dynamic-breakpoints) is useful when the developer wants to halt the graph under *a particular condition*. This uses a `NodeInterrupt`, which is a special type of exception that can be raised from within a node based upon some condition. As an example, we can define a dynamic breakpoint that triggers when the `input` is longer than 5 characters.

    ```python
    def my_node(state: State) -> State:
        if len(state['input']) > 5:
            raise NodeInterrupt(f"Received input that is longer than 5 characters: {state['input']}")

        return state
    ```


    Let's assume we run the graph with an input that triggers the dynamic breakpoint and then attempt to resume the graph execution simply by passing in `None` for the input.

    ```python
    # Attempt to continue the graph execution with no change to state after we hit the dynamic breakpoint 
    for event in graph.stream(None, thread_config, stream_mode="values"):
        print(event)
    ```

    The graph will *interrupt* again because this node will be *re-run* with the same graph state. We need to change the graph state such that the condition that triggers the dynamic breakpoint is no longer met. So, we can simply edit the graph state to an input that meets the condition of our dynamic breakpoint (< 5 characters) and re-run the node.

    ```python 
    # Update the state to pass the dynamic breakpoint
    graph.update_state(config=thread_config, values={"input": "foo"})
    for event in graph.stream(None, thread_config, stream_mode="values"):
        print(event)
    ```

    Alternatively, what if we want to keep our current input and skip the node (`my_node`) that performs the check? To do this, we can simply perform the graph update with `as_node="my_node"` and pass in `None` for the values. This will make no update the graph state, but run the update as `my_node`, effectively skipping the node and bypassing the dynamic breakpoint.

    ```python
    # This update will skip the node `my_node` altogether
    graph.update_state(config=thread_config, values=None, as_node="my_node")
    for event in graph.stream(None, thread_config, stream_mode="values"):
        print(event)
    ```

## The `Command` primitive

Graph execution can be resumed using the [Command](../reference/types.md#langgraph.types.Command) primitive. The `Command` primitive currently supports two ways of **resuming** graph execution after an `interrupt`:

1. **Pass a `resume` value to the `interrupt`**: Provide data, such as a user's response, to the graph using `Command(resume=value)`. Execution resumes from the beginning of the node where the `interrupt` was used, however, this time the `interrupt(...)` call will return the value passed in the `Command(resume=value)` instead of pausing the graph.
2. **Update the graph state**: Modify the graph state using `Command(update=update)`. Note that resumption starts from the beginning of the node where the `interrupt` was used. Execution resumes from the beginning of the node where the `interrupt` was used, but with the updated state.

```python
# Resume graph execution with the user's input.
graph.invoke(Command(resume={"age": "25"}), thread_config)
```

By leveraging `Command`, you can resume graph execution, handle user inputs, and dynamically adjust the graph's state or flow.

## Using with `invoke` and `ainvoke`

If you use `stream` or `ainvoke` to run the graph, you will not receive the interrupt information. To access this information, you must use the [get_state](../reference/graphs.md#langgraph.graph.graph.CompiledGraph.get_state) method to retrieve the graph state after calling `invoke` or `ainvoke`.

```python
`invoke` and `ainvoke` do not return the interrupt information. To access this information, you must use the [get_state](../reference/graphs.md#langgraph.graph.graph.CompiledGraph.get_state) method to retrieve the graph state after calling `invoke` or `ainvoke`.

```python
# Run the graph up to the breakpoint
result = graph.invoke(inputs, thread_config)
# Get the graph state to get interrupt information.
state = graph.get_state(thread_config)
# Resume the graph with the user's input.
graph.invoke(Command(resume={"age": "25"}), thread_config)
```

## How does resuming from a breakpoint work?

Execution always resumes from the **beginning** of the **graph node**, not the exact point of the `interrupt`.

Please note that this is **unlike** a traditional breakpoint or Python's `input()` function. As a result, you should structure your graph nodes to handle the `interrupt` and `resume` logic effectively.

Keep the following considerations in mind when using the `interrupt` function:

1. **Side effects**: Place side-effecting code, such as API calls, **after** the `interrupt` to avoid duplication, as these are re-triggered every time the node resumes.
2. **Multiple interrupts**: Using multiple `interrupt` calls in a node can be very useful (e.g., for run-time validation), but the order and number of calls must remain consistent to prevent mismatched resume values.

## Best practices

We recommend the `interrupt` function for setting breakpoints and using `Command(resume=value)` to resume execution.

This approach allows you to collect user input without having to immediately modify the graph state or add any special
attributes to the graph state to represent the user input.

## Additional Resources 📚

- [**Conceptual Guide: Persistence**](https://langchain-ai.github.io/langgraph/concepts/persistence/#replay): Read the persistence guide for more context on replaying.
- [**How to View and Update Past Graph State**](../how-tos/human_in_the_loop/time-travel.ipynb): Step-by-step instructions for working with graph state that demonstrate the **replay** and **fork** actions.