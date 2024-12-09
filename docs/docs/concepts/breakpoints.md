# Breakpoints

Breakpoints pause graph execution at specific points, enabling [**human-in-the-loop**](./human_in_the_loop.md) workflows and debugging. Breakpoints depend
on LangGraph's [**persistence layer**](./persistence.md), which saves the state after each graph step. To use breakpoints, you will need to:

1. [**Specify a checkpointer**](persistence.md#checkpoints) to save the graph state after each step.
2. [**Set breakpoints**](#setting-breakpoints) to specify where execution should pause.
3. Run the graph with a [**thread ID**](./persistence.md#threads) to pause execution at the breakpoint.
4. [**Resume execution**](#resuming) from the paused state.

## Types of Breakpoints

There are three ways to add breakpoints to your graph:

1. [**Static breakpoints**](#static-breakpoints): Pause execution **before** or **after** a node by specifying `interrupt_before` and `interrupt_after` during [graph compilation](#compiling-your-graph).
2. [**Dynamic breakpoints**](#dynamic-breakpoints): Pause execution **inside** a node based on a condition that is not known until runtime. These consist of `interrupt` and `NodeInterrupt`.

## Static Breakpoints

Static breakpoints are triggered either **before** or **after** a node executes. You can set static breakpoints at:

1. **"compile" time** via the `compile` method.
2. **run time** via the `invoke`/`stream` method.

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

## Dynamic Breakpoints

You may want to raise a breakpoint from inside a node, potentially based on some condition that is not known until runtime. We refer to these as **dynamic breakpoints**.

1. `interrupt` **function (recommended)**: Interrupts the graph within a node and surfaces a value to the client as part of the interrupt information.
2. `NodeInterrupt` exception: An older, less flexible method for interrupting.

## `interrupt` function

```python
from langgraph.types import interrupt

def human_approval(state: State):
    ...
    answer = interrupt(
        # This value will be sent to the client.
        # It can be any JSON serializable value.
        {
            "question": "OK to proceed?",
            # Surface some context to the client.
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
thread_config = {
    "configurable": {
        "thread_id": "some_thread"
    }
}
graph.invoke(inputs, config=thread_config)

# Resume the graph with the user's input
graph.invoke(Command(resume={"approved": True}), config=thread_config)
```

### `NodeInterrupt`

`NodeInterrupts` are an older method for interrupting the graph. We recommend using the `interrupt` function instead.

A `NodeInterrupt` is useful when the developer wants to halt the graph under *a particular condition*. This uses a `NodeInterrupt`, which is a special type of exception that can be raised from within a node based upon some condition. As an example, we can define a dynamic breakpoint that triggers when the `input` is longer than 5 characters.

Alternatively, the developer can define some *condition* that must be met for a breakpoint to be triggered. This concept of [dynamic breakpoints](./low_level.md#dynamic-breakpoints) is useful when the developer wants to halt the graph under *a particular condition*. This uses a `NodeInterrupt`, which is a special type of exception that can be raised from within a node based upon some condition. As an example, we can define a dynamic breakpoint that triggers when the `input` is longer than 5 characters.

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


## Updating with as_node

Alternatively, what if we want to keep our current input and skip the node (`my_node`) that performs the check? To do this, we can simply perform the graph update with `as_node="my_node"` and pass in `None` for the values. This will make no update the graph state, but run the update as `my_node`, effectively skipping the node and bypassing the dynamic breakpoint.

```python
# This update will skip the node `my_node` altogether
graph.update_state(config=thread_config, values=None, as_node="my_node")
for event in graph.stream(None, thread_config, stream_mode="values"):
    print(event)
```


Throw a `NodeInterrupt` exception to interrupt the graph.

```python
def my_node(state: State) -> State:
    if len(state['input']) > 5:
        raise NodeInterrupt(f"Received input that is longer than 5 characters: {state['input']}")

    return state
```

!!! note "Use `interrupt` instead of `NodeInterrupt` if on recent LangGraph."

    The `NodeInterrupt` exception is an older method for interrupting the graph. We recommend using the `interrupt` function instead as it allows passing a `resume` value to the client. This allows addin

    The `interrupt` function allows resuming using a `Command` primitive, which provides more flexibility than the `NodeInterrupt` exception.

## Resuming

When you run a graph with breakpoints, execution will pause at the breakpoint. To resume execution, you can:

1. [**Use the `Command` primitive**](#using-the-command-primitive): Pass a value to the `interrupt` or update the graph state.
2. [**Without the Command Primitive**](#without-the-command-primitive): Update the graph state and resume

### Using the `Command` Primitive

The new [Command](../reference/types.md#langgraph.types.Command) primitive provides a flexible way to resume execution after an `interrupt`.

```python
graph.invoke(inputs, config=config) # This will pause at the breakpoint

# Do something (e.g., get human input)
graph.invoke(
    Command(
        # Use `resume` to pass a value to the `interrupt`.
        resume=resume, 
    ), 
    config=config
)
```

### Without the Command Primitive

Before the command primitive was introduced, the way to resume execution was to:

1. (optional) Update the graph state based on user input (e.g., to incorporate human feedback).  
2. Resume `graph.invoke(None, config=config)` using a `None` and the same `config` as the original invocation (which contains the thread ID).

```python
graph.invoke(inputs, config=config) # This will pause at the breakpoint
...
# Do something (e.g., get human input)
...

graph.update_state(update, config=config)
graph.invoke(None, config=config)
```

## Comparison of methods





### How does an `interrupt` work?

Execution always resumes from the **beginning** of the **graph node**, not the exact point of the `interrupt`.

Please note that this is **unlike** a traditional breakpoint or Python's `input()` function. As a result, you should structure your graph nodes to handle the `interrupt` and `resume` logic effectively.

Keep the following considerations in mind when using the `interrupt` function:

1. **Side effects**: Place side-effecting code, such as API calls, **after** the `interrupt` to avoid duplication, as these are re-triggered every time the node resumes.
2. **Multiple interrupts**: Using multiple `interrupt` calls in a node can be very useful (e.g., for run-time validation), but the order and number of calls must remain consistent to prevent mismatched resume values.


### Options for resuming execution

After an `interrupt`, graph execution can be resumed using the [Command](../reference/types.md#langgraph.types.Command) primitive. The `Command` primitive provides several options to control and modify the graph's state during resumption:

1. **Pass a value to the `interrupt`**: Provide data, such as a user's response, to the graph using `Command(resume=value)`. Execution resumes from the beginning of the node where the `interrupt` was used, however, this time the `interrupt(...)` call will return the value passed in the `Command(resume=value)` instead of pausing the graph.
2. **Update the graph state**: Modify the graph state using `Command(update=update)`. Note that resumption starts from the beginning of the node where the `interrupt` was used. Execution resumes from the beginning of the node where the `interrupt` was used, but with the updated state.
3. **Navigate to another node**: Direct the graph to continue execution at a different node using `Command(goto="node_name")`.

```python
# Resume graph execution with the user's input.
graph.invoke(Command(resume={"age": "25"}), thread_config)
```

By leveraging `Command`, you can resume graph execution, handle user inputs, and dynamically adjust the graph's state or flow.

??? note "Using other types of breakpoints"

    The `interrupt` function was introduced to address difficulties with the older methods that necessitated updating the graph state when resuming execution. You can read more about these methods in the [low-level guide](./low_level.md#breakpoints). A [previous version of this guide](v0-human-in-the-loop.md) covers the older method of setting breakpoints using **static breakpoints** and the `NodeInterrupt` exception. 

    See [this guide](../how-tos/human_in_the_loop/breakpoints.ipynb) for a full walkthrough of how to add breakpoints.


## Best practices

We currently recommend the `interrupt` function for setting breakpoints and resuming execution.
This function is more flexible and easier to use than the older methods of setting breakpoints using static breakpoints and the `NodeInterrupt` exception.

## Additional Resources 📚

- [**Conceptual Guide: Persistence**](https://langchain-ai.github.io/langgraph/concepts/persistence/#replay): Read the persistence guide for more context on replaying.
- [**How to View and Update Past Graph State**](../how-tos/human_in_the_loop/time-travel.ipynb): Step-by-step instructions for working with graph state that demonstrate the **replay** and **fork** actions.