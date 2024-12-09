## Breakpoints

Breakpoints enable **human-in-the-loop** workflows by **pausing** graph execution to allow for human review before continuing.

There are two types of breakpoints:

1. **Static breakpoints**: Pause the graph **before** or **after** a node executes. This is achieved by specifying the `interrupt_before` and `interrupt_after` keys when [compiling your graph](#compiling-your-graph).
2. **Dynamic breakpoints**: Pause the graph from **inside** a node. This is achieved by using the `interrupt` function or raising a `NodeInterrupt` exception.

Please see the [Human-in-the-Loop guide](../human_in_the_loop) for information about breakpoints.

1. **Static breakpoints**: Pause the graph **before** or **after** a node executes.
2. **Dynamic breakpoints**: Pause the graph from **inside** a node.

### Static Breakpoints

To set static breakpoints, specify the `interrupt_before` and/or `interrupt_after` key when [compiling your graph](#compiling-your-graph).

```python
graph = graph_builder.compile(
    interrupt_before=["node_a"], 
    interrupt_after=["node_b", "node_c"],
    checkpointer=..., # Required
)
```

When using sub-graphs, specify the `interrupt_before` and `interrupt_after` values when compiling the subgraph.

### Dynamic Breakpoints

There are two ways to interrupt the graph dynamically:

1. `interrupt` **function (recommended)**: Interrupts the graph within a node and surfaces a value to the client as part of the interrupt information.
2. `NodeInterrupt` exception: An older, less flexible method for interrupting.

#### `interrupt`

```python
from langgraph.types import interrupt

def node(state: State):
    ...
    client_value = interrupt(
        # This value will be sent to the client.
        # It can be any JSON serializable value.
        {"key": "value"}
    )
    ...
```

#### `NodeInterrupt`

Throw a `NodeInterrupt` exception to interrupt the graph.

```python
def my_node(state: State) -> State:
    if len(state['input']) > 5:
        raise NodeInterrupt(f"Received input that is longer than 5 characters: {state['input']}")

    return state
```

### Resuming
1. **Static breakpoints**: Pause the graph **before** or **after** a node executes. This is achieved by specifying the `interrupt_before` and `interrupt_after` keys when [compiling your graph](#compiling-your-graph).
2. **Dynamic breakpoints**: Pause the graph from **inside** a node. This is achieved by using the `interrupt` function or raising a `NodeInterrupt` exception.

When a breakpoint is hit, graph execution will pause.
Please see the [Human-in-the-Loop guide](../human_in_the_loop) for conceptual information about breakpoints.

=== "Command"

    Resume execution using the new `Command` primitive.

    ```python
    graph.invoke(inputs, config=config) # This will pause at the breakpoint
    ...
    # Do something (e.g., get human input)
    ...
    graph.invoke(
        Command(
            # Use `resume` to pass a value to the `interrupt`.
            resume=resume, 
            # For other kinds of breakpoints, use `update` to update the state.
            update=update,
        ), 
        config=config
    )
    ```

=== "Without the Command Primitive"

    Resume execution without the `Command` primitive (older versions of LangGraph).

    ```python
    graph.invoke(inputs, config=config) # This will pause at the breakpoint
    ...
    # Do something (e.g., get human input)
    ...

    graph.update_state(update, config=config)
    graph.invoke(None, config=config)
    ```

See [this guide](../how-tos/human_in_the_loop/breakpoints.ipynb) for a full walkthrough of how to add breakpoints.

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


