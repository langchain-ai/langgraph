# Breakpoints

Breakpoints pause graph execution at specific points, enabling [**human-in-the-loop**](./human_in_the_loop.md) workflows and debugging. Breakpoints are powered by  LangGraph's [**persistence layer**](./persistence.md), which saves the state after each graph step. 

## Requirements

To use breakpoints, you will need to:

1. [**Specify a checkpointer**](persistence.md#checkpoints) to save the graph state after each step.
2. [**Set breakpoints**](#setting-breakpoints) to specify where execution should pause.
3. Run the graph with a [**thread ID**](./persistence.md#threads) to pause execution at the breakpoint.
4. **Resume execution** using `invoke`/`ainvoke`/`stream`/`astream` (see [**The `Command` primitive**](#the-command-primitive)).

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

Graph execution can be resumed using the [Command](../reference/types.md#langgraph.types.Command) primitive which can be passed through the `invoke`, `ainvoke`, `stream` or `astream` methods.

The `Command` primitive provides several options to control and modify the graph's state during resumption:

1. **Pass a value to the `interrupt`**: Provide data, such as a user's response, to the graph using `Command(resume=value)`. Execution resumes from the beginning of the node where the `interrupt` was used, however, this time the `interrupt(...)` call will return the value passed in the `Command(resume=value)` instead of pausing the graph.
2. **Update the graph state**: Modify the graph state using `Command(update=update)`. Note that resumption starts from the beginning of the node where the `interrupt` was used. Execution resumes from the beginning of the node where the `interrupt` was used, but with the updated state.
3. **Navigate to another node**: Direct the graph to continue execution at a different node using `Command(goto="node_name")`.

```python
# Resume graph execution with the user's input.
graph.invoke(Command(resume={"age": "25"}), thread_config)
```

By leveraging `Command`, you can resume graph execution, handle user inputs, and dynamically adjust the graph's state or flow.

## Using with `invoke` and `ainvoke`

When you use `stream` or `astream` to run the graph, you will receive an `Interrupt` event that let you know that a breakpoint has been hit.

`invoke` and `ainvoke` do not return the interrupt information. To access this information, you must use the [get_state](../reference/graphs.md#langgraph.graph.graph.CompiledGraph.get_state) method to retrieve the graph state after calling `invoke` or `ainvoke`.

```python
# Run the graph up to the breakpoint
result = graph.invoke(inputs, thread_config)
# Get the graph state to get interrupt information.
state = graph.get_state(thread_config)
# Print the state values
print(state.values)
# Print the pending tasks
print(state.tasks)
# Resume the graph with the user's input.
graph.invoke(Command(resume={"age": "25"}), thread_config)
```

```pycon
{'foo': 'bar'} # State values
(
    PregelTask(
        id='5d8ffc92-8011-0c9b-8b59-9d3545b7e553', 
        name='node_foo', 
        path=('__pregel_pull', 'node_foo'), 
        error=None, 
        interrupts=(Interrupt(value='value_in_interrupt', resumable=True, ns=['node_foo:5d8ffc92-8011-0c9b-8b59-9d3545b7e553'], when='during'),), state=None, 
        result=None
    ),
) # Pending tasks. interrupts 
```

## How does resuming from a breakpoint work?

> Resuming from a breakpoint is **different** from traditional breakpoints or Python's `input()` function, where execution resumes from the exact point where the breakpoint was triggered.

A critical aspect of using breakpoints is understanding how resuming from a breakpoint works. When you resume execution after a breakpoint, the graph execution starts from the **beginning** of the **graph node** where the last breakpoint was triggered.

**All** code from the beginning of the node to the **breakpoint** will be re-executed. 

```python
counter = 0
def node(state: State):
    # All the code from the beginning of the node to the breakpoint will be re-executed
    # when the graph resumes.
    global counter
    counter += 1
    print(f"> Entered the node: {counter} # of times")
    # Pause the graph and wait for user input.
    answer = interrupt()
    print("The value of counter is:", counter)
    ...
```

Upon **resuming** the graph, the counter will be incremented a second time, resulting in the following output:

```pycon
> Entered the node: 2 # of times
The value of counter is: 2
```

Keep the following considerations in mind when using the `interrupt` function:

1. **Side effects**: Place side-effecting code, such as API calls, **after** the `interrupt` to avoid duplication, as these are re-triggered every time the node resumes.
2. **Multiple interrupts**: Using multiple `interrupt` calls in a node can be very useful (e.g., for run-time validation), but the order and number of calls must remain consistent to prevent mismatched resume values. As a result, we recommend that you structure your code in a way that avoids providing both a `resume` and a state `update` value (e.g., `Command(resume=resume, update=update)`) at the same time.
3. **Subgraphs**: If you're invoking a subgraph [as a function](low_level.md#as-a-function), the **parent** graph will be re-run from the **beginning of the node** where the subgraph was invoked.

## Best practices

* Use the `interrupt` function to set breakpoints and collect user input.
* Use `Command` to resume execution and control the graph state.
* Consider putting all side effects (e.g., API calls) after the `interrupt` to prevent duplication. See [How does resuming from a breakpoint work?](#how-does-resuming-from-a-breakpoint-work)

## Additional Resources 📚

- [**Conceptual Guide: Persistence**](persistence.md): Read the persistence guide for more context about persistence.
- [**Conceptual Guide: Human-in-the-loop**](human_in_the_loop.md): Read the human-in-the-loop guide for more context on integrating human feedback into LangGraph applications using breakpoints.
- [**How to View and Update Past Graph State**](../how-tos/human_in_the_loop/time-travel.ipynb): Step-by-step instructions for working with graph state that demonstrate the **replay** and **fork** actions.