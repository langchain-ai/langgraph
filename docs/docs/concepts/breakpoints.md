---
search:
  boost: 2
---

# Breakpoints

Breakpoints pause graph execution at specific points and enable stepping through execution step by step. Breakpoints are powered by LangGraph's [**persistence layer**](./persistence.md), which saves the state after each graph step.

Breakpoints can be used to inspect the state of the graph at any point in time, allowing you to view the state of the graph and the inputs to each node. This is useful for debugging and understanding how the graph is executing.

With breakpoints graph execution can be paused **indefinitely** until the user decides to resume execution as the state of the graph is persisted into the checkpointer.

Breakpoints can also be used to enable [**human-in-the-loop**](./human_in_the_loop.md) workflows, though we recommend using the [`interrupt` function](./human_in_the_loop.md#interrupt) for this purpose.

## Requirements

To use breakpoints, you will need to:

1. [**Specify a checkpointer**](persistence.md#checkpoints) to save the graph state after each step.
2. [**Set breakpoints**](#setting-breakpoints) to specify where execution should pause.
3. **Run the graph** with a [**thread ID**](./persistence.md#threads) to pause execution at the breakpoint.
4. **Resume execution** using `invoke`/`ainvoke`/`stream`/`astream` (see [**The `Command` primitive**](./human_in_the_loop.md#the-command-primitive)).

## Setting breakpoints

There are two places where you can set breakpoints:

1. **Before** or **after** a node executes by setting breakpoints at **compile time** or **run time**. We call these [**static breakpoints**](#static-breakpoints).
2. **Inside** a node using the [`NodeInterrupt` exception](#nodeinterrupt-exception).

=== "Compile time"

    ```python
    graph = graph_builder.compile(
        # highlight-next-line
        interrupt_before=["node_a"], # (1)!
        # highlight-next-line
        interrupt_after=["node_b", "node_c"], # (2)!
        checkpointer=checkpointer, # (3)!
    )

    config = {
        "configurable": {
            "thread_id": "some_thread"
        }
    }

    # Run the graph until the breakpoint
    graph.invoke(inputs, config=thread_config) # (4)!

    # Resume the graph
    graph.invoke(None, config=thread_config) # (5)!
    ```

    1. `interrupt_before` specifies the nodes where execution should pause before the node is executed.
    2. `interrupt_after` specifies the nodes where execution should pause after the node is executed.
    3. A checkpointer is required to enable breakpoints.
    4. The graph is run until the first breakpoint is hit.
    5. The graph is resumed by passing in `None` for the input. This will run the graph until the next breakpoint is hit.

=== "Run time"

    ```python
    # highlight-next-line
    graph.invoke( # (1)!
        inputs, 
        # highlight-next-line
        interrupt_before=["node_a"], # (2)!
        # highlight-next-line
        interrupt_after=["node_b", "node_c"] # (3)!
        config={
            "configurable": {"thread_id": "some_thread"}
        }, 
    )

    config = {
        "configurable": {
            "thread_id": "some_thread"
        }
    }

    # Run the graph until the breakpoint
    graph.invoke(inputs, config=config) # (4)!

    # Resume the graph
    graph.invoke(None, config=config) # (5)!
    ```

    1. `graph.invoke` is called with the `interrupt_before` and `interrupt_after` parameters. This is a run-time configuration and can be changed for every invocation.
    2. `interrupt_before` specifies the nodes where execution should pause before the node is executed.
    3. `interrupt_after` specifies the nodes where execution should pause after the node is executed.
    4. The graph is run until the first breakpoint is hit.
    5. The graph is resumed by passing in `None` for the input. This will run the graph until the next breakpoint is hit.

    !!! note

        You cannot set static breakpoints at runtime for **sub-graphs**.
        If you have a sub-graph, you must set the breakpoints at compilation time.




### Static breakpoints

Static breakpoints are triggered either **before** or **after** a node executes. You can set static breakpoints by specifying `interrupt_before` and `interrupt_after` at **"compile" time** or **run time**.

```python
from IPython.display import Image, display
from typing_extensions import TypedDict

from langgraph.checkpoint.memory import InMemorySaver 
from langgraph.graph import StateGraph, START, END


class State(TypedDict):
    input: str


def step_1(state):
    print("---Step 1---")
    pass


def step_2(state):
    print("---Step 2---")
    pass


def step_3(state):
    print("---Step 3---")
    pass


builder = StateGraph(State)
builder.add_node("step_1", step_1)
builder.add_node("step_2", step_2)
builder.add_node("step_3", step_3)
builder.add_edge(START, "step_1")
builder.add_edge("step_1", "step_2")
builder.add_edge("step_2", "step_3")
builder.add_edge("step_3", END)

# Set up a checkpointer 
checkpointer = InMemorySaver() # (1)!

graph = builder.compile(
    checkpointer=checkpointer, # (2)!
    interrupt_before=["step_3"] # (3)!
)

# View
display(Image(graph.get_graph().draw_mermaid_png()))
```

```python
# Input
initial_input = {"input": "hello world"}

# Thread
thread = {"configurable": {"thread_id": "1"}}

# Run the graph until the first interruption
for event in graph.stream(initial_input, thread, stream_mode="values"):
    print(event)
    
# This will run until the breakpoint
# You can get the state of the graph at this point
print(graph.get_state(config))

# You can continue the graph execution by passing in `None` for the input
for event in graph.stream(None, thread, stream_mode="values"):
    print(event)
```

Static breakpoints can be especially useful for debugging if you want to step through the graph execution one
node at a time or if you want to pause the graph execution at specific nodes.

### `NodeInterrupt` exception

We recommend that you [**use the `interrupt` function instead**][langgraph.types.interrupt] of the `NodeInterrupt` exception if you're trying to implement
[human-in-the-loop](./human_in_the_loop.md) workflows. The `interrupt` function is easier to use and more flexible.

??? node "`NodeInterrupt` exception"

    The developer can define some *condition* that must be met for a breakpoint to be triggered. This concept of _dynamic breakpoints_ is useful when the developer wants to halt the graph under *a particular condition*. This uses a `NodeInterrupt`, which is a special type of exception that can be raised from within a node based upon some condition. As an example, we can define a dynamic breakpoint that triggers when the `input` is longer than 5 characters.

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

## Additional Resources 📚

- [**Conceptual Guide: Persistence**](persistence.md): Read the persistence guide for more context about persistence.
- [**Conceptual Guide: Human-in-the-loop**](human_in_the_loop.md): Read the human-in-the-loop guide for more context on integrating human feedback into LangGraph applications using breakpoints.
- [**How to View and Update Past Graph State**](../how-tos/human_in_the_loop/time-travel.ipynb): Step-by-step instructions for working with graph state that demonstrate the **replay** and **fork** actions.