# Human-in-the-loop

**Human-in-the-loop** (or "on-the-loop") workflows enhance agent capabilities by incorporating human interactions at key points. Common interaction patterns include:

1. ✅ **Approval**: Pause the agent, present its current state to the user, and allow the user to approve or reject a proposed action.
2. 📝 **Editing**: Pause the agent, present its current state to the user, and allow the user to make modifications to the agent's state.
3. 💬 **Input**: Introduce a dedicated graph node to explicitly collect user input, which is then integrated into the agent's state.

Use-cases for these interaction patterns include:

1. [**Reviewing tool calls**](#reviewing-tool-calls): Pause the agent to review and edit the results of tool executions.
2. [**Time travel**](#time-travel): Replay or fork the agent's past actions for further exploration.

## Persistence

Human-in-the-loop patterns are enabled by LangGraph's built-in [persistence](./persistence.md) layer, which writes a checkpoint of the graph state at each step. 
Persistence allows pausing graph execution so that a human can review and / or edit the current state of the graph and then resume with the human's input.

## Breakpoints

Breakpoints allow **pausing** graph execution to allow for human review before **resuming** execution. This functionality is enabled by LangGraph's built-in [checkpointer](./persistence.md#checkpointer), which writes a checkpoint of the graph state at each step.

There are two types of breakpoints:

1. [**Static breakpoints**](#static-breakpoints): Pause the graph **before** or **after** a node executes.
2. [**Dynamic breakpoints**](#dynamic-breakpoints): Pause the graph from **inside** a node often based on some condition.
 
!!! important "Checkpointer Required"

    You must compile your graph with a checkpointer to use breakpoints.

### Static Breakpoints

Use static breakpoints if you want to **ALWAYS** pause the graph either **before** or **after** one or more nodes execute.

To set static breakpoints, specify the `interrupt_before` and/or `interrupt_after` key when [compiling your graph](#compiling-your-graph).

```python
# Compile our graph with a checkpointer and a breakpoint before "node_a" and after "node_b" and "node_c"
graph = graph_builder.compile(
    interrupt_before=["node_a"], 
    interrupt_after=["node_b", "node_c"],
    checkpointer=checkpointer, # Required
)
```

When using sub-graphs, specify the `interrupt_before` and `interrupt_after` values when compiling the subgraph.


Here, we compile our graph with a checkpointer and a breakpoint at the node we want to interrupt before, `step_for_human_in_the_loop`. We then perform one of the above interaction patterns, which will create a new checkpoint if a human edits the graph state. The new checkpoint is saved to the `thread` and we can resume the graph execution from there by passing in `None` as the input.

```python
# Compile our graph with a checkpointer and a breakpoint before "step_for_human_in_the_loop"
graph = builder.compile(checkpointer=checkpointer, interrupt_before=["step_for_human_in_the_loop"])

# Run the graph up to the breakpoint
thread_config = {"configurable": {"thread_id": "1"}}
for event in graph.stream(inputs, thread_config, stream_mode="values"):
    print(event)
    
# Perform some action that requires human in the loop

# Continue the graph execution from the current checkpoint 
for event in graph.stream(None, thread_config, stream_mode="values"):
    print(event)
```

### Dynamic Breakpoints

Alternatively, you may want to raise a breakpoint from inside a node, potentially based on some condition that is not known until runtime. This is called a dynamic breakpoint.

This concept of [dynamic breakpoints](./low_level.md#dynamic-breakpoints) is useful when the developer wants to halt the graph under *a particular condition*. This uses a `NodeInterrupt`, which is a special type of exception that can be raised from within a node based upon some condition. As an example, we can define a dynamic breakpoint that triggers when the `input` is longer than 5 characters.

There are two ways to interrupt the graph dynamically:

1. `interrupt` **function (recommended)**: Interrupts the graph within a node and surfaces a value to the client as part of the interrupt information.
2. `NodeInterrupt` exception: An older, less flexible method for interrupting.

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

Alternatively, what if we want to keep our current input and skip the node (`my_node`) that performs the check? To do this, we can simply perform the graph update with `as_node="my_node"` and pass in `None` for the values. This will make no update the graph state, but run the update as `my_node`, effectively skipping the node and bypassing the dynamic breakpoint.

```python
# This update will skip the node `my_node` altogether
graph.update_state(config=thread_config, values=None, as_node="my_node")
for event in graph.stream(None, thread_config, stream_mode="values"):
    print(event)
```

See [our guide](../how-tos/human_in_the_loop/dynamic_breakpoints.ipynb) for a detailed how-to on doing this!

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

When a breakpoint is hit, graph execution will pause.

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


## Interaction Patterns

### Approval

![](./img/human_in_the_loop/approval.png)

Sometimes we want to approve certain steps in our agent's execution. 
 
We can interrupt our agent at a [breakpoint](./low_level.md#breakpoints) prior to the step that we want to approve.

This is generally recommend for sensitive actions (e.g., using external APIs or writing to a database).
 
With persistence, we can surface the current agent state as well as the next step to a user for review and approval. 
 
If approved, the graph resumes execution from the last saved checkpoint, which is saved to the `thread`:

```python
# Compile our graph with a checkpointer and a breakpoint before the step to approve
graph = builder.compile(checkpointer=checkpointer, interrupt_before=["node_2"])

# Run the graph up to the breakpoint
for event in graph.stream(inputs, thread, stream_mode="values"):
    print(event)
    
# ... Get human approval ...

# If approved, continue the graph execution from the last saved checkpoint
for event in graph.stream(None, thread, stream_mode="values"):
    print(event)
```

See [our guide](../how-tos/human_in_the_loop/breakpoints.ipynb) for a detailed how-to on doing this!

### Editing

![](./img/human_in_the_loop/edit_graph_state.png)

Sometimes we want to review and edit the agent's state. 
 
As with approval, we can interrupt our agent at a [breakpoint](./low_level.md#breakpoints) prior to the step we want to check. 
 
We can surface the current state to a user and allow the user to edit the agent state.
 
This can, for example, be used to correct the agent if it made a mistake (e.g., see the section on tool calling below).

We can edit the graph state by forking the current checkpoint, which is saved to the `thread`.

We can then proceed with the graph from our forked checkpoint as done before. 

```python
# Compile our graph with a checkpointer and a breakpoint before the step to review
graph = builder.compile(checkpointer=checkpointer, interrupt_before=["node_2"])

# Run the graph up to the breakpoint
for event in graph.stream(inputs, thread, stream_mode="values"):
    print(event)
    
# Review the state, decide to edit it, and create a forked checkpoint with the new state
graph.update_state(thread, {"state": "new state"})

# Continue the graph execution from the forked checkpoint
for event in graph.stream(None, thread, stream_mode="values"):
    print(event)
```

See [this guide](../how-tos/human_in_the_loop/edit-graph-state.ipynb) for a detailed how-to on doing this!

### Input

![](./img/human_in_the_loop/wait_for_input.png)

Sometimes we want to explicitly get human input at a particular step in the graph. 
 
We can create a graph node designated for this (e.g., `human_input` in our example diagram).
 
As with approval and editing, we can interrupt our agent at a [breakpoint](./low_level.md#breakpoints) prior to this node.
 
We can then perform a state update that includes the human input, just as we did with editing state.

But, we add one thing: 

We can use `as_node=human_input` with the state update to specify that the state update *should be treated as a node*.

The is subtle, but important: 

With editing, the user makes a decision about whether or not to edit the graph state.

With input, we explicitly define a node in our graph for collecting human input!

The state update with the human input then runs *as this node*.

```python
# Compile our graph with a checkpointer and a breakpoint before the step to to collect human input
graph = builder.compile(checkpointer=checkpointer, interrupt_before=["human_input"])

# Run the graph up to the breakpoint
for event in graph.stream(inputs, thread, stream_mode="values"):
    print(event)
    
# Update the state with the user input as if it was the human_input node
graph.update_state(thread, {"user_input": user_input}, as_node="human_input")

# Continue the graph execution from the checkpoint created by the human_input node
for event in graph.stream(None, thread, stream_mode="values"):
    print(event)
```

See [this guide](../how-tos/human_in_the_loop/wait-user-input.ipynb) for a detailed how-to on doing this!

## Use-cases

### Reviewing Tool Calls

Some user interaction patterns combine the above ideas.

For example, many agents use [tool calling](https://python.langchain.com/docs/how_to/tool_calling/) to make decisions. 

Tool calling presents a challenge because the agent must get two things right: 

(1) The name of the tool to call 

(2) The arguments to pass to the tool

Even if the tool call is correct, we may also want to apply discretion: 

(3) The tool call may be a sensitive operation that we want to approve 

With these points in mind, we can combine the above ideas to create a human-in-the-loop review of a tool call.

```python
# Compile our graph with a checkpointer and a breakpoint before the step to to review the tool call from the LLM 
graph = builder.compile(checkpointer=checkpointer, interrupt_before=["human_review"])

# Run the graph up to the breakpoint
for event in graph.stream(inputs, thread, stream_mode="values"):
    print(event)
    
# Review the tool call and update it, if needed, as the human_review node
graph.update_state(thread, {"tool_call": "updated tool call"}, as_node="human_review")

# Otherwise, approve the tool call and proceed with the graph execution with no edits 

# Continue the graph execution from either: 
# (1) the forked checkpoint created by human_review or 
# (2) the checkpoint saved when the tool call was originally made (no edits in human_review)
for event in graph.stream(None, thread, stream_mode="values"):
    print(event)
```

See [this guide](../how-tos/human_in_the_loop/review-tool-calls.ipynb) for a detailed how-to on doing this!

### Time Travel

When working with agents, we often want closely examine their decision making process: 

(1) Even when they arrive a desired final result, the reasoning that led to that result is often important to examine.

(2) When agents make mistakes, it is often valuable to understand why.

(3) In either of the above cases, it is useful to manually explore alternative decision making paths.

Collectively, we call these debugging concepts `time-travel` and they are composed of `replaying` and `forking`.

#### Replaying

![](./img/human_in_the_loop/replay.png)

Sometimes we want to simply replay past actions of an agent. 
 
Above, we showed the case of executing an agent from the current state (or checkpoint) of the graph.

We by simply passing in `None` for the input with a `thread`.

```
thread = {"configurable": {"thread_id": "1"}}
for event in graph.stream(None, thread, stream_mode="values"):
    print(event)
```

Now, we can modify this to replay past actions from a *specific* checkpoint by passing in the checkpoint ID.

To get a specific checkpoint ID, we can easily get all of the checkpoints in the thread and filter to the one we want.

```python
all_checkpoints = []
for state in app.get_state_history(thread):
    all_checkpoints.append(state)
```

Each checkpoint has a unique ID, which we can use to replay from a specific checkpoint.

Assume from reviewing the checkpoints that we want to replay from one, `xxx`.

We just pass in the checkpoint ID when we run the graph.

```python
config = {'configurable': {'thread_id': '1', 'checkpoint_id': 'xxx'}}
for event in graph.stream(None, config, stream_mode="values"):
    print(event)
```
 
Importantly, the graph knows which checkpoints have been previously executed. 

So, it will re-play any previously executed nodes rather than re-executing them.

See [this additional conceptual guide](https://langchain-ai.github.io/langgraph/concepts/persistence/#replay) for related context on replaying.

See see [this guide](../how-tos/human_in_the_loop/time-travel.ipynb) for a detailed how-to on doing time-travel!

#### Forking

![](./img/human_in_the_loop/forking.png)

Sometimes we want to fork past actions of an agent, and explore different paths through the graph.

`Editing`, as discussed above, is *exactly* how we do this for the *current* state of the graph! 

But, what if we want to fork *past* states of the graph?

For example, let's say we want to edit a particular checkpoint, `xxx`.

We pass this `checkpoint_id` when we update the state of the graph.

```python
config = {"configurable": {"thread_id": "1", "checkpoint_id": "xxx"}}
graph.update_state(config, {"state": "updated state"}, )
```

This creates a new forked checkpoint, `xxx-fork`, which we can then run the graph from.

```python
config = {'configurable': {'thread_id': '1', 'checkpoint_id': 'xxx-fork'}}
for event in graph.stream(None, config, stream_mode="values"):
    print(event)
```

See [this additional conceptual guide](https://langchain-ai.github.io/langgraph/concepts/persistence/#update-state) for related context on forking.

See see [this guide](../how-tos/human_in_the_loop/time-travel.ipynb) for a detailed how-to on doing time-travel!
