# Human-in-the-loop

Human-in-the-loop (or "on-the-loop") enhances agent capabilities through several common user interaction patterns:

(1) `Approval` - We can interrupt our agent, surface the current state to a user, and allow the user to accept an action. 

(2) `Editing` - We can interrupt our agent, surface the current state to a user, and allow the user to edit the agent state. 

(3) `Input` - We can explicitly create a graph node to collect human input and pass that input directly to the agent state.

(4) `Debugging` - We can manually re-play and / or fork past actions of an agent.

## Persistence

All of these interaction patterns are enabled by LangGraph's built-in [persistence](./persistence.md) layer, which will write a checkpoint of the graph state at each step. This allows us to stop the graph at a specific step (with a [breakpoint](./low_level.md#breakpoints)). Once the graph is stopped, the above human-in-the-loop interaction patterns can be implemented as shown below!

## Approval

![](./img/human_in_the_loop/approval.png)

Sometimes we want to approve certain steps in our agent's execution. 
 
We can interrupt our agent at a [breakpoint](./low_level.md#breakpoints) prior to the step that we want to approve.

This is generally recommend for sensitive actions (e.g., using external APIs or writing to a database).
 
With persistence, we can surface the current agent state as well as the next step to a user for review and approval. 
 
If approved, the graph resumes execution from the last saved checkpoint, which is saved to the `thread`:

```python
# Compile our graph with a checkpoitner and a breakpoint before the step to approve
graph = builder.compile(checkpointer=checkpoitner, interrupt_before=["node_2"])

# Run the graph up to the breakpoint
for event in graph.stream(inputs, thread, stream_mode="values"):
    print(event)
    
# ... Get human approval ...

# If approved, continue the graph execution from the last saved checkpoint
for event in graph.stream(None, thread, stream_mode="values"):
    print(event)
```

See [our guide](../how-tos/human_in_the_loop/breakpoints.ipynb) for a detailed how-to on doing this!

## Editing

![](./img/human_in_the_loop/edit_graph_state.png)

Sometimes we want to review and edit the agent's state. 
 
As with approval, we can interrupt our agent at a [breakpoint](./low_level.md#breakpoints) prior the the step we want to check. 
 
We can surface the current state to a user and allow the user to edit the agent state.
 
This can, for example, be used to correct the agent if it made a mistake (e.g., see the section on tool calling below).

We can edit the graph state by forking the current checkpoint, which is saved to the `thread`.

We can then proceed with the graph from our forked checkpoint as done before. 

```python
# Compile our graph with a checkpoitner and a breakpoint before the step to review
graph = builder.compile(checkpointer=checkpoitner, interrupt_before=["node_2"])

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

## Input

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

The the state update with the human input then runs *as this node*.

```python
# Compile our graph with a checkpoitner and a breakpoint before the step to to collect human input
graph = builder.compile(checkpointer=checkpoitner, interrupt_before=["human_input"])

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

## Review Tool Calls

Some user interaction patterns combine the above ideas.

For example, many agents use [tool calling](https://python.langchain.com/docs/how_to/tool_calling/) to make decisions. 

Tool calling presents a challenge because the agent must get two things right: 

(1) The name of the tool to call 

(2) The arguments to pass to the tool

Even if the tool call is correct, we may also want to apply discretion: 

(3) The tool call may be a sensitive operation that we want to approve 

With these points in mind, we can combine the above ideas to create a human-in-the-loop review of a tool call.

```python
# Compile our graph with a checkpoitner and a breakpoint before the step to to review the tool call from the LLM 
graph = builder.compile(checkpointer=checkpoitner, interrupt_before=["human_review"])

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

## Debugging

When working with agents, we often want closely examine their decision making process: 

(1) Even when they arrive a desired final result, the reasoning that led to that result is often important to examine.

(2) When agents make mistakes, it is often valuable to understand why.

(3) In either of the above cases, it is useful to manually explore alternative decision making paths.

Collectively, we call these debugging concepts `time-travel` and they are composed of `replaying` and `forking`.

### Replaying

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

### Forking

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