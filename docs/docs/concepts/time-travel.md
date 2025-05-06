---
search:
  boost: 2
---

# Time Travel ⏱️

When working with non-deterministic systems that make model-based decisions (e.g., agents powered by LLMs), it can be useful to examine their decision-making process in detail:

1. 🤔 **Understand Reasoning**: Analyze the steps that led to a successful result.
2. 🐞 **Debug Mistakes**: Identify where and why errors occurred.
3. 🔍 **Explore Alternatives**: Test different paths to uncover better solutions.


LangGraph provides **time travel** functionality to support these use cases. Specifically, you can **resume execution from a prior checkpoint** — either replaying the same state or modifying it to explore alternatives. In all cases, resuming past execution produces a **new fork** in the history.

## Using checkpoints to time travel

To resume from a previous checkpoint:

1. Retrieve the available checkpoints:
   ```python
   all_checkpoints = []
   for state in graph.get_state_history(thread):
       all_checkpoints.append(state)
   ```
2. Identify the desired checkpoint’s ID (e.g., `xyz`).
3. Configure the graph to resume from that checkpoint:
   ```python
   config = {"configurable": {"thread_id": "1", "checkpoint_id": "xyz"}}
   for event in graph.stream(None, config, stream_mode="values"):
       print(event)
   ```

This replays all steps **before** the given checkpoint and re-executes the steps **after** it, creating a **new fork** even if the downstream steps were already executed previously.

## Modifying state when resuming

You can also modify the graph’s state before continuing from a checkpoint:

```python
config = {"configurable": {"thread_id": "1", "checkpoint_id": "xyz"}}
graph.update_state(config, {"state": "updated state"})
```

This creates a **forked checkpoint** (e.g., `xyz-fork`), from which you can continue execution:

```python
config = {"configurable": {"thread_id": "1", "checkpoint_id": "xyz-fork"}}
for event in graph.stream(None, config, stream_mode="values"):
    print(event)
```

## Summary

* Resuming from any prior checkpoint — whether replaying or modifying state — **always generates a new fork**.
* This allows you to explore alternative decision paths, debug specific steps, and analyze agent behavior under different conditions without altering the original run.

## Replaying

Replaying allows us to revisit and reproduce an agent's past actions, up to and including a specific step (checkpoint).

To replay actions before a specific checkpoint, start by retrieving all checkpoints for the thread:

```python
all_checkpoints = []
for state in graph.get_state_history(thread):
    all_checkpoints.append(state)
```

Each checkpoint has a unique ID. After identifying the desired checkpoint, for instance, `xyz`, include its ID in the configuration:

```python
config = {'configurable': {'thread_id': '1', 'checkpoint_id': 'xyz'}}
for event in graph.stream(None, config, stream_mode="values"):
    print(event)
```

The graph replays previously executed steps _before_ the provided `checkpoint_id` and executes the steps _after_ `checkpoint_id` (i.e., a new fork), even if they have been executed previously.

## Forking

Forking allows you to revisit an agent's past actions and explore alternative paths within the graph.

To edit a specific checkpoint, such as `xyz`, provide its `checkpoint_id` when updating the graph's state:

```python
config = {"configurable": {"thread_id": "1", "checkpoint_id": "xyz"}}
graph.update_state(config, {"state": "updated state"})
```

This creates a new forked checkpoint, xyz-fork, from which you can continue running the graph:

```python
config = {'configurable': {'thread_id': '1', 'checkpoint_id': 'xyz-fork'}}
for event in graph.stream(None, config, stream_mode="values"):
    print(event)
```

