# Conceptual Guides

Welcome to LangGraph, a Python library for building complex, scalable AI agents using graph-based state machines. In this guide, we'll explore the core concepts behind LangGraph and why it's uniquely suited for creating reliable, fault-tolerant agent systems. We assume you have already learned the basic covered in the [introduction tutorial](https://langchain-ai.github.io/langgraph/tutorials/introduction/#requirements) and want to deepen your understanding of LangGraph's underlying design and inner workings.

First off, why graphs?

## Background: Agents & AI Workflows as Graphs

While everyone has a slightly different definition of what constitutes an "AI Agent", we will take "agent" to mean any system that tasks a language model with controlling a looping workflow and takes actions. The prototypical LLM agent uses a ~["reasoning and action" (ReAct)](https://arxiv.org/abs/2210.03629)-style design, applying an LLM to power a basic loop with the following steps:

- reason and plan actions to take
- take actions using tools (regular software functions)
- observe the effects of the tools and re-plan or react as appropriate

While LLM agents are surprisingly effective at this, the naive agent loop doesn't deliver the [reliability users expect at scale](https://en.wikipedia.org/wiki/High_availability). They're beautifully stochastic. Well-designed systems take advantage of that randomness and apply it sensibly within a well-designed composite system and make that system **tolerant** to mistakes in the LLM's outputs, because mistakes **will** occur.

We think agents are exciting and new, but AI design patterns should apply applicable good engineering practices from Software 2.0. Some similarities include:

- AI applications must balance autonomous operations with user control.
- Agent applications resemble distributed systems in their need for error tolerance and correction.
- Multi-agent systems resemble multi-player web apps in their need for parallelism + conflict resolution.
- Everyone loves an undo button and version control.

LangGraph's primary [StateGraph](https://langchain-ai.github.io/langgraph/reference/graphs/#langgraph.graph.StateGraph) abstraction is designed to support these and other needs, providing an API that is lower level than other agent frameworks such as LangChain's [AgentExecutor](https://python.langchain.com/v0.1/docs/modules/agents/) to give you full control of where and how to apply "AI."

It extends Google's [Pregel](https://research.google/pubs/pregel-a-system-for-large-scale-graph-processing/) graph processing framework to provide fault tolerance and recovery when running long or error-prone workloads. When developing, you can focus on a local action or task-specific agent, and the system composes these actions to form a more capable and scalable application.

Its parallelism and `State` reduction functionality let you control what happens if, for example, multiple agents return conflicting information.

And finally, its persistent, versioned checkpointing system lets you roll back the agent's state, explore other paths, and maintain full control of what is going on.

The following sections go into greater detail about how and why all of this works.

## Core Design

At its core, LangGraph models agent workflows as state machines. You define the behavior of your agents using three key components:

1. `State`: A shared data structure that represents the current snapshot of your application. It can be any Python type, but is typically a `TypedDict` or Pydantic `BaseModel`.

2. `Nodes`: Python functions that encode the logic of your agents. They receive the current `State` as input, perform some computation or side-effect, and return an updated `State`.

3. `Edges`: Control flow rules that determine which `Node` to execute next based on the current `State`. They can be conditional branches or fixed transitions.

By composing `Nodes` and `Edges`, you can create complex, looping workflows that evolve the `State` over time. The real power, though, comes from how LangGraph manages that `State`.

Or in short: *nodes do the work. edges tell what to do next*.

LangGraph's underlying graph algorithm uses [message passing](https://en.wikipedia.org/wiki/Message_passing) to define a general program. When a `Node` completes, it sends a message along one or more edges to other node(s). These nodes run their functions, pass the resulting messages to the next set of nodes, and on and on it goes. Inspired by [Pregel](https://research.google/pubs/pregel-a-system-for-large-scale-graph-processing/), the program proceeds in discrete "super-steps" that are all executed conceptually in parallel. Whenever the graph is run, all the nodes start in an `inactive` state. Whenever an incoming edge (or "channel") receives a new message (state), the node becomes `active`, runs the function, and responds with updates. At the end of each superstep, each node votes to `halt` by marking itself as `inactive` if it has no more incoming messages. The graph terminates when all nodes are `inactive` and when no messages are in transit.

We will go through a full execution of a StateGraph later, but first, lets explore these concepts in more detail.

## Nodes

In StateGraph, nodes are typically python functions (sync or `async`) where the **first** positional argument is the [state](#state), and (optionally), the **second** positional argument is a "config", containing optional [configurable parameters](#configuration) (such as a `thread_id`).

Similar to `NetworkX`, you add these nodes to a graph using the [add_node](https://langchain-ai.github.io/langgraph/reference/graphs/#langgraph.graph.MessageGraph) method:

```python
from langchain_core.runnables import RunnableConfig
from langgraph.graph import StateGraph

builder = StateGraph(dict)


def my_node(state: dict, config: RunnableConfig):
    print("In node: ", config["configurable"]["user_id"])
    return {"results": f"Hello, {state['input']}!"}


# The second argument is optional
def my_other_node(state: dict):
    return state


builder.add_node("my_node", my_node)
builder.add_node("other_node", my_other_node)
builder.set_entry_point("my_node")
builder.add_edge("my_node", "other_node")
builder.set_finish_point("other_node")
graph = builder.compile()
graph.invoke({"input": "Will"}, {"configurable": {"user_id": "abcd-123"}})
# In node:  abcd-123
# {'results': 'Hello, Will!'}
```

Behind the seens, functions are converted to [RunnableLambda's](https://api.python.langchain.com/en/latest/runnables/langchain_core.runnables.base.RunnableLambda.html#langchain_core.runnables.base.RunnableLambda), which add batch and async support to your function, along with native tracing and debugging.

## Edges

Edges define how the logic is routed and how the graph decides to stop. Similar to nodes, they accept the current `state` of the graph and return a value.

By default, the value is the name of the node or nodes to send the state to next. All those nodes will be run in parallel as a part of the next superstep.

If you want to reuse an edge, you can optionally provide a dictionary that maps the edge's output to the name of the next node.

If you **always** want to go from node A to node B, you can use the [add_edge](https://langchain-ai.github.io/langgraph/reference/graphs/#langgraph.graph.StateGraph.add_edge) method directly.

If you want to **optionally** route to 1 or more edges (or optionally terminate), you can use the [add_conditional_edges](https://langchain-ai.github.io/langgraph/reference/graphs/#langgraph.graph.StateGraph.add_conditional_edges) method.

If a node has multiple out-going edges, **all** of those destination nodes will be executed in parallel as a part of the next superstep.

## State Management

LangGraph introduces two key ideas to state management: state schemas and reducers.

The state schema defines the type of the object that is given to each of the graph's `Node`'s.


Reducers define how to apply `Node` outputs to the current `State`. For example, you might use a reducer to merge a new dialogue response into a conversation history, or average together outputs from multiple agent nodes. By annotating your `State` fields with reducer functions, you can precisely control how data flows through your application.

We'll illustrate how reducers work with an example. Compare the following two `State`'s. Can you guess the output in both case?

```python
from typing import Annotated

from langgraph.graph import StateGraph
from typing_extensions import TypedDict


class StateA(TypedDict):
    value: int


builder = StateGraph(StateA)
builder.add_node("my_node", lambda state: {"value": 1})
builder.set_entry_point("my_node")
builder.set_finish_point("my_node")
graph = builder.compile()
graph.invoke({"value": 5})
```

And `StateB`:

```python
from typing import Annotated

from langgraph.graph import StateGraph
from typing_extensions import TypedDict


def add(existing: int, new: int):
    return existing + new


class StateB(TypedDict):
    # highlight-next-line
    value: Annotated[int, add]


builder = StateGraph(StateB)
builder.add_node("my_node", lambda state: {"value": 1})
builder.set_entry_point("my_node")
builder.set_finish_point("my_node")
graph = builder.compile()
graph.invoke({"value": 5})
```

If you guesed "1" and "6", then you're correct!

In the first case (`StateA`), the result is "1", since the default **reducer** for your state is a direct overwrite.
In the second case (`StateB`), the result is "6" since we have have created the `add` function as the **reducer**. This function takes the existing state (for that field) and the state update (if provided) and returns the updated value for that state.

In general, **reducers** provided as annotations tell the graph **how to process updates for this field**.

While we typically use `TypedDict`'s as the graph's `state_schema` (i.e., `State`), it can be most any [type](https://docs.python.org/3/library/stdtypes.html#type-objects), meaning the following graph is also completely valid:

```python
# Analogous to StateA above
builder = StateGraph(int)
builder.add_node("my_node", lambda state: 1)
builder.set_entry_point("my_node")
builder.set_finish_point("my_node")
builder.compile().invoke(5)

# Analogous to StateB
def add(left, right):
    return left + right


builder = StateGraph(Annotated[int, add])
builder.add_node("my_node", lambda state: 1)
builder.set_entry_point("my_node")
builder.set_finish_point("my_node")
graph = builder.compile()
graph.invoke(5)
```

This also means you can [use a pydantic BaseModel](https://langchain-ai.github.io/langgraph/how-tos/state-model/) as your graph state to add **default values** and additional data validation.

When building simple chatbots like ChatGPT, the state can be as simple as a list of chat messages. This is the state used by [MessageGraph](https://langchain-ai.github.io/langgraph/reference/graphs/?h=message+graph#langgraph.graph.MessageGraph) (a light wrapper of `StateGraph`), which is only slightly more involved than the following:

```python
builder = StateGraph(Annotated[list, add])
```

Using a shared state within a graph comes with some design tradeoffs. For instance, you may think it feels like using dreaded global variables (though this can be addressed by namespacing arguments). However, sharing a typed state provides a number of benefits relevant to building AI workflows, including:

1. The data flow is fully inspectable before and after each "superstep".
2. The state is mutable, making it easy to let users or other software write to the same state between supersteps to control an agent's direction (using [update_state](https://langchain-ai.github.io/langgraph/reference/graphs/#langgraph.graph.graph.CompiledGraph.update_state))
3. It is well-defined when checkpointing, making it easy to save and resume or even fully version control the execution of your entire workflows in whatever storage backend you wish.

We will talk about checkpointing more in the next section.

## Persistence

Any "intelligent" system needs memory to function. AI agents are no different, requiring memory across one or more timeframes:

- they _always_ need to remember the steps already taken **within this task** (to avoid repeating itself when answering a given query)
- they _typically_ need to remember the previous turns within a multi-turn conversation with a user (for coreference resolution and additional context)
- they _ideally_ "remember" context from previous interactions with the user and from actions in a given "environment" (such as an application context) to be more personalized and efficient in its behavior

That last form of memory covers a lot (personalization, optimization, continual learning, etc.) and is beyond the scope of this conversation, though it can be easily integrated in any LangGraph workflow, and we are actively exploring the best way to expose this functionality natively.

The first two forms of memory are natively supported by the [StateGraph](https://langchain-ai.github.io/langgraph/reference/graphs/#langgraph.graph.StateGraph) API via [checkpointers](https://langchain-ai.github.io/langgraph/reference/checkpoints/#basecheckpointsaver).

#### Checkpoints

A checkpoint represents the state of a `thread` within a (potentially) multi-turn interaction between your application and a user (or users or other systems). Checkpoints that are made _within_ a single run will have a set of `next` nodes that will be executed when starting from this state. Checkpoints that are made at the end of a given run are identical, except there are no `next` nodes to transition to (the graph is awaiting user input).

Checkpointing supports chat memory and much more, letting you tag and persist every state your system has taken, regardless of whether it is within a single run or across many turns. Let's explore a bit why that is useful:

#### Single-turn Memory

**Within** a given run, each step of the agent is checkpointed. This means you could ask your agent to go create world peace. In the likely scenario that it runs into an error as it fails to do so, you can resume its quest at any time by resuming from one of its saved checkpoints.

This also lets you build **human-in-the-loop** workflows, common in use cases like [customer support bots](https://langchain-ai.github.io/langgraph/tutorials/customer-support/customer-support/), [programming assistants](https://langchain-ai.github.io/langgraph/tutorials/usaco/usaco/), and other applications. Before or after executing a given node, you can `interrupt` the graph's execution and "escalate" control to a user or support person. That person may respond immediately. Or they could respond a month from now. Either way, your workflow can resume at any time as if no time had passed at all.

#### Multi-turn Memory

Checkpoints are saved under a "thread_id" to support multi-turn interactions between users and your system. To the developer, there is absolutely no difference in how you configure your graph to add multi-turn memory support, since the checkpointing works the same throughout.

If you have some portion of state that you want to retain across turns and some state that you want to treat as "ephemeral", you can always clear the relevant state in the graph's final node.

Using checkpointing is as easy as calling `compile(checkpointer=my_checkpointer)` and then invoking it with a `thread_id` within its `configurable` parameters. You can see more in the following sections!

## Configuration

For any given graph deployment, you'll likely want some amount of configurable values that you can control at runtime. These differ from the graph **inputs** in that they aren't meant to be treated as state variables. They are more akin to "[out-of-band](https://en.wikipedia.org/wiki/Out-of-band)" communication.

A common example is a conversational `thread_id`, a `user_id`, a choice of which LLM to use, how many documents to return in a retriever, etc. While you **could** pass this within the state, it is nicer to separate out from the regular data flow.

#### Example

Let's review another example to see how our multi-turn memory works! Can you guess what `result` and `result2` look like if you run this graph?

```python
from typing import Annotated

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import StateGraph
from typing_extensions import TypedDict


def add(left, right):
    return left + right


class State(TypedDict):
    total: Annotated[int, add]
    turn: str


builder = StateGraph(State)
builder.add_node("add_one", lambda x: {"total": 1})
builder.set_entry_point("add_one")
builder.set_finish_point("add_one")

memory = MemorySaver()
graph = builder.compile(checkpointer=memory)
thread_id = "some-thread"
config = {"configurable": {"thread_id": thread_id}}
result = graph.invoke({"total": 1, "turn": "First Turn"}, config)
result2 = graph.invoke({"turn": "Next Turn"}, config)
result3 = graph.invoke({"total": 5}, config)
result4 = graph.invoke({"total": 5}, {"configurable": {"thread_id": "new-thread-id"}})
```

If you guessed the following, you're correct!

```python
>>> result
{'total': 2, 'turn': 'First Turn'}
>>> result2
{'total': 3, 'turn': 'Next Turn'}
>>> result3
{'total': 9, 'turn': 'Next Turn'}
>>> result4
{'total': 6}

```

For the first run, no checkpoint existed, so the graph ran on the raw input. The "total" value is incremented from 1 to 2, and the "turn" is set to "First Turn".

For the second run, the user provides an update to "turn" but no total! Since we are loading from the state, the previous result is incremented by one (in our "add_one" node), and the "turn" is overwritten by the user.

For the third run, the "turn" remains the same, since it is loaded from the checkpoint but not overwritten by the user. The "total" is incremented by the value provided by the user, since this is **reduced** (i.e., used to update the existing value) by the `add` function.

For the fourth run, we are using a **new thread id** for which no checkpoint is found, so the result is just the user's provided **total** incremented by one.

You probably noticed that this user-facing behavior is equivalent to running the following **without a checkpointer**.

```python
graph = builder.compile()
result = graph.invoke({"total": 1, "turn": "First Turn"})
result2 = graph.invoke({**result, "turn": "Next Turn"})
result3 = graph.invoke({**result2, "total": result2["total"] + 5})
result4 = graph.invoke({"total": 5})
```

Run this for yourself to confirm equivalence. User inputs and checkpoint loading is treated more or less the same as any other **state update**.

Now that we've introduced the core concepts behind LangGraph, it may be instructive to walk through an end-to-end example to see how all the pieces fit together.

## Data flow of a single execution of a StateGraph

As engineers, we are never really satisfied until we know what's going on "under the hood". In the previous sections, we explained some of the LangGraph's core concepts. Now it's time to really show how they fit together.

Let's extend our toy example above with a conditional edge and then walk through two consecutive invocations.

```
from typing import Annotated, Literal

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import StateGraph
from typing_extensions import TypedDict


def add(left, right):
    return left + right


class State(TypedDict):
    total: Annotated[int, add]


builder = StateGraph(State)
builder.add_node("add_one", lambda x: {"total": 1})
builder.add_node("double", lambda x: {"total": x["total"]})
builder.set_entry_point("add_one")


def route(state: State) -> Literal["double", "__end__"]:
    if state["total"] < 6:
        return "double"
    return "__end__"


builder.add_conditional_edges("add_one", route)
builder.add_edge("double", "add_one")

memory = MemorySaver()
graph = builder.compile(checkpointer=memory)
```

...

And then call it for the first time:

```python
thread_id = "some-thread"
config = {"configurable": {"thread_id": thread_id}}
for step in graph.stream({"total": 1}, config, stream_mode="debug"):
    print(step["step"], step["type"], step["payload"].get("values"))
# 0 checkpoint {'total': 1}
# 1 task None
# 1 task_result None
# 1 checkpoint {'total': 2}
# 2 task None
# 2 task_result None
# 2 checkpoint {'total': 4}
# 3 task None
# 3 task_result None
# 3 checkpoint {'total': 5}
# 4 task None
# 4 task_result None
# 4 checkpoint {'total': 10}
# 5 task None
# 5 task_result None
# 5 checkpoint {'total': 11}
```

To inspect the trace of this run, check out the [LangSmith link here](https://smith.langchain.com/public/0c543370-d459-4b8d-9962-058f67bdc9ce/r). We'll walk through the execution below:

1. First, the graph looks for a checkpoint. None is found, so the state is thus initialized with a total of 0.
2. Next, the graph applies the user's input as an update to the state. The reducer adds the input (1) to the existing value (0). At the end of this superstep, the total is (1).
3. After that, the "add_one" node is called, returning 1.
4. Next, the reducer adds this update to the existing total (1). The state is now 2.
5. Then, the conditional edge "`route`" is called. Since the value is less than 6, we continue to the 'double' node.
6. Double takes the existing state (2), and returns it. The reducer is then called and adds it to the existing state. The state is now 4.
7. The graph then loops back through add_one (5), checks the conditional edge and proceeds to since it's < 6. After doubling, the total is (10).
8. The fixed edge loops back to add_one (11), checks the conditional edge, and since it is greater than 6, the program terminates.

For our second run, we will use the same configuration:

```python
result2 = graph.invoke({"total": -2, "turn": "First Turn"}, config)
7 checkpoint {'total': 10}
8 task None
8 task_result None
8 checkpoint {'total': 11}
```

To inspect the trace of this run, check out the [LangSmith link here](https://smith.langchain.com/public/f64d6733-b22e-403d-a822-e45fbaa5051d/r). We'll walk through the execution below:

1. First, the graph looks for the checkpoint. It loads it to memory as the initial state. Total is (11) as before.
2. Next, it applies the update from the user's input. The `add` **reducer** updates the total from 11 to -9.
3. After that, the 'add_one' node is called with this state. It returns 1.
4. That update is applied using the reducer, raising the value to 10.
5. Next, the "route" conditional edge is triggered. Since the value is greater than 6, we terminate the program, ending where we started at (11).

