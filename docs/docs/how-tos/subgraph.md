# Use subgraphs

This guide explains the mechanics of using [subgraphs](../concepts/subgraphs.md). A common application of subgraphs is to build [multi-agent](../concepts/multi_agent.md) systems.

When adding subgraphs, you need to define how the parent graph and the subgraph communicate:

* [Shared state schemas](#shared-state-schemas) — parent and subgraph have **shared state keys** in their state [schemas](../concepts/low_level.md#state)
* [Different state schemas](#different-state-schemas) — **no shared state keys** in parent and subgraph [schemas](../concepts/low_level.md#state)

## Setup

:::python
```bash
pip install -U langgraph
```
:::

:::js
```bash
npm install @langchain/langgraph
```
:::

!!! tip "Set up LangSmith for LangGraph development"

    Sign up for [LangSmith](https://smith.langchain.com) to quickly spot issues and improve the performance of your LangGraph projects. LangSmith lets you use trace data to debug, test, and monitor your LLM apps built with LangGraph — read more about how to get started [here](https://docs.smith.langchain.com).

## Shared state schemas

A common case is for the parent graph and subgraph to communicate over a shared state key (channel) in the [schema](../concepts/low_level.md#state). For example, in [multi-agent](../concepts/multi_agent.md) systems, the agents often communicate over a shared [messages](https://langchain-ai.github.io/langgraph/concepts/low_level.md#why-use-messages) key.

If your subgraph shares state keys with the parent graph, you can follow these steps to add it to your graph:

:::python
1. Define the subgraph workflow (`subgraph_builder` in the example below) and compile it
2. Pass compiled subgraph to the `.add_node` method when defining the parent graph workflow

```python
from typing_extensions import TypedDict
from langgraph.graph.state import StateGraph, START

class State(TypedDict):
    foo: str

# Subgraph

def subgraph_node_1(state: State):
    return {"foo": "hi! " + state["foo"]}

subgraph_builder = StateGraph(State)
subgraph_builder.add_node(subgraph_node_1)
subgraph_builder.add_edge(START, "subgraph_node_1")
subgraph = subgraph_builder.compile()

# Parent graph

builder = StateGraph(State)
builder.add_node("node_1", subgraph)
builder.add_edge(START, "node_1")
graph = builder.compile()
```
:::

:::js
1. Define the subgraph workflow (`subgraphBuilder` in the example below) and compile it
2. Pass compiled subgraph to the `.addNode` method when defining the parent graph workflow

```typescript
import { StateGraph, START } from "@langchain/langgraph";
import { z } from "zod";

const State = z.object({
  foo: z.string(),
});

// Subgraph
const subgraphBuilder = new StateGraph(State)
  .addNode("subgraphNode1", (state) => {
    return { foo: "hi! " + state.foo };
  })
  .addEdge(START, "subgraphNode1");

const subgraph = subgraphBuilder.compile();

// Parent graph
const builder = new StateGraph(State)
  .addNode("node1", subgraph)
  .addEdge(START, "node1");

const graph = builder.compile();
```
:::

??? example "Full example: shared state schemas"

    :::python
    ```python
    from typing_extensions import TypedDict
    from langgraph.graph.state import StateGraph, START

    # Define subgraph
    class SubgraphState(TypedDict):
        foo: str  # (1)! 
        bar: str  # (2)!
    
    def subgraph_node_1(state: SubgraphState):
        return {"bar": "bar"}
    
    def subgraph_node_2(state: SubgraphState):
        # note that this node is using a state key ('bar') that is only available in the subgraph
        # and is sending update on the shared state key ('foo')
        return {"foo": state["foo"] + state["bar"]}
    
    subgraph_builder = StateGraph(SubgraphState)
    subgraph_builder.add_node(subgraph_node_1)
    subgraph_builder.add_node(subgraph_node_2)
    subgraph_builder.add_edge(START, "subgraph_node_1")
    subgraph_builder.add_edge("subgraph_node_1", "subgraph_node_2")
    subgraph = subgraph_builder.compile()
    
    # Define parent graph
    class ParentState(TypedDict):
        foo: str
    
    def node_1(state: ParentState):
        return {"foo": "hi! " + state["foo"]}
    
    builder = StateGraph(ParentState)
    builder.add_node("node_1", node_1)
    builder.add_node("node_2", subgraph)
    builder.add_edge(START, "node_1")
    builder.add_edge("node_1", "node_2")
    graph = builder.compile()
    
    for chunk in graph.stream({"foo": "foo"}):
        print(chunk)
    ```

    1. This key is shared with the parent graph state
    2. This key is private to the `SubgraphState` and is not visible to the parent graph
    
    ```
    {'node_1': {'foo': 'hi! foo'}}
    {'node_2': {'foo': 'hi! foobar'}}
    ```
    :::

    :::js
    ```typescript
    import { StateGraph, START } from "@langchain/langgraph";
    import { z } from "zod";

    // Define subgraph
    const SubgraphState = z.object({
      foo: z.string(),  // (1)! 
      bar: z.string(),  // (2)!
    });
    
    const subgraphBuilder = new StateGraph(SubgraphState)
      .addNode("subgraphNode1", (state) => {
        return { bar: "bar" };
      })
      .addNode("subgraphNode2", (state) => {
        // note that this node is using a state key ('bar') that is only available in the subgraph
        // and is sending update on the shared state key ('foo')
        return { foo: state.foo + state.bar };
      })
      .addEdge(START, "subgraphNode1")
      .addEdge("subgraphNode1", "subgraphNode2");
    
    const subgraph = subgraphBuilder.compile();
    
    // Define parent graph
    const ParentState = z.object({
      foo: z.string(),
    });
    
    const builder = new StateGraph(ParentState)
      .addNode("node1", (state) => {
        return { foo: "hi! " + state.foo };
      })
      .addNode("node2", subgraph)
      .addEdge(START, "node1")
      .addEdge("node1", "node2");
    
    const graph = builder.compile();
    
    for await (const chunk of await graph.stream({ foo: "foo" })) {
      console.log(chunk);
    }
    ```

    3. This key is shared with the parent graph state
    4. This key is private to the `SubgraphState` and is not visible to the parent graph
    
    ```
    { node1: { foo: 'hi! foo' } }
    { node2: { foo: 'hi! foobar' } }
    ```
    :::

## Different state schemas

For more complex systems you might want to define subgraphs that have a **completely different schema** from the parent graph (no shared keys). For example, you might want to keep a private message history for each of the agents in a [multi-agent](../concepts/multi_agent.md) system.

If that's the case for your application, you need to define a node **function that invokes the subgraph**. This function needs to transform the input (parent) state to the subgraph state before invoking the subgraph, and transform the results back to the parent state before returning the state update from the node.

:::python
```python
from typing_extensions import TypedDict
from langgraph.graph.state import StateGraph, START

class SubgraphState(TypedDict):
    bar: str

# Subgraph

def subgraph_node_1(state: SubgraphState):
    return {"bar": "hi! " + state["bar"]}

subgraph_builder = StateGraph(SubgraphState)
subgraph_builder.add_node(subgraph_node_1)
subgraph_builder.add_edge(START, "subgraph_node_1")
subgraph = subgraph_builder.compile()

# Parent graph

class State(TypedDict):
    foo: str

def call_subgraph(state: State):
    subgraph_output = subgraph.invoke({"bar": state["foo"]})  # (1)!
    return {"foo": subgraph_output["bar"]}  # (2)!

builder = StateGraph(State)
builder.add_node("node_1", call_subgraph)
builder.add_edge(START, "node_1")
graph = builder.compile()
```

1. Transform the state to the subgraph state
2. Transform response back to the parent state
:::

:::js
```typescript
import { StateGraph, START } from "@langchain/langgraph";
import { z } from "zod";

const SubgraphState = z.object({
  bar: z.string(),
});

// Subgraph
const subgraphBuilder = new StateGraph(SubgraphState)
  .addNode("subgraphNode1", (state) => {
    return { bar: "hi! " + state.bar };
  })
  .addEdge(START, "subgraphNode1");

const subgraph = subgraphBuilder.compile();

// Parent graph
const State = z.object({
  foo: z.string(),
});

const builder = new StateGraph(State)
  .addNode("node1", async (state) => {
    const subgraphOutput = await subgraph.invoke({ bar: state.foo }); // (1)!
    return { foo: subgraphOutput.bar }; // (2)!
  })
  .addEdge(START, "node1");

const graph = builder.compile();
```

1. Transform the state to the subgraph state
2. Transform response back to the parent state
:::

??? example "Full example: different state schemas"

    :::python
    ```python
    from typing_extensions import TypedDict
    from langgraph.graph.state import StateGraph, START

    # Define subgraph
    class SubgraphState(TypedDict):
        # note that none of these keys are shared with the parent graph state
        bar: str
        baz: str
    
    def subgraph_node_1(state: SubgraphState):
        return {"baz": "baz"}
    
    def subgraph_node_2(state: SubgraphState):
        return {"bar": state["bar"] + state["baz"]}
    
    subgraph_builder = StateGraph(SubgraphState)
    subgraph_builder.add_node(subgraph_node_1)
    subgraph_builder.add_node(subgraph_node_2)
    subgraph_builder.add_edge(START, "subgraph_node_1")
    subgraph_builder.add_edge("subgraph_node_1", "subgraph_node_2")
    subgraph = subgraph_builder.compile()
    
    # Define parent graph
    class ParentState(TypedDict):
        foo: str
    
    def node_1(state: ParentState):
        return {"foo": "hi! " + state["foo"]}
    
    def node_2(state: ParentState):
        response = subgraph.invoke({"bar": state["foo"]})  # (1)!
        return {"foo": response["bar"]}  # (2)!
    
    
    builder = StateGraph(ParentState)
    builder.add_node("node_1", node_1)
    builder.add_node("node_2", node_2)
    builder.add_edge(START, "node_1")
    builder.add_edge("node_1", "node_2")
    graph = builder.compile()
    
    for chunk in graph.stream({"foo": "foo"}, subgraphs=True):
        print(chunk)
    ```

    1. Transform the state to the subgraph state
    2. Transform response back to the parent state

    ```
    ((), {'node_1': {'foo': 'hi! foo'}})
    (('node_2:9c36dd0f-151a-cb42-cbad-fa2f851f9ab7',), {'grandchild_1': {'my_grandchild_key': 'hi Bob, how are you'}})
    (('node_2:9c36dd0f-151a-cb42-cbad-fa2f851f9ab7',), {'grandchild_2': {'bar': 'hi! foobaz'}})
    ((), {'node_2': {'foo': 'hi! foobaz'}})
    ```
    :::

    :::js
    ```typescript
    import { StateGraph, START } from "@langchain/langgraph";
    import { z } from "zod";

    // Define subgraph
    const SubgraphState = z.object({
      // note that none of these keys are shared with the parent graph state
      bar: z.string(),
      baz: z.string(),
    });
    
    const subgraphBuilder = new StateGraph(SubgraphState)
      .addNode("subgraphNode1", (state) => {
        return { baz: "baz" };
      })
      .addNode("subgraphNode2", (state) => {
        return { bar: state.bar + state.baz };
      })
      .addEdge(START, "subgraphNode1")
      .addEdge("subgraphNode1", "subgraphNode2");
    
    const subgraph = subgraphBuilder.compile();
    
    // Define parent graph
    const ParentState = z.object({
      foo: z.string(),
    });
    
    const builder = new StateGraph(ParentState)
      .addNode("node1", (state) => {
        return { foo: "hi! " + state.foo };
      })
      .addNode("node2", async (state) => {
        const response = await subgraph.invoke({ bar: state.foo }); // (1)!
        return { foo: response.bar }; // (2)!
      })
      .addEdge(START, "node1")
      .addEdge("node1", "node2");
    
    const graph = builder.compile();
    
    for await (const chunk of await graph.stream(
      { foo: "foo" }, 
      { subgraphs: true }
    )) {
      console.log(chunk);
    }
    ```

    3. Transform the state to the subgraph state
    4. Transform response back to the parent state

    ```
    [[], { node1: { foo: 'hi! foo' } }]
    [['node2:9c36dd0f-151a-cb42-cbad-fa2f851f9ab7'], { subgraphNode1: { baz: 'baz' } }]
    [['node2:9c36dd0f-151a-cb42-cbad-fa2f851f9ab7'], { subgraphNode2: { bar: 'hi! foobaz' } }]
    [[], { node2: { foo: 'hi! foobaz' } }]
    ```
    :::

??? example "Full example: different state schemas (two levels of subgraphs)"

    This is an example with two levels of subgraphs: parent -> child -> grandchild.

    :::python
    ```python
    # Grandchild graph
    from typing_extensions import TypedDict
    from langgraph.graph.state import StateGraph, START, END
    
    class GrandChildState(TypedDict):
        my_grandchild_key: str
    
    def grandchild_1(state: GrandChildState) -> GrandChildState:
        # NOTE: child or parent keys will not be accessible here
        return {"my_grandchild_key": state["my_grandchild_key"] + ", how are you"}
    
    
    grandchild = StateGraph(GrandChildState)
    grandchild.add_node("grandchild_1", grandchild_1)
    
    grandchild.add_edge(START, "grandchild_1")
    grandchild.add_edge("grandchild_1", END)
    
    grandchild_graph = grandchild.compile()
    
    # Child graph
    class ChildState(TypedDict):
        my_child_key: str
    
    def call_grandchild_graph(state: ChildState) -> ChildState:
        # NOTE: parent or grandchild keys won't be accessible here
        grandchild_graph_input = {"my_grandchild_key": state["my_child_key"]}  # (1)!
        grandchild_graph_output = grandchild_graph.invoke(grandchild_graph_input)
        return {"my_child_key": grandchild_graph_output["my_grandchild_key"] + " today?"}  # (2)!
    
    child = StateGraph(ChildState)
    child.add_node("child_1", call_grandchild_graph)  # (3)!
    child.add_edge(START, "child_1")
    child.add_edge("child_1", END)
    child_graph = child.compile()
    
    # Parent graph
    class ParentState(TypedDict):
        my_key: str
    
    def parent_1(state: ParentState) -> ParentState:
        # NOTE: child or grandchild keys won't be accessible here
        return {"my_key": "hi " + state["my_key"]}
    
    def parent_2(state: ParentState) -> ParentState:
        return {"my_key": state["my_key"] + " bye!"}
    
    def call_child_graph(state: ParentState) -> ParentState:
        child_graph_input = {"my_child_key": state["my_key"]}  # (4)!
        child_graph_output = child_graph.invoke(child_graph_input)
        return {"my_key": child_graph_output["my_child_key"]}  # (5)!
    
    parent = StateGraph(ParentState)
    parent.add_node("parent_1", parent_1)
    parent.add_node("child", call_child_graph)  # (6)!
    parent.add_node("parent_2", parent_2)
    
    parent.add_edge(START, "parent_1")
    parent.add_edge("parent_1", "child")
    parent.add_edge("child", "parent_2")
    parent.add_edge("parent_2", END)
    
    parent_graph = parent.compile()
    
    for chunk in parent_graph.stream({"my_key": "Bob"}, subgraphs=True):
        print(chunk)
    ```

    1. We're transforming the state from the child state channels (`my_child_key`) to the child state channels (`my_grandchild_key`)
    2. We're transforming the state from the grandchild state channels (`my_grandchild_key`) back to the child state channels (`my_child_key`)
    3. We're passing a function here instead of just compiled graph (`grandchild_graph`)
    4. We're transforming the state from the parent state channels (`my_key`) to the child state channels (`my_child_key`)
    5. We're transforming the state from the child state channels (`my_child_key`) back to the parent state channels (`my_key`)
    6. We're passing a function here instead of just a compiled graph (`child_graph`)

    ```
    ((), {'parent_1': {'my_key': 'hi Bob'}})
    (('child:2e26e9ce-602f-862c-aa66-1ea5a4655e3b', 'child_1:781bb3b1-3971-84ce-810b-acf819a03f9c'), {'grandchild_1': {'my_grandchild_key': 'hi Bob, how are you'}})
    (('child:2e26e9ce-602f-862c-aa66-1ea5a4655e3b',), {'child_1': {'my_child_key': 'hi Bob, how are you today?'}})
    ((), {'child': {'my_key': 'hi Bob, how are you today?'}})
    ((), {'parent_2': {'my_key': 'hi Bob, how are you today? bye!'}})
    ```
    :::

    :::js
    ```typescript
    import { StateGraph, START, END } from "@langchain/langgraph";
    import { z } from "zod";

    // Grandchild graph
    const GrandChildState = z.object({
      myGrandchildKey: z.string(),
    });
    
    const grandchild = new StateGraph(GrandChildState)
      .addNode("grandchild1", (state) => {
        // NOTE: child or parent keys will not be accessible here
        return { myGrandchildKey: state.myGrandchildKey + ", how are you" };
      })
      .addEdge(START, "grandchild1")
      .addEdge("grandchild1", END);
    
    const grandchildGraph = grandchild.compile();
    
    // Child graph
    const ChildState = z.object({
      myChildKey: z.string(),
    });
    
    const child = new StateGraph(ChildState)
      .addNode("child1", async (state) => {
        // NOTE: parent or grandchild keys won't be accessible here
        const grandchildGraphInput = { myGrandchildKey: state.myChildKey }; // (1)!
        const grandchildGraphOutput = await grandchildGraph.invoke(grandchildGraphInput);
        return { myChildKey: grandchildGraphOutput.myGrandchildKey + " today?" }; // (2)!
      }) // (3)!
      .addEdge(START, "child1")
      .addEdge("child1", END);
    
    const childGraph = child.compile();
    
    // Parent graph
    const ParentState = z.object({
      myKey: z.string(),
    });
    
    const parent = new StateGraph(ParentState)
      .addNode("parent1", (state) => {
        // NOTE: child or grandchild keys won't be accessible here
        return { myKey: "hi " + state.myKey };
      })
      .addNode("child", async (state) => {
        const childGraphInput = { myChildKey: state.myKey }; // (4)!
        const childGraphOutput = await childGraph.invoke(childGraphInput);
        return { myKey: childGraphOutput.myChildKey }; // (5)!
      }) // (6)!
      .addNode("parent2", (state) => {
        return { myKey: state.myKey + " bye!" };
      })
      .addEdge(START, "parent1")
      .addEdge("parent1", "child")
      .addEdge("child", "parent2")
      .addEdge("parent2", END);
    
    const parentGraph = parent.compile();
    
    for await (const chunk of await parentGraph.stream(
      { myKey: "Bob" }, 
      { subgraphs: true }
    )) {
      console.log(chunk);
    }
    ```

    7. We're transforming the state from the child state channels (`myChildKey`) to the grandchild state channels (`myGrandchildKey`)
    8. We're transforming the state from the grandchild state channels (`myGrandchildKey`) back to the child state channels (`myChildKey`)
    9. We're passing a function here instead of just compiled graph (`grandchildGraph`)
    10. We're transforming the state from the parent state channels (`myKey`) to the child state channels (`myChildKey`)
    11. We're transforming the state from the child state channels (`myChildKey`) back to the parent state channels (`myKey`)
    12. We're passing a function here instead of just a compiled graph (`childGraph`)

    ```
    [[], { parent1: { myKey: 'hi Bob' } }]
    [['child:2e26e9ce-602f-862c-aa66-1ea5a4655e3b', 'child1:781bb3b1-3971-84ce-810b-acf819a03f9c'], { grandchild1: { myGrandchildKey: 'hi Bob, how are you' } }]
    [['child:2e26e9ce-602f-862c-aa66-1ea5a4655e3b'], { child1: { myChildKey: 'hi Bob, how are you today?' } }]
    [[], { child: { myKey: 'hi Bob, how are you today?' } }]
    [[], { parent2: { myKey: 'hi Bob, how are you today? bye!' } }]
    ```
    :::

## Add persistence 

You only need to **provide the checkpointer when compiling the parent graph**. LangGraph will automatically propagate the checkpointer to the child subgraphs.

:::python
```python
from langgraph.graph import START, StateGraph
from langgraph.checkpoint.memory import MemorySaver
from typing_extensions import TypedDict

class State(TypedDict):
    foo: str

# Subgraph

def subgraph_node_1(state: State):
    return {"foo": state["foo"] + "bar"}

subgraph_builder = StateGraph(State)
subgraph_builder.add_node(subgraph_node_1)
subgraph_builder.add_edge(START, "subgraph_node_1")
subgraph = subgraph_builder.compile()

# Parent graph

builder = StateGraph(State)
builder.add_node("node_1", subgraph)
builder.add_edge(START, "node_1")

checkpointer = MemorySaver()
graph = builder.compile(checkpointer=checkpointer)
```    
:::

:::js
```typescript
import { StateGraph, START, MemorySaver } from "@langchain/langgraph";
import { z } from "zod";

const State = z.object({
  foo: z.string(),
});

// Subgraph
const subgraphBuilder = new StateGraph(State)
  .addNode("subgraphNode1", (state) => {
    return { foo: state.foo + "bar" };
  })
  .addEdge(START, "subgraphNode1");

const subgraph = subgraphBuilder.compile();

// Parent graph
const builder = new StateGraph(State)
  .addNode("node1", subgraph)
  .addEdge(START, "node1");

const checkpointer = new MemorySaver();
const graph = builder.compile({ checkpointer });
```    
:::

If you want the subgraph to **have its own memory**, you can compile it with the appropriate checkpointer option. This is useful in [multi-agent](../concepts/multi_agent.md) systems, if you want agents to keep track of their internal message histories:

:::python
```python
subgraph_builder = StateGraph(...)
subgraph = subgraph_builder.compile(checkpointer=True)
```
:::

:::js
```typescript
const subgraphBuilder = new StateGraph(...)
const subgraph = subgraphBuilder.compile({ checkpointer: true });
```
:::

## View subgraph state

When you enable [persistence](../concepts/persistence.md), you can [inspect the graph state](../concepts/persistence.md#checkpoints) (checkpoint) via the appropriate method. To view the subgraph state, you can use the subgraphs option.

:::python
You can inspect the graph state via `graph.get_state(config)`. To view the subgraph state, you can use `graph.get_state(config, subgraphs=True)`.
:::

:::js
You can inspect the graph state via `graph.getState(config)`. To view the subgraph state, you can use `graph.getState(config, { subgraphs: true })`.
:::

!!! important "Available **only** when interrupted"

    Subgraph state can only be viewed **when the subgraph is interrupted**. Once you resume the graph, you won't be able to access the subgraph state.

??? example "View interrupted subgraph state"

    :::python
    ```python
    from langgraph.graph import START, StateGraph
    from langgraph.checkpoint.memory import MemorySaver
    from langgraph.types import interrupt, Command
    from typing_extensions import TypedDict
    
    class State(TypedDict):
        foo: str
    
    # Subgraph
    
    def subgraph_node_1(state: State):
        value = interrupt("Provide value:")
        return {"foo": state["foo"] + value}
    
    subgraph_builder = StateGraph(State)
    subgraph_builder.add_node(subgraph_node_1)
    subgraph_builder.add_edge(START, "subgraph_node_1")
    
    subgraph = subgraph_builder.compile()
    
    # Parent graph
        
    builder = StateGraph(State)
    builder.add_node("node_1", subgraph)
    builder.add_edge(START, "node_1")
    
    checkpointer = MemorySaver()
    graph = builder.compile(checkpointer=checkpointer)
    
    config = {"configurable": {"thread_id": "1"}}
    
    graph.invoke({"foo": ""}, config)
    parent_state = graph.get_state(config)
    subgraph_state = graph.get_state(config, subgraphs=True).tasks[0].state  # (1)!
    
    # resume the subgraph
    graph.invoke(Command(resume="bar"), config)
    ```
    
    1. This will be available only when the subgraph is interrupted. Once you resume the graph, you won't be able to access the subgraph state.
    :::

    :::js
    ```typescript
    import { StateGraph, START, MemorySaver, interrupt, Command } from "@langchain/langgraph";
    import { z } from "zod";
    
    const State = z.object({
      foo: z.string(),
    });
    
    // Subgraph
    const subgraphBuilder = new StateGraph(State)
      .addNode("subgraphNode1", (state) => {
        const value = interrupt("Provide value:");
        return { foo: state.foo + value };
      })
      .addEdge(START, "subgraphNode1");
    
    const subgraph = subgraphBuilder.compile();
    
    // Parent graph
    const builder = new StateGraph(State)
      .addNode("node1", subgraph)
      .addEdge(START, "node1");
    
    const checkpointer = new MemorySaver();
    const graph = builder.compile({ checkpointer });
    
    const config = { configurable: { thread_id: "1" } };
    
    await graph.invoke({ foo: "" }, config);
    const parentState = await graph.getState(config);
    const subgraphState = (await graph.getState(config, { subgraphs: true })).tasks[0].state; // (1)!
    
    // resume the subgraph
    await graph.invoke(new Command({ resume: "bar" }), config);
    ```
    
    2. This will be available only when the subgraph is interrupted. Once you resume the graph, you won't be able to access the subgraph state.
    :::

## Stream subgraph outputs

To include outputs from subgraphs in the streamed outputs, you can set the subgraphs option in the stream method of the parent graph. This will stream outputs from both the parent graph and any subgraphs.

:::python
```python
for chunk in graph.stream(
    {"foo": "foo"},
    subgraphs=True, # (1)!
    stream_mode="updates",
):
    print(chunk)
```

1. Set `subgraphs=True` to stream outputs from subgraphs.
:::

:::js
```typescript
for await (const chunk of await graph.stream(
  { foo: "foo" },
  {
    subgraphs: true, // (1)!
    streamMode: "updates",
  }
)) {
  console.log(chunk);
}
```

1. Set `subgraphs: true` to stream outputs from subgraphs.
:::

??? example "Stream from subgraphs"

    :::python
    ```python
    from typing_extensions import TypedDict
    from langgraph.graph.state import StateGraph, START

    # Define subgraph
    class SubgraphState(TypedDict):
        foo: str
        bar: str
    
    def subgraph_node_1(state: SubgraphState):
        return {"bar": "bar"}
    
    def subgraph_node_2(state: SubgraphState):
        # note that this node is using a state key ('bar') that is only available in the subgraph
        # and is sending update on the shared state key ('foo')
        return {"foo": state["foo"] + state["bar"]}
    
    subgraph_builder = StateGraph(SubgraphState)
    subgraph_builder.add_node(subgraph_node_1)
    subgraph_builder.add_node(subgraph_node_2)
    subgraph_builder.add_edge(START, "subgraph_node_1")
    subgraph_builder.add_edge("subgraph_node_1", "subgraph_node_2")
    subgraph = subgraph_builder.compile()
    
    # Define parent graph
    class ParentState(TypedDict):
        foo: str
    
    def node_1(state: ParentState):
        return {"foo": "hi! " + state["foo"]}
    
    builder = StateGraph(ParentState)
    builder.add_node("node_1", node_1)
    builder.add_node("node_2", subgraph)
    builder.add_edge(START, "node_1")
    builder.add_edge("node_1", "node_2")
    graph = builder.compile()

    for chunk in graph.stream(
        {"foo": "foo"},
        stream_mode="updates",
        subgraphs=True, # (1)!
    ):
        print(chunk)
    ```
  
    1. Set `subgraphs=True` to stream outputs from subgraphs.

    ```
    ((), {'node_1': {'foo': 'hi! foo'}})
    (('node_2:e58e5673-a661-ebb0-70d4-e298a7fc28b7',), {'subgraph_node_1': {'bar': 'bar'}})
    (('node_2:e58e5673-a661-ebb0-70d4-e298a7fc28b7',), {'subgraph_node_2': {'foo': 'hi! foobar'}})
    ((), {'node_2': {'foo': 'hi! foobar'}})
    ```
    :::

    :::js
    ```typescript
    import { StateGraph, START } from "@langchain/langgraph";
    import { z } from "zod";

    // Define subgraph
    const SubgraphState = z.object({
      foo: z.string(),
      bar: z.string(),
    });
    
    const subgraphBuilder = new StateGraph(SubgraphState)
      .addNode("subgraphNode1", (state) => {
        return { bar: "bar" };
      })
      .addNode("subgraphNode2", (state) => {
        // note that this node is using a state key ('bar') that is only available in the subgraph
        // and is sending update on the shared state key ('foo')
        return { foo: state.foo + state.bar };
      })
      .addEdge(START, "subgraphNode1")
      .addEdge("subgraphNode1", "subgraphNode2");
    
    const subgraph = subgraphBuilder.compile();
    
    // Define parent graph
    const ParentState = z.object({
      foo: z.string(),
    });
    
    const builder = new StateGraph(ParentState)
      .addNode("node1", (state) => {
        return { foo: "hi! " + state.foo };
      })
      .addNode("node2", subgraph)
      .addEdge(START, "node1")
      .addEdge("node1", "node2");
    
    const graph = builder.compile();

    for await (const chunk of await graph.stream(
      { foo: "foo" },
      {
        streamMode: "updates",
        subgraphs: true, // (1)!
      }
    )) {
      console.log(chunk);
    }
    ```
  
    2. Set `subgraphs: true` to stream outputs from subgraphs.

    ```
    [[], { node1: { foo: 'hi! foo' } }]
    [['node2:e58e5673-a661-ebb0-70d4-e298a7fc28b7'], { subgraphNode1: { bar: 'bar' } }]
    [['node2:e58e5673-a661-ebb0-70d4-e298a7fc28b7'], { subgraphNode2: { foo: 'hi! foobar' } }]
    [[], { node2: { foo: 'hi! foobar' } }]
    ```
    :::