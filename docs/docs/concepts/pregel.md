---
search:
  boost: 2
---

# LangGraph runtime

:::python
@[Pregel] implements LangGraph's runtime, managing the execution of LangGraph applications.

Compiling a @[StateGraph][StateGraph] or creating an @[entrypoint][entrypoint] produces a @[Pregel] instance that can be invoked with input.
:::

:::js
@[Pregel] implements LangGraph's runtime, managing the execution of LangGraph applications.

Compiling a @[StateGraph][StateGraph] or creating an @[entrypoint][entrypoint] produces a @[Pregel] instance that can be invoked with input.
:::

This guide explains the runtime at a high level and provides instructions for directly implementing applications with Pregel.

:::python

> **Note:** The @[Pregel] runtime is named after [Google's Pregel algorithm](https://research.google/pubs/pub37252/), which describes an efficient method for large-scale parallel computation using graphs.

:::

:::js

> **Note:** The @[Pregel] runtime is named after [Google's Pregel algorithm](https://research.google/pubs/pub37252/), which describes an efficient method for large-scale parallel computation using graphs.

:::

## Overview

In LangGraph, Pregel combines [**actors**](https://en.wikipedia.org/wiki/Actor_model) and **channels** into a single application. **Actors** read data from channels and write data to channels. Pregel organizes the execution of the application into multiple steps, following the **Pregel Algorithm**/**Bulk Synchronous Parallel** model.

Each step consists of three phases:

- **Plan**: Determine which **actors** to execute in this step. For example, in the first step, select the **actors** that subscribe to the special **input** channels; in subsequent steps, select the **actors** that subscribe to channels updated in the previous step.
- **Execution**: Execute all selected **actors** in parallel, until all complete, or one fails, or a timeout is reached. During this phase, channel updates are invisible to actors until the next step.
- **Update**: Update the channels with the values written by the **actors** in this step.

Repeat until no **actors** are selected for execution, or a maximum number of steps is reached.

## Actors

An **actor** is a `PregelNode`. It subscribes to channels, reads data from them, and writes data to them. It can be thought of as an **actor** in the Pregel algorithm. `PregelNodes` implement LangChain's Runnable interface.

## Channels

Channels are used to communicate between actors (PregelNodes). Each channel has a value type, an update type, and an update function – which takes a sequence of updates and modifies the stored value. Channels can be used to send data from one chain to another, or to send data from a chain to itself in a future step. LangGraph provides a number of built-in channels:

:::python

- @[LastValue][LastValue]: The default channel, stores the last value sent to the channel, useful for input and output values, or for sending data from one step to the next.
- @[Topic][Topic]: A configurable PubSub Topic, useful for sending multiple values between **actors**, or for accumulating output. Can be configured to deduplicate values or to accumulate values over the course of multiple steps.
- @[BinaryOperatorAggregate][BinaryOperatorAggregate]: stores a persistent value, updated by applying a binary operator to the current value and each update sent to the channel, useful for computing aggregates over multiple steps; e.g.,`total = BinaryOperatorAggregate(int, operator.add)`
  :::

:::js

- @[LastValue]: The default channel, stores the last value sent to the channel, useful for input and output values, or for sending data from one step to the next.
- @[Topic]: A configurable PubSub Topic, useful for sending multiple values between **actors**, or for accumulating output. Can be configured to deduplicate values or to accumulate values over the course of multiple steps.
- @[BinaryOperatorAggregate]: stores a persistent value, updated by applying a binary operator to the current value and each update sent to the channel, useful for computing aggregates over multiple steps; e.g.,`total = BinaryOperatorAggregate(int, operator.add)`
  :::

## Examples

:::python
While most users will interact with Pregel through the @[StateGraph][StateGraph] API or the @[entrypoint][entrypoint] decorator, it is possible to interact with Pregel directly.
:::

:::js
While most users will interact with Pregel through the @[StateGraph] API or the @[entrypoint] decorator, it is possible to interact with Pregel directly.
:::

Below are a few different examples to give you a sense of the Pregel API.

=== "Single node"

    :::python
    ```python
    from langgraph.channels import EphemeralValue
    from langgraph.pregel import Pregel, NodeBuilder

    node1 = (
        NodeBuilder().subscribe_only("a")
        .do(lambda x: x + x)
        .write_to("b")
    )

    app = Pregel(
        nodes={"node1": node1},
        channels={
            "a": EphemeralValue(str),
            "b": EphemeralValue(str),
        },
        input_channels=["a"],
        output_channels=["b"],
    )

    app.invoke({"a": "foo"})
    ```

    ```con
    {'b': 'foofoo'}
    ```
    :::

    :::js
    ```typescript
    import { EphemeralValue } from "@langchain/langgraph/channels";
    import { Pregel, NodeBuilder } from "@langchain/langgraph/pregel";

    const node1 = new NodeBuilder()
      .subscribeOnly("a")
      .do((x: string) => x + x)
      .writeTo("b");

    const app = new Pregel({
      nodes: { node1 },
      channels: {
        a: new EphemeralValue<string>(),
        b: new EphemeralValue<string>(),
      },
      inputChannels: ["a"],
      outputChannels: ["b"],
    });

    await app.invoke({ a: "foo" });
    ```

    ```console
    { b: 'foofoo' }
    ```
    :::

=== "Multiple nodes"

    :::python
    ```python
    from langgraph.channels import LastValue, EphemeralValue
    from langgraph.pregel import Pregel, NodeBuilder

    node1 = (
        NodeBuilder().subscribe_only("a")
        .do(lambda x: x + x)
        .write_to("b")
    )

    node2 = (
        NodeBuilder().subscribe_only("b")
        .do(lambda x: x + x)
        .write_to("c")
    )


    app = Pregel(
        nodes={"node1": node1, "node2": node2},
        channels={
            "a": EphemeralValue(str),
            "b": LastValue(str),
            "c": EphemeralValue(str),
        },
        input_channels=["a"],
        output_channels=["b", "c"],
    )

    app.invoke({"a": "foo"})
    ```

    ```con
    {'b': 'foofoo', 'c': 'foofoofoofoo'}
    ```
    :::

    :::js
    ```typescript
    import { LastValue, EphemeralValue } from "@langchain/langgraph/channels";
    import { Pregel, NodeBuilder } from "@langchain/langgraph/pregel";

    const node1 = new NodeBuilder()
      .subscribeOnly("a")
      .do((x: string) => x + x)
      .writeTo("b");

    const node2 = new NodeBuilder()
      .subscribeOnly("b")
      .do((x: string) => x + x)
      .writeTo("c");

    const app = new Pregel({
      nodes: { node1, node2 },
      channels: {
        a: new EphemeralValue<string>(),
        b: new LastValue<string>(),
        c: new EphemeralValue<string>(),
      },
      inputChannels: ["a"],
      outputChannels: ["b", "c"],
    });

    await app.invoke({ a: "foo" });
    ```

    ```console
    { b: 'foofoo', c: 'foofoofoofoo' }
    ```
    :::

=== "Topic"

    :::python
    ```python
    from langgraph.channels import EphemeralValue, Topic
    from langgraph.pregel import Pregel, NodeBuilder

    node1 = (
        NodeBuilder().subscribe_only("a")
        .do(lambda x: x + x)
        .write_to("b", "c")
    )

    node2 = (
        NodeBuilder().subscribe_to("b")
        .do(lambda x: x["b"] + x["b"])
        .write_to("c")
    )

    app = Pregel(
        nodes={"node1": node1, "node2": node2},
        channels={
            "a": EphemeralValue(str),
            "b": EphemeralValue(str),
            "c": Topic(str, accumulate=True),
        },
        input_channels=["a"],
        output_channels=["c"],
    )

    app.invoke({"a": "foo"})
    ```

    ```pycon
    {'c': ['foofoo', 'foofoofoofoo']}
    ```
    :::

    :::js
    ```typescript
    import { EphemeralValue, Topic } from "@langchain/langgraph/channels";
    import { Pregel, NodeBuilder } from "@langchain/langgraph/pregel";

    const node1 = new NodeBuilder()
      .subscribeOnly("a")
      .do((x: string) => x + x)
      .writeTo("b", "c");

    const node2 = new NodeBuilder()
      .subscribeTo("b")
      .do((x: { b: string }) => x.b + x.b)
      .writeTo("c");

    const app = new Pregel({
      nodes: { node1, node2 },
      channels: {
        a: new EphemeralValue<string>(),
        b: new EphemeralValue<string>(),
        c: new Topic<string>({ accumulate: true }),
      },
      inputChannels: ["a"],
      outputChannels: ["c"],
    });

    await app.invoke({ a: "foo" });
    ```

    ```console
    { c: ['foofoo', 'foofoofoofoo'] }
    ```
    :::

=== "BinaryOperatorAggregate"

    This examples demonstrates how to use the BinaryOperatorAggregate channel to implement a reducer.

    :::python
    ```python
    from langgraph.channels import EphemeralValue, BinaryOperatorAggregate
    from langgraph.pregel import Pregel, NodeBuilder


    node1 = (
        NodeBuilder().subscribe_only("a")
        .do(lambda x: x + x)
        .write_to("b", "c")
    )

    node2 = (
        NodeBuilder().subscribe_only("b")
        .do(lambda x: x + x)
        .write_to("c")
    )

    def reducer(current, update):
        if current:
            return current + " | " + update
        else:
            return update

    app = Pregel(
        nodes={"node1": node1, "node2": node2},
        channels={
            "a": EphemeralValue(str),
            "b": EphemeralValue(str),
            "c": BinaryOperatorAggregate(str, operator=reducer),
        },
        input_channels=["a"],
        output_channels=["c"],
    )

    app.invoke({"a": "foo"})
    ```
    :::

    :::js
    ```typescript
    import { EphemeralValue, BinaryOperatorAggregate } from "@langchain/langgraph/channels";
    import { Pregel, NodeBuilder } from "@langchain/langgraph/pregel";

    const node1 = new NodeBuilder()
      .subscribeOnly("a")
      .do((x: string) => x + x)
      .writeTo("b", "c");

    const node2 = new NodeBuilder()
      .subscribeOnly("b")
      .do((x: string) => x + x)
      .writeTo("c");

    const reducer = (current: string, update: string) => {
      if (current) {
        return current + " | " + update;
      } else {
        return update;
      }
    };

    const app = new Pregel({
      nodes: { node1, node2 },
      channels: {
        a: new EphemeralValue<string>(),
        b: new EphemeralValue<string>(),
        c: new BinaryOperatorAggregate<string>({ operator: reducer }),
      },
      inputChannels: ["a"],
      outputChannels: ["c"],
    });

    await app.invoke({ a: "foo" });
    ```
    :::

=== "Cycle"

    :::python

    This example demonstrates how to introduce a cycle in the graph, by having
    a chain write to a channel it subscribes to. Execution will continue
    until a `None` value is written to the channel.

    ```python
    from langgraph.channels import EphemeralValue
    from langgraph.pregel import Pregel, NodeBuilder, ChannelWriteEntry

    example_node = (
        NodeBuilder().subscribe_only("value")
        .do(lambda x: x + x if len(x) < 10 else None)
        .write_to(ChannelWriteEntry("value", skip_none=True))
    )

    app = Pregel(
        nodes={"example_node": example_node},
        channels={
            "value": EphemeralValue(str),
        },
        input_channels=["value"],
        output_channels=["value"],
    )

    app.invoke({"value": "a"})
    ```

    ```pycon
    {'value': 'aaaaaaaaaaaaaaaa'}
    ```
    :::

    :::js

    This example demonstrates how to introduce a cycle in the graph, by having
    a chain write to a channel it subscribes to. Execution will continue
    until a `null` value is written to the channel.

    ```typescript
    import { EphemeralValue } from "@langchain/langgraph/channels";
    import { Pregel, NodeBuilder, ChannelWriteEntry } from "@langchain/langgraph/pregel";

    const exampleNode = new NodeBuilder()
      .subscribeOnly("value")
      .do((x: string) => x.length < 10 ? x + x : null)
      .writeTo(new ChannelWriteEntry("value", { skipNone: true }));

    const app = new Pregel({
      nodes: { exampleNode },
      channels: {
        value: new EphemeralValue<string>(),
      },
      inputChannels: ["value"],
      outputChannels: ["value"],
    });

    await app.invoke({ value: "a" });
    ```

    ```console
    { value: 'aaaaaaaaaaaaaaaa' }
    ```
    :::

## High-level API

LangGraph provides two high-level APIs for creating a Pregel application: the [StateGraph (Graph API)](./low_level.md) and the [Functional API](functional_api.md).

=== "StateGraph (Graph API)"

    :::python

    The @[StateGraph (Graph API)][StateGraph] is a higher-level abstraction that simplifies the creation of Pregel applications. It allows you to define a graph of nodes and edges. When you compile the graph, the StateGraph API automatically creates the Pregel application for you.

    ```python
    from typing import TypedDict, Optional

    from langgraph.constants import START
    from langgraph.graph import StateGraph

    class Essay(TypedDict):
        topic: str
        content: Optional[str]
        score: Optional[float]

    def write_essay(essay: Essay):
        return {
            "content": f"Essay about {essay['topic']}",
        }

    def score_essay(essay: Essay):
        return {
            "score": 10
        }

    builder = StateGraph(Essay)
    builder.add_node(write_essay)
    builder.add_node(score_essay)
    builder.add_edge(START, "write_essay")

    # Compile the graph.
    # This will return a Pregel instance.
    graph = builder.compile()
    ```
    :::

    :::js

    The @[StateGraph (Graph API)][StateGraph] is a higher-level abstraction that simplifies the creation of Pregel applications. It allows you to define a graph of nodes and edges. When you compile the graph, the StateGraph API automatically creates the Pregel application for you.

    ```typescript
    import { START, StateGraph } from "@langchain/langgraph";

    interface Essay {
      topic: string;
      content?: string;
      score?: number;
    }

    const writeEssay = (essay: Essay) => {
      return {
        content: `Essay about ${essay.topic}`,
      };
    };

    const scoreEssay = (essay: Essay) => {
      return {
        score: 10
      };
    };

    const builder = new StateGraph<Essay>({
      channels: {
        topic: null,
        content: null,
        score: null,
      }
    })
      .addNode("writeEssay", writeEssay)
      .addNode("scoreEssay", scoreEssay)
      .addEdge(START, "writeEssay");

    // Compile the graph.
    // This will return a Pregel instance.
    const graph = builder.compile();
    ```
    :::

    The compiled Pregel instance will be associated with a list of nodes and channels. You can inspect the nodes and channels by printing them.

    :::python
    ```python
    print(graph.nodes)
    ```

    You will see something like this:

    ```pycon
    {'__start__': <langgraph.pregel.read.PregelNode at 0x7d05e3ba1810>,
     'write_essay': <langgraph.pregel.read.PregelNode at 0x7d05e3ba14d0>,
     'score_essay': <langgraph.pregel.read.PregelNode at 0x7d05e3ba1710>}
    ```

    ```python
    print(graph.channels)
    ```

    You should see something like this

    ```pycon
    {'topic': <langgraph.channels.last_value.LastValue at 0x7d05e3294d80>,
     'content': <langgraph.channels.last_value.LastValue at 0x7d05e3295040>,
     'score': <langgraph.channels.last_value.LastValue at 0x7d05e3295980>,
     '__start__': <langgraph.channels.ephemeral_value.EphemeralValue at 0x7d05e3297e00>,
     'write_essay': <langgraph.channels.ephemeral_value.EphemeralValue at 0x7d05e32960c0>,
     'score_essay': <langgraph.channels.ephemeral_value.EphemeralValue at 0x7d05e2d8ab80>,
     'branch:__start__:__self__:write_essay': <langgraph.channels.ephemeral_value.EphemeralValue at 0x7d05e32941c0>,
     'branch:__start__:__self__:score_essay': <langgraph.channels.ephemeral_value.EphemeralValue at 0x7d05e2d88800>,
     'branch:write_essay:__self__:write_essay': <langgraph.channels.ephemeral_value.EphemeralValue at 0x7d05e3295ec0>,
     'branch:write_essay:__self__:score_essay': <langgraph.channels.ephemeral_value.EphemeralValue at 0x7d05e2d8ac00>,
     'branch:score_essay:__self__:write_essay': <langgraph.channels.ephemeral_value.EphemeralValue at 0x7d05e2d89700>,
     'branch:score_essay:__self__:score_essay': <langgraph.channels.ephemeral_value.EphemeralValue at 0x7d05e2d8b400>,
     'start:write_essay': <langgraph.channels.ephemeral_value.EphemeralValue at 0x7d05e2d8b280>}
    ```
    :::

    :::js
    ```typescript
    console.log(graph.nodes);
    ```

    You will see something like this:

    ```console
    {
      __start__: PregelNode { ... },
      writeEssay: PregelNode { ... },
      scoreEssay: PregelNode { ... }
    }
    ```

    ```typescript
    console.log(graph.channels);
    ```

    You should see something like this

    ```console
    {
      topic: LastValue { ... },
      content: LastValue { ... },
      score: LastValue { ... },
      __start__: EphemeralValue { ... },
      writeEssay: EphemeralValue { ... },
      scoreEssay: EphemeralValue { ... },
      'branch:__start__:__self__:writeEssay': EphemeralValue { ... },
      'branch:__start__:__self__:scoreEssay': EphemeralValue { ... },
      'branch:writeEssay:__self__:writeEssay': EphemeralValue { ... },
      'branch:writeEssay:__self__:scoreEssay': EphemeralValue { ... },
      'branch:scoreEssay:__self__:writeEssay': EphemeralValue { ... },
      'branch:scoreEssay:__self__:scoreEssay': EphemeralValue { ... },
      'start:writeEssay': EphemeralValue { ... }
    }
    ```
    :::

=== "Functional API"

    :::python

    In the [Functional API](functional_api.md), you can use an @[`entrypoint`][entrypoint] to create a Pregel application. The `entrypoint` decorator allows you to define a function that takes input and returns output.

    ```python
    from typing import TypedDict, Optional

    from langgraph.checkpoint.memory import InMemorySaver
    from langgraph.func import entrypoint

    class Essay(TypedDict):
        topic: str
        content: Optional[str]
        score: Optional[float]


    checkpointer = InMemorySaver()

    @entrypoint(checkpointer=checkpointer)
    def write_essay(essay: Essay):
        return {
            "content": f"Essay about {essay['topic']}",
        }

    print("Nodes: ")
    print(write_essay.nodes)
    print("Channels: ")
    print(write_essay.channels)
    ```

    ```pycon
    Nodes:
    {'write_essay': <langgraph.pregel.read.PregelNode object at 0x7d05e2f9aad0>}
    Channels:
    {'__start__': <langgraph.channels.ephemeral_value.EphemeralValue object at 0x7d05e2c906c0>, '__end__': <langgraph.channels.last_value.LastValue object at 0x7d05e2c90c40>, '__previous__': <langgraph.channels.last_value.LastValue object at 0x7d05e1007280>}
    ```
    :::

    :::js

    In the [Functional API](functional_api.md), you can use an @[`entrypoint`][entrypoint] to create a Pregel application. The `entrypoint` decorator allows you to define a function that takes input and returns output.

    ```typescript
    import { MemorySaver } from "@langchain/langgraph";
    import { entrypoint } from "@langchain/langgraph/func";

    interface Essay {
      topic: string;
      content?: string;
      score?: number;
    }

    const checkpointer = new MemorySaver();

    const writeEssay = entrypoint(
      { checkpointer, name: "writeEssay" },
      async (essay: Essay) => {
        return {
          content: `Essay about ${essay.topic}`,
        };
      }
    );

    console.log("Nodes: ");
    console.log(writeEssay.nodes);
    console.log("Channels: ");
    console.log(writeEssay.channels);
    ```

    ```console
    Nodes:
    { writeEssay: PregelNode { ... } }
    Channels:
    {
      __start__: EphemeralValue { ... },
      __end__: LastValue { ... },
      __previous__: LastValue { ... }
    }
    ```
    :::
