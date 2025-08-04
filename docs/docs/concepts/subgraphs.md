# Subgraphs

A subgraph is a [graph](./low_level.md#graphs) that is used as a [node](./low_level.md#nodes) in another graph — this is the concept of encapsulation applied to LangGraph. Subgraphs allow you to build complex systems with multiple components that are themselves graphs.

![Subgraph](./img/subgraph.png)

Some reasons for using subgraphs are:

- building [multi-agent systems](./multi_agent.md)
- when you want to reuse a set of nodes in multiple graphs
- when you want different teams to work on different parts of the graph independently, you can define each part as a subgraph, and as long as the subgraph interface (the input and output schemas) is respected, the parent graph can be built without knowing any details of the subgraph

The main question when adding subgraphs is how the parent graph and subgraph communicate, i.e. how they pass the [state](./low_level.md#state) between each other during the graph execution. There are two scenarios:

- parent and subgraph have **shared state keys** in their state [schemas](./low_level.md#state). In this case, you can [include the subgraph as a node in the parent graph](../how-tos/subgraph.ipynb#shared-state-schemas)

  :::python

  ```python
  from langgraph.graph import StateGraph, MessagesState, START

  # Subgraph

  def call_model(state: MessagesState):
      response = model.invoke(state["messages"])
      return {"messages": response}

  subgraph_builder = StateGraph(State)
  subgraph_builder.add_node(call_model)
  ...
  # highlight-next-line
  subgraph = subgraph_builder.compile()

  # Parent graph

  builder = StateGraph(State)
  # highlight-next-line
  builder.add_node("subgraph_node", subgraph)
  builder.add_edge(START, "subgraph_node")
  graph = builder.compile()
  ...
  graph.invoke({"messages": [{"role": "user", "content": "hi!"}]})
  ```

  :::

  :::js

  ```typescript
  import { StateGraph, MessagesZodState, START } from "@langchain/langgraph";

  // Subgraph

  const subgraphBuilder = new StateGraph(MessagesZodState).addNode(
    "callModel",
    async (state) => {
      const response = await model.invoke(state.messages);
      return { messages: response };
    }
  );
  // ... other nodes and edges
  // highlight-next-line
  const subgraph = subgraphBuilder.compile();

  // Parent graph

  const builder = new StateGraph(MessagesZodState)
    // highlight-next-line
    .addNode("subgraphNode", subgraph)
    .addEdge(START, "subgraphNode");
  const graph = builder.compile();
  // ...
  await graph.invoke({ messages: [{ role: "user", content: "hi!" }] });
  ```

  :::

- parent graph and subgraph have **different schemas** (no shared state keys in their state [schemas](./low_level.md#state)). In this case, you have to [call the subgraph from inside a node in the parent graph](../how-tos/subgraph.ipynb#different-state-schemas): this is useful when the parent graph and the subgraph have different state schemas and you need to transform state before or after calling the subgraph

  :::python

  ```python
  from typing_extensions import TypedDict, Annotated
  from langchain_core.messages import AnyMessage
  from langgraph.graph import StateGraph, MessagesState, START
  from langgraph.graph.message import add_messages

  class SubgraphMessagesState(TypedDict):
      # highlight-next-line
      subgraph_messages: Annotated[list[AnyMessage], add_messages]

  # Subgraph

  # highlight-next-line
  def call_model(state: SubgraphMessagesState):
      response = model.invoke(state["subgraph_messages"])
      return {"subgraph_messages": response}

  subgraph_builder = StateGraph(SubgraphMessagesState)
  subgraph_builder.add_node("call_model_from_subgraph", call_model)
  subgraph_builder.add_edge(START, "call_model_from_subgraph")
  ...
  # highlight-next-line
  subgraph = subgraph_builder.compile()

  # Parent graph

  def call_subgraph(state: MessagesState):
      response = subgraph.invoke({"subgraph_messages": state["messages"]})
      return {"messages": response["subgraph_messages"]}

  builder = StateGraph(State)
  # highlight-next-line
  builder.add_node("subgraph_node", call_subgraph)
  builder.add_edge(START, "subgraph_node")
  graph = builder.compile()
  ...
  graph.invoke({"messages": [{"role": "user", "content": "hi!"}]})
  ```

  :::

  :::js

  ```typescript
  import { StateGraph, MessagesZodState, START } from "@langchain/langgraph";
  import { z } from "zod";

  const SubgraphState = z.object({
    // highlight-next-line
    subgraphMessages: MessagesZodState.shape.messages,
  });

  // Subgraph

  const subgraphBuilder = new StateGraph(SubgraphState)
    // highlight-next-line
    .addNode("callModelFromSubgraph", async (state) => {
      const response = await model.invoke(state.subgraphMessages);
      return { subgraphMessages: response };
    })
    .addEdge(START, "callModelFromSubgraph");
  // ...
  // highlight-next-line
  const subgraph = subgraphBuilder.compile();

  // Parent graph

  const builder = new StateGraph(MessagesZodState)
    // highlight-next-line
    .addNode("subgraphNode", async (state) => {
      const response = await subgraph.invoke({
        subgraphMessages: state.messages,
      });
      return { messages: response.subgraphMessages };
    })
    .addEdge(START, "subgraphNode");
  const graph = builder.compile();
  // ...
  await graph.invoke({ messages: [{ role: "user", content: "hi!" }] });
  ```

  :::
