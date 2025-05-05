# Subgraphs

A subgraph is a [graph](./low_level.md#graphs) that is used as a [node](./low_level.md#nodes) in another graph — concept of encapsulation applied to LangGraph. Subgraphs allow you to build complex systems with multiple components that are themselves graphs.

Some reasons for using subgraphs are:

- building [multi-agent systems](./multi_agent.md)
- when you want to reuse a set of nodes in multiple graphs
- when you want different teams to work on different parts of the graph independently, you can define each part as a subgraph, and as long as the subgraph interface (the input and output schemas) is respected, the parent graph can be built without knowing any details of the subgraph

The main question when adding subgraphs is how the parent graph and subgraph communicate, i.e. how they pass the [state](./low_level.md#state) between each other during the graph execution. There are two scenarios:

* parent graph and subgraph **share schema keys**. In this case, you can [add a node with the compiled subgraph](../how-tos/subgraph.ipynb#shared-state-schemas)
* parent graph and subgraph have **different schemas**. In this case, you have to [add a node function that invokes the subgraph](../how-tos/subgraph.ipynb#different-state-schemas): this is useful when the parent graph and the subgraph have different state schemas and you need to transform state before or after calling the subgraph

![Subgraph](./img/subgraph.png)