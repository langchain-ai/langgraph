# GRAPH_RECURSION_LIMIT

Your LangGraph [`StateGraph`](https://langchain-ai.github.io/langgraph/reference/graphs/#langgraph.graph.state.StateGraph) reached the maximum number of steps before hitting a stop condition.
This is often due to an infinite loop caused by code like the example below:

:::python

```python
class State(TypedDict):
    some_key: str

builder = StateGraph(State)
builder.add_node("a", ...)
builder.add_node("b", ...)
builder.add_edge("a", "b")
builder.add_edge("b", "a")
...

graph = builder.compile()
```

:::

:::js

```typescript
import { StateGraph } from "@langchain/langgraph";
import { z } from "zod";

const State = z.object({
  someKey: z.string(),
});

const builder = new StateGraph(State)
  .addNode("a", ...)
  .addNode("b", ...)
  .addEdge("a", "b")
  .addEdge("b", "a")
  ...

const graph = builder.compile();
```

:::

However, complex graphs may hit the default limit naturally.

## Troubleshooting

- If you are not expecting your graph to go through many iterations, you likely have a cycle. Check your logic for infinite loops.

:::python

- If you have a complex graph, you can pass in a higher `recursion_limit` value into your `config` object when invoking your graph like this:

```python
graph.invoke({...}, {"recursion_limit": 100})
```

:::

:::js

- If you have a complex graph, you can pass in a higher `recursionLimit` value into your `config` object when invoking your graph like this:

```typescript
await graph.invoke({...}, { recursionLimit: 100 });
```

:::
