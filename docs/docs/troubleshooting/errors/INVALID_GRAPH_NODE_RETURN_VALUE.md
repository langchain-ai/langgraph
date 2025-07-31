# INVALID_GRAPH_NODE_RETURN_VALUE

:::python
A LangGraph [`StateGraph`](https://langchain-ai.github.io/langgraph/reference/graphs/#langgraph.graph.state.StateGraph)
received a non-dict return type from a node. Here's an example:

```python
class State(TypedDict):
    some_key: str

def bad_node(state: State):
    # Should return a dict with a value for "some_key", not a list
    return ["whoops"]

builder = StateGraph(State)
builder.add_node(bad_node)
...

graph = builder.compile()
```

Invoking the above graph will result in an error like this:

```python
graph.invoke({ "some_key": "someval" });
```

```
InvalidUpdateError: Expected dict, got ['whoops']
For troubleshooting, visit: https://python.langchain.com/docs/troubleshooting/errors/INVALID_GRAPH_NODE_RETURN_VALUE
```

Nodes in your graph must return a dict containing one or more keys defined in your state.
:::

:::js
A LangGraph [`StateGraph`](https://langchain-ai.github.io/langgraph/reference/graphs/#langgraph.graph.state.StateGraph)
received a non-object return type from a node. Here's an example:

```typescript
import { z } from "zod";
import { StateGraph } from "@langchain/langgraph";

const State = z.object({
  someKey: z.string(),
});

const badNode = (state: z.infer<typeof State>) => {
  // Should return an object with a value for "someKey", not an array
  return ["whoops"];
};

const builder = new StateGraph(State).addNode("badNode", badNode);
// ...

const graph = builder.compile();
```

Invoking the above graph will result in an error like this:

```typescript
await graph.invoke({ someKey: "someval" });
```

```
InvalidUpdateError: Expected object, got ['whoops']
For troubleshooting, visit: https://langchain-ai.github.io/langgraphjs/troubleshooting/errors/INVALID_GRAPH_NODE_RETURN_VALUE
```

Nodes in your graph must return an object containing one or more keys defined in your state.
:::

## Troubleshooting

The following may help resolve this error:

:::python

- If you have complex logic in your node, make sure all code paths return an appropriate dict for your defined state.
  :::

:::js

- If you have complex logic in your node, make sure all code paths return an appropriate object for your defined state.
  :::
