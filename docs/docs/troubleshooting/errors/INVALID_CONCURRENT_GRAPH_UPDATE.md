# INVALID_CONCURRENT_GRAPH_UPDATE

A LangGraph [`StateGraph`](https://langchain-ai.github.io/langgraph/reference/graphs/#langgraph.graph.state.StateGraph) received concurrent updates to its state from multiple nodes to a state property that doesn't
support it.

One way this can occur is if you are using a [fanout](https://langchain-ai.github.io/langgraph/how-tos/map-reduce/)
or other parallel execution in your graph and you have defined a graph like this:

:::python

```python hl_lines="2"
class State(TypedDict):
    some_key: str

def node(state: State):
    return {"some_key": "some_string_value"}

def other_node(state: State):
    return {"some_key": "some_string_value"}


builder = StateGraph(State)
builder.add_node(node)
builder.add_node(other_node)
builder.add_edge(START, "node")
builder.add_edge(START, "other_node")
graph = builder.compile()
```

:::

:::js

```typescript hl_lines="2"
import { StateGraph, Annotation, START } from "@langchain/langgraph";
import { z } from "zod";

const State = z.object({
  someKey: z.string(),
});

const builder = new StateGraph(State)
  .addNode("node", (state) => {
    return { someKey: "some_string_value" };
  })
  .addNode("otherNode", (state) => {
    return { someKey: "some_string_value" };
  })
  .addEdge(START, "node")
  .addEdge(START, "otherNode");

const graph = builder.compile();
```

:::

:::python
If a node in the above graph returns `{ "some_key": "some_string_value" }`, this will overwrite the state value for `"some_key"` with `"some_string_value"`.
However, if multiple nodes in e.g. a fanout within a single step return values for `"some_key"`, the graph will throw this error because
there is uncertainty around how to update the internal state.
:::

:::js
If a node in the above graph returns `{ someKey: "some_string_value" }`, this will overwrite the state value for `someKey` with `"some_string_value"`.
However, if multiple nodes in e.g. a fanout within a single step return values for `someKey`, the graph will throw this error because
there is uncertainty around how to update the internal state.
:::

To get around this, you can define a reducer that combines multiple values:

:::python

```python hl_lines="5-6"
import operator
from typing import Annotated

class State(TypedDict):
    # The operator.add reducer fn makes this append-only
    some_key: Annotated[list, operator.add]
```

:::

:::js

```typescript hl_lines="4-7"
import { withLangGraph } from "@langchain/langgraph";
import { z } from "zod";

const State = z.object({
  someKey: withLangGraph(z.array(z.string()), {
    reducer: {
      fn: (existing, update) => existing.concat(update),
    },
    default: () => [],
  }),
});
```

:::

This will allow you to define logic that handles the same key returned from multiple nodes executed in parallel.

## Troubleshooting

The following may help resolve this error:

- If your graph executes nodes in parallel, make sure you have defined relevant state keys with a reducer.
