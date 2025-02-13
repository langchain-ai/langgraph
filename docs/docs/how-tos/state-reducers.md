# How to update graph state from nodes

This guide demonstrates how to define and update [state](../concepts/low_level.md/#state) in LangGraph. We will demonstrate:

1. How to use state to define a graph's [schema](../concepts/low_level.md/#schema)
2. How to use [reducers](../concepts/low_level.md/#reducers) to control how state updates are processed.

We will use [messages](../concepts/low_level.md/#messagesstate) in our examples. This represents a versatile formulation of state for many LLM applications. See our [concepts page](../concepts/low_level.md/#working-with-messages-in-graph-state) for more detail.

## Setup

First, let's install langgraph:

```python
%%capture --no-stderr
%pip install -U langgraph
```

<div class="admonition tip">
     <p class="admonition-title">Set up <a href="https://smith.langchain.com">LangSmith</a> for better debugging</p>
     <p style="padding-top: 5px;">
         Sign up for LangSmith to quickly spot issues and improve the performance of your LangGraph projects. LangSmith lets you use trace data to debug, test, and monitor your LLM aps built with LangGraph — read more about how to get started in the <a href="https://docs.smith.langchain.com">docs</a>. 
     </p>
 </div>

## Example graph

### Define state
[State](../concepts/low_level.md/#state) in LangGraph can be a `TypedDict`, `Pydantic` model, or dataclass. Below we will use `TypedDict`. See [this guide](../how-tos/state-model.ipynb) for detail on using Pydantic.

By default, graphs will have the same input and output schema, and the state determines that schema. See [this guide](../how-tos/input_output_schema.ipynb) for how to define distinct input and output schemas.

Let's consider a simple example:


```python exec="on" source="above" session="1"
from langchain_core.messages import AnyMessage
from typing_extensions import TypedDict


class State(TypedDict):
    messages: list[AnyMessage]
    extra_field: int
```

This state tracks a list of [message](https://python.langchain.com/docs/concepts/messages/) objects, as well as an extra integer field.

### Define graph structure

Let's build an example graph with a single node. Our [node](../concepts/low_level.md#nodes) is just a Python function that reads our graph's state and makes updates to it. The first argument to this function will always be the state:

```python exec="on" source="above" session="1"
from langchain_core.messages import AIMessage


def node(state: State):
    messages = state["messages"]
    new_message = AIMessage("Hello!")

    return {"messages": messages + [new_message], "extra_field": 10}
```

This node simply appends a message to our message list, and populates an extra field.

!!! important

    Nodes should return updates to the state directly, instead of mutating the state.

Let's next define a simple graph containing this node. We use [StateGraph](../concepts/low_level.md#stategraph) to define a graph that operates on this state. We then use [add_node](../concepts/low_level.md#messagesstate) populate our graph.


```python exec="on" source="above" session="1"
from langgraph.graph import StateGraph

graph_builder = StateGraph(State)
graph_builder.add_node(node)
graph_builder.set_entry_point("node")
graph = graph_builder.compile()
```

LangGraph provides built-in utilities for visualizing your graph. Let's inspect our graph. See [this guide](../how-tos/visualization.ipynb) for detail on visualization.


```python
from IPython.display import Image, display

display(Image(graph.get_graph().draw_mermaid_png()))
```

![](data:image/png;base64,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)

In this case, our graph just executes a single node.

### Use graph

Let's proceed with a simple invocation:


```python exec="on" source="above" session="1" result="ansi"
from langchain_core.messages import HumanMessage

result = graph.invoke({"messages": [HumanMessage("Hi")]})
result
```

Note that:

- We kicked off invocation by updating a single key of the state.
- We receive the entire state in the invocation result.

For convenience, we frequently inspect the content of [message objects](https://python.langchain.com/docs/concepts/messages/) via pretty-print:


```python exec="on" source="above" session="1" result="ansi"
for message in result["messages"]:
    message.pretty_print()
```

## Process state updates with reducers

Each key in the state can have its own independent [reducer](../concepts/low_level.md#reducers) function, which controls how updates from nodes are applied. If no reducer function is explicitly specified then it is assumed that all updates to the key should override it.

For `TypedDict` state schemas, we can define reducers by annotating the corresponding field of the state with a reducer function.

In the earlier example, our node updated the `"messages"` key in the state by appending a message to it. Below, we add a reducer to this key, such that updates are automatically appended:


```python exec="on" source="above" session="1"
from typing_extensions import Annotated


def add(left, right):
    """Can also import `add` from the `operator` built-in."""
    return left + right


class State(TypedDict):
    # highlight-next-line
    messages: Annotated[list[AnyMessage], add]
    extra_field: int
```

Now our node can be simplified:


```python exec="on" source="above" session="1"
def node(state: State):
    new_message = AIMessage("Hello!")
    # highlight-next-line
    return {"messages": [new_message], "extra_field": 10}
```


```python exec="on" source="above" session="1" result="ansi"
from langgraph.graph import START


graph = StateGraph(State).add_node(node).add_edge(START, "node").compile()

result = graph.invoke({"messages": [HumanMessage("Hi")]})

for message in result["messages"]:
    message.pretty_print()
```

### MessagesState

In practice, there are additional considerations for updating lists of messages:

- We may wish to update an existing message in the state.
- We may want to accept short-hands for [message formats](../concepts/low_level.md#using-messages-in-your-graph), such as [OpenAI format](https://python.langchain.com/docs/concepts/messages/#openai-format).

LangGraph includes a built-in reducer `add_messages` that handles these considerations:


```python exec="on" source="above" session="1"
from langgraph.graph.message import add_messages


class State(TypedDict):
    # highlight-next-line
    messages: Annotated[list[AnyMessage], add_messages]
    extra_field: int


def node(state: State):
    new_message = AIMessage("Hello!")
    return {"messages": [new_message], "extra_field": 10}


graph = StateGraph(State).add_node(node).set_entry_point("node").compile()
```


```python exec="on" source="above" session="1" result="ansi"
# highlight-next-line
input_message = {"role": "user", "content": "Hi"}

result = graph.invoke({"messages": [input_message]})

for message in result["messages"]:
    message.pretty_print()
```

This is a versatile representation of state for applications involving [chat models](https://python.langchain.com/docs/concepts/chat_models/). LangGraph includes a pre-built `MessagesState` for convenience, so that we can have:


```python exec="on" source="above" session="1"
from langgraph.graph import MessagesState


class State(MessagesState):
    extra_field: int
```

## Next steps

- Continue with the [Graph API Basics](index.md#graph-api-basics) guides.
- See more detail on [state management](index.md#state-management).
