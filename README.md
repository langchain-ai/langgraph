# 🦜🕸️LangGraph

[![Downloads](https://static.pepy.tech/badge/langgraph/month)](https://pepy.tech/project/langgraph)
[![Open Issues](https://img.shields.io/github/issues-raw/langchain-ai/langgraph)](https://github.com/langchain-ai/langgraph/issues)
[![](https://dcbadge.vercel.app/api/server/6adMQxSpJS?compact=true&style=flat)](https://discord.com/channels/1038097195422978059/1170024642245832774)
[![Docs](https://img.shields.io/badge/docs-latest-blue)](https://langchain-ai.github.io/langgraph/)

⚡ Building language agents as graphs ⚡

## Overview

[LangGraph](https://langchain-ai.github.io/langgraph/) is a library for building stateful, multi-actor applications with LLMs.
It extends the [LangChain Expression Language](https://python.langchain.com/docs/expression_language/) with the ability to coordinate multiple chains (or actors) across multiple steps of computation in cycles.
It is inspired by [Pregel](https://research.google/pubs/pub37252/) and [Apache Beam](https://beam.apache.org/).
The current public interface draws inspiration from [NetworkX](https://networkx.org/documentation/latest/).

The main use is for adding **cycles** to your LLM application.
Crucially, LangGraph is not **optimized for acyclic**, or Directed Acyclic Graph (DAG), workflows.
If you want to build a DAG, you can just use [LangChain Expression Language](https://python.langchain.com/docs/expression_language/).

Cycles are important for agent-like behaviors, where you call an LLM in a loop, asking it what action to take next.

## Installation

```shell
pip install -U langgraph
```

## Quick start

One of the central concepts of LangGraph is state. Each graph execution creates a state that is passed between nodes in the graph as they execute, and each node updates this internal state with its return value after it executes. The way that the graph updates its internal state is defined by either the type of graph chosen or a custom function.

State in LangGraph can be pretty general, but to keep things simpler to start, we'll show off an example where the graph's state is limited to a list of chat messages using the built-in `MessageGraph` class. This is convenient when using LangGraph with LangChain chat models because we can directly return chat model output.

First, install the LangChain OpenAI integration package:

```python
pip install langchain_openai
```

We also need to export some environment variables:

```shell
export OPENAI_API_KEY=sk-...
```

And now we're ready! The graph below contains a single node called `"oracle"` that executes a chat model, then returns the result:

```python
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from langgraph.graph import END, MessageGraph

model = ChatOpenAI(temperature=0)

graph = MessageGraph()

graph.add_node("oracle", model)
graph.add_edge("oracle", END)

graph.set_entry_point("oracle")

runnable = graph.compile()
```

Let's run it!

```python
runnable.invoke(HumanMessage("What is 1 + 1?"))
```

```
[HumanMessage(content='What is 1 + 1?'), AIMessage(content='1 + 1 equals 2.')]
```

So what did we do here? Let's break it down step by step:

1. First, we initialize our model and a `MessageGraph`.
2. Next, we add a single node to the graph, called `"oracle"`, which simply calls the model with the given input.
3. We add an edge from this `"oracle"` node to the special string `END` (`"__end__"`). This means that execution will end after the current node.
4. We set `"oracle"` as the entrypoint to the graph.
5. We compile the graph, ensuring that it can be run.

Then, when we execute the graph:

1. LangGraph adds the input message to the internal state, then passes the state to the entrypoint node, `"oracle"`.
2. The `"oracle"` node executes, invoking the chat model.
3. The chat model returns an `AIMessage`. LangGraph adds this to the state.
4. Execution progresses to the special `END` value and outputs the final state.

And as a result, we get a list of two chat messages as output.

### Interaction with LCEL

As an aside for those already familiar with LangChain - `add_node` actually takes any function or [runnable](https://python.langchain.com/docs/expression_language/interface/) as input. In the above example, the model is used "as-is", but we could also have passed in a function:

```python
def call_oracle(messages: list):
    return model.invoke(messages)

graph.add_node("oracle", call_oracle)
```

Just make sure you are mindful of the fact that the input to the [runnable](https://python.langchain.com/docs/expression_language/interface/) is the **entire current state**. So this will fail:

```python
# This will not work with MessageGraph!
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant named {name} who always speaks in pirate dialect"),
    MessagesPlaceholder(variable_name="messages"),
])

chain = prompt | model

# State is a list of messages, but our chain expects a dict input:
#
# { "name": some_string, "messages": [] }
#
# Therefore, the graph will throw an exception when it executes here.
graph.add_node("oracle", chain)
```

## Conditional edges

Now, let's move onto something a little bit less trivial. LLMs struggle with math,so let's allow the LLM to conditionally call a `"multiply"` node using [tool calling](https://python.langchain.com/docs/modules/model_io/chat/function_calling/).

We'll recreate our graph with an additional `"multiply"` that will take the result of the most recent message, if it is a tool call, and calculate the result.
We'll also [bind](https://api.python.langchain.com/en/latest/chat_models/langchain_openai.chat_models.base.ChatOpenAI.html#langchain_openai.chat_models.base.ChatOpenAI.bind_tools) the calculator's schema to the OpenAI model as a tool to allow the model to optionally use the tool necessary to respond to the current state:

```python
from langchain_core.tools import tool
from langgraph.prebuilt import ToolNode

@tool
def multiply(first_number: int, second_number: int):
    """Multiplies two numbers together."""
    return first_number * second_number

model = ChatOpenAI(temperature=0)
model_with_tools = model.bind_tools([multiply])

builder = MessageGraph()

builder.add_node("oracle", model_with_tools)

tool_node = ToolNode([multiply])
builder.add_node("multiply", tool_node)

builder.add_edge("multiply", END)

builder.set_entry_point("oracle")
```

Now let's think - what do we want to have happened?

- If the `"oracle"` node returns a message expecting a tool call, we want to execute the `"multiply"` node
- If not, we can just end execution

We can achieve this using **conditional edges**, which call a function on the current state and routes execution to a node the function's output.

Here's what that looks like:

```python
from typing import Literal

def router(state: List[BaseMessage]) -> Literal["multiply", "__end__"]:
    tool_calls = state[-1].additional_kwargs.get("tool_calls", [])
    if len(tool_calls):
        return "multiply"
    else:
        return "__end__"

builder.add_conditional_edges("oracle", router)
```

If the model output contains a tool call, we move to the `"multiply"` node. Otherwise, we end execution.

Great! Now all that's left is to compile the graph and try it out. Math-related questions are routed to the calculator tool:

```python
runnable = builder.compile()

runnable.invoke(HumanMessage("What is 123 * 456?"))
```

```

[HumanMessage(content='What is 123 * 456?'),
 AIMessage(content='', additional_kwargs={'tool_calls': [{'id': 'call_OPbdlm8Ih1mNOObGf3tMcNgb', 'function': {'arguments': '{"first_number":123,"second_number":456}', 'name': 'multiply'}, 'type': 'function'}]}),
 ToolMessage(content='56088', tool_call_id='call_OPbdlm8Ih1mNOObGf3tMcNgb')]
```

While conversational responses are outputted directly:

```python
runnable.invoke(HumanMessage("What is your name?"))
```

```
[HumanMessage(content='What is your name?'),
 AIMessage(content='My name is Assistant. How can I assist you today?')]
```

## Cycles

Now, let's go over a more general cyclic example. We will recreate the `AgentExecutor` class from LangChain. The agent itself will use chat models and tool calling.
This agent will represent all its state as a list of messages.

We will need to install some LangChain community packages, as well as [Tavily](https://app.tavily.com/sign-in) to use as an example tool.

```shell
pip install -U langgraph langchain_openai tavily-python
```

We also need to export some additional environment variables for OpenAI and Tavily API access.

```shell
export OPENAI_API_KEY=sk-...
export TAVILY_API_KEY=tvly-...
```

Optionally, we can set up [LangSmith](https://docs.smith.langchain.com/) for best-in-class observability.

```shell
export LANGCHAIN_TRACING_V2="true"
export LANGCHAIN_API_KEY=ls__...
```

### Set up the tools

As above, we will first define the tools we want to use.
For this simple example, we will use a web search tool.
However, it is really easy to create your own tools - see documentation [here](https://python.langchain.com/docs/modules/agents/tools/custom_tools) on how to do that.

```python
from langchain_community.tools.tavily_search import TavilySearchResults

tools = [TavilySearchResults(max_results=1)]
```

We can now wrap these tools in a simple LangGraph [ToolNode](https://langchain-ai.github.io/langgraph/reference/prebuilt/#toolnode).
This class receives the list of messages (containing [tool_calls](https://api.python.langchain.com/en/latest/messages/langchain_core.messages.ai.AIMessage.html#langchain_core.messages.ai.AIMessage.tool_calls), calls the tool(s) the LLM has requested to run, and returns the output as new [ToolMessage](https://api.python.langchain.com/en/latest/messages/langchain_core.messages.tool.ToolMessage.html#langchain_core.messages.tool.ToolMessage)(s).


```python
from langgraph.prebuilt import ToolNode

tool_node = ToolNode(tools)
```

### Set up the model

Now we need to load the chat model to use.

```python
from langchain_openai import ChatOpenAI

# We will set streaming=True so that we can stream tokens
# See the streaming section for more information on this.
model = ChatOpenAI(model="gpt-3.5-turbo", temperature=0, streaming=True)
```

After we've done this, we should make sure the model knows that it has these tools available to call.
We can do this by converting the LangChain tools into the format for OpenAI tool calling using the [bind_tools()](https://api.python.langchain.com/en/latest/chat_models/langchain_openai.chat_models.base.ChatOpenAI.html#langchain_openai.chat_models.base.ChatOpenAI.bind_tools) method.

```python
model = model.bind_tools(tools)
```

### Define the agent state

This time, we'll use the more general `StateGraph`.
This graph is parameterized by a state object that it passes around to each node.
Remember that each node then returns operations to update that state.
These operations can either SET specific attributes on the state (e.g. overwrite the existing values) or ADD to the existing attribute.
Whether to set or add is denoted by annotating the state object you construct the graph with.

For this example, the state we will track will just be a list of messages.
We want each node to just add messages to that list.
Therefore, we will use a `TypedDict` with one key (`messages`) and annotate it so that we always **add** to the `messages` key when updating it using the  is always added to with the second parameter (`operator.add`).
(Note: the state can be any [type](https://docs.python.org/3/library/stdtypes.html#type-objects), including [pydantic BaseModel's](https://docs.pydantic.dev/latest/api/base_model/)).

```python
from typing import TypedDict, Annotated

def add_messages(left: list, right: list):
    """Add-don't-overwrite."""
    return left + right

class AgentState(TypedDict):
    # The `add_messages` function within the annotation defines
    # *how* updates should be merged into the state.
    messages: Annotated[list, add_messages]
```

You can think of the `MessageGraph` used in the initial example as a preconfigured version of this graph, where the state is directly an array of messages,
and the update step always appends the returned values of a node to the internal state.

### Define the nodes

We now need to define a few different nodes in our graph.
In `langgraph`, a node can be either a regular python function or a [runnable](https://python.langchain.com/docs/expression_language/).

There are two main nodes we need for this:

1. The agent: responsible for deciding what (if any) actions to take.
2. A function to invoke tools: if the agent decides to take an action, this node will then execute that action. We already defined this above.

We will also need to define some edges.
Some of these edges may be conditional.
The reason they are conditional is that the destination depends on the contents of the graph's `State`.

The path that is taken is not known until that node is run (the LLM decides). For our use case, we will need one of each type of edge:

1. Conditional Edge: after the agent is called, we should either:

   a. Run tools if the agent said to take an action, OR

   b. Finish (respond to the user) if the agent did not ask to run tools

2. Normal Edge: after the tools are invoked, the graph should always return to the agent to decide what to do next

Let's define the nodes, as well as a function to define the conditional edge to take.

```python
from typing import Literal

# Define the function that determines whether to continue or not
def should_continue(state: AgentState) -> Literal["action", "__end__"]:
    messages = state['messages']
    last_message = messages[-1]
    # If the LLM makes a tool call, then we route to the "action" node
    if last_message.tool_calls:
        return "action"
    # Otherwise, we stop (reply to the user)
    return "__end__"


# Define the function that calls the model
def call_model(state: AgentState):
    messages = state['messages']
    response = model.invoke(messages)
    # We return a list, because this will get added to the existing list
    return {"messages": [response]}
```

### Define the graph

We can now put it all together and define the graph!

```python
from langgraph.graph import StateGraph, END
# Define a new graph
workflow = StateGraph(AgentState)

# Define the two nodes we will cycle between
workflow.add_node("agent", call_model)
workflow.add_node("action", tool_node)

# Set the entrypoint as `agent`
# This means that this node is the first one called
workflow.set_entry_point("agent")

# We now add a conditional edge
workflow.add_conditional_edges(
    # First, we define the start node. We use `agent`.
    # This means these are the edges taken after the `agent` node is called.
    "agent",
    # Next, we pass in the function that will determine which node is called next.
    should_continue,
)

# We now add a normal edge from `tools` to `agent`.
# This means that after `tools` is called, `agent` node is called next.
workflow.add_edge('action', 'agent')

# Finally, we compile it!
# This compiles it into a LangChain Runnable,
# meaning you can use it as you would any other runnable
app = workflow.compile()
```

### Use it!

We can now use it!
This now exposes the [same interface](https://python.langchain.com/docs/expression_language/) as all other LangChain runnables.
This [runnable](https://python.langchain.com/docs/expression_language/interface/) accepts a list of messages.

```python
from langchain_core.messages import HumanMessage

inputs = {"messages": [HumanMessage(content="what is the weather in sf")]}
app.invoke(inputs)
```

This may take a little bit - it's making a few calls behind the scenes.
In order to start seeing some intermediate results as they happen, we can use streaming - see below for more information on that.

## Streaming

LangGraph has support for several different types of streaming.

### Streaming Node Output

One of the benefits of using LangGraph is that it is easy to stream output as it's produced by each node.

```python
inputs = {"messages": [HumanMessage(content="what is the weather in sf")]}
for output in app.stream(inputs, stream_mode="updates"):
    # stream() yields dictionaries with output keyed by node name
    for key, value in output.items():
        print(f"Output from node '{key}':")
        print("---")
        print(value)
    print("\n---\n")
```

```
Output from node 'agent':
---
{'messages': [AIMessage(content='', additional_kwargs={'function_call': {'arguments': '{\n  "query": "weather in San Francisco"\n}', 'name': 'tavily_search_results_json'}})]}

---

Output from node 'action':
---
{'messages': [FunctionMessage(content="[{'url': 'https://weatherspark.com/h/m/557/2024/1/Historical-Weather-in-January-2024-in-San-Francisco-California-United-States', 'content': 'January 2024 Weather History in San Francisco California, United States  Daily Precipitation in January 2024 in San Francisco Observed Weather in January 2024 in San Francisco  San Francisco Temperature History January 2024 Hourly Temperature in January 2024 in San Francisco  Hours of Daylight and Twilight in January 2024 in San FranciscoThis report shows the past weather for San Francisco, providing a weather history for January 2024. It features all historical weather data series we have available, including the San Francisco temperature history for January 2024. You can drill down from year to month and even day level reports by clicking on the graphs.'}]", name='tavily_search_results_json')]}

---

Output from node 'agent':
---
{'messages': [AIMessage(content="I couldn't find the current weather in San Francisco. However, you can visit [WeatherSpark](https://weatherspark.com/h/m/557/2024/1/Historical-Weather-in-January-2024-in-San-Francisco-California-United-States) to check the historical weather data for January 2024 in San Francisco.")]}

---

Output from node '__end__':
---
{'messages': [HumanMessage(content='what is the weather in sf'), AIMessage(content='', additional_kwargs={'function_call': {'arguments': '{\n  "query": "weather in San Francisco"\n}', 'name': 'tavily_search_results_json'}}), FunctionMessage(content="[{'url': 'https://weatherspark.com/h/m/557/2024/1/Historical-Weather-in-January-2024-in-San-Francisco-California-United-States', 'content': 'January 2024 Weather History in San Francisco California, United States  Daily Precipitation in January 2024 in San Francisco Observed Weather in January 2024 in San Francisco  San Francisco Temperature History January 2024 Hourly Temperature in January 2024 in San Francisco  Hours of Daylight and Twilight in January 2024 in San FranciscoThis report shows the past weather for San Francisco, providing a weather history for January 2024. It features all historical weather data series we have available, including the San Francisco temperature history for January 2024. You can drill down from year to month and even day level reports by clicking on the graphs.'}]", name='tavily_search_results_json'), AIMessage(content="I couldn't find the current weather in San Francisco. However, you can visit [WeatherSpark](https://weatherspark.com/h/m/557/2024/1/Historical-Weather-in-January-2024-in-San-Francisco-California-United-States) to check the historical weather data for January 2024 in San Francisco.")]}

---
```

### Streaming LLM Tokens

You can also access the LLM tokens as they are produced by each node.
In this case only the "agent" node produces LLM tokens.
In order for this to work properly, you must be using an LLM that supports streaming as well as have set it when constructing the LLM (e.g. `ChatOpenAI(model="gpt-3.5-turbo-1106", streaming=True)`)

```python
inputs = {"messages": [HumanMessage(content="what is the weather in sf")]}
async for output in app.astream_log(inputs, include_types=["llm"]):
    # astream_log() yields the requested logs (here LLMs) in JSONPatch format
    for op in output.ops:
        if op["path"] == "/streamed_output/-":
            # this is the output from .stream()
            ...
        elif op["path"].startswith("/logs/") and op["path"].endswith(
            "/streamed_output/-"
        ):
            # because we chose to only include LLMs, these are LLM tokens
            print(op["value"])
```

```
content='' additional_kwargs={'function_call': {'arguments': '', 'name': 'tavily_search_results_json'}}
content='' additional_kwargs={'function_call': {'arguments': '{\n', 'name': ''}}
content='' additional_kwargs={'function_call': {'arguments': ' ', 'name': ''}}
content='' additional_kwargs={'function_call': {'arguments': ' "', 'name': ''}}
content='' additional_kwargs={'function_call': {'arguments': 'query', 'name': ''}}
...
```

## When to Use

When should you use this versus [LangChain Expression Language](https://python.langchain.com/docs/expression_language/)?

If you need cycles.

Langchain Expression Language allows you to easily define chains (DAGs) but does not have a good mechanism for adding in cycles.
`langgraph` adds that syntax.

## Documentation

We hope this gave you a taste of what you can build! Check out the rest of the docs to learn more.

### Tutorials

Learn to build with LangGraph through guided examples in the [LangGraph Tutorials](https://langchain-ai.github.io/langgraph/tutorials/).

We recommend starting with the [Introduction to LangGraph](https://langchain-ai.github.io/langgraph/tutorials/introduction/) before trying out the more advanced guides.

### How-to Guides

The [LangGraph how-to guides](https://langchain-ai.github.io/langgraph/how-tos/) show how to accomplish specific things within LangGraph, from streaming, to adding memory & persistance, to common design patterns (branching, subgraphs, etc.), these are the place to go if you want to copy and run a specific code snippet.

### Reference

LangGraph's API has a few important classes and methods that are all covered in the [Reference Documents](https://langchain-ai.github.io/langgraph/reference/graphs/). Check these out to see the specific funcion arguments and simple examples of how to use the graph + checkpointing APIs or to see some of the higher-level prebuilt components.
