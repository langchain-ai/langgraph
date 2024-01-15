# 🦜🕸️LangGraph

⚡ Building language agents as graphs ⚡

## Overview

LangGraph is a library for building stateful, multi-actor applications with LLMs, built on top of (and intended to be used with) [LangChain](https://github.com/langchain-ai/langchain).
It extends the [LangChain Expression Language](https://python.langchain.com/docs/expression_language/) with the ability to coordinate multiple chains (or actors) across multiple steps of computation in a cyclic manner.
It is inspired by [Pregel](https://research.google/pubs/pub37252/) and [Apache Beam](https://beam.apache.org/).
The current interface exposed is one inspired by [NetworkX](https://networkx.org/documentation/latest/).

The main use is for adding **cycles** to your LLM application.
Crucially, this is NOT a **DAG** framework.
If you want to build a DAG, you should use just use [LangChain Expression Language](https://python.langchain.com/docs/expression_language/).

Cycles are important for agent-like behaviors, where you call an LLM in a loop, asking it what action to take next.

## Installation

```shell
pip install langgraph
```

## Quick Start

Here we will go over an example of recreating the [`AgentExecutor`](https://python.langchain.com/docs/modules/agents/concepts#agentexecutor) class from LangChain.
The benefits of creating it with LangGraph is that it is more modifiable.

We will also want to install some LangChain packages, as well as [Tavily](https://app.tavily.com/sign-in) to use as an example tool.

```shell
pip install -U langchain langchain_openai langchainhub tavily-python
```

We also need to export some environment variables needed for our agent.

```shell
export OPENAI_API_KEY=sk-...
export TAVILY_API_KEY=tvly-...
```

Optionally, we can set up [LangSmith](https://docs.smith.langchain.com/) for best-in-class observability.

```shell
export LANGCHAIN_TRACING_V2="true"
export LANGCHAIN_API_KEY=ls__...
export LANGCHAIN_ENDPOINT=https://api.langchain.plus
```

### Define the LangChain Agent

This is the LangChain agent.
Crucially, this agent is just responsible for deciding what actions to take.
For more information on what is happening here, please see [this documentation](https://python.langchain.com/docs/modules/agents/quick_start).

```python
from langchain import hub
from langchain.agents import create_openai_functions_agent
from langchain_openai.chat_models import ChatOpenAI
from langchain_community.tools.tavily_search import TavilySearchResults

tools = [TavilySearchResults(max_results=1)]

# Get the prompt to use - you can modify this!
prompt = hub.pull("hwchase17/openai-functions-agent")

# Choose the LLM that will drive the agent
# We set streaming=True so that we can stream tokens (we will cover this more detail later on)
llm = ChatOpenAI(model="gpt-3.5-turbo-1106", streaming=True)

# Construct the OpenAI Functions agent
agent_runnable = create_openai_functions_agent(llm, tools, prompt)
```

### Define the agent state

The main type of graph in `langgraph` is the `StatefulGraph`.
This graph is parameterized by a state object that it passes around to each node.
Each node then returns operations to update that state.
These operations can either SET specific attributes on the state (e.g. overwrite the existing values) or ADD to the existing attribute.
Whether to set or add is denoted by annotating the state object you construct the graph with.

The state for the traditional LangChain agent has a few attributes:

1. `input`: This is the input string representing the main ask from the user, passed in as input.
2. `chat_history`: This is any previous conversation messages, also passed in as input.
3. `intermediate_steps`: This is list of actions and corresponding observations that the agent takes over time. This is updated each iteration of the agent.
4. `agent_outcome`: This is the response from the agent, either an AgentAction or AgentFinish. The AgentExecutor should finish when this is an AgentFinish, otherwise it should call the requested tools.

Let's make these ideas concrete by create an agent state!

```python
from typing import TypedDict, Annotated, Sequence, Union
from langchain_core.agents import AgentAction, AgentFinish
from langchain_core.messages import BaseMessage
import operator


class AgentState(TypedDict):
   # The input string
   input: str
   # The list of previous messages in the conversation
   chat_history: Sequence[BaseMessage]
   # The outcome of a given call to the agent
   # Needs `None` as a valid type, since this is what this will start as
   agent_outcome: Union[AgentAction, AgentFinish, None]
   # List of actions and corresponding observations
   # Here we annotate this with `operator.add` to indicate that operations to
   # this state should be ADDED to the existing values (not overwrite it)
   intermediate_steps: Annotated[list[tuple[AgentAction, str]], operator.add]

```

### Define the nodes

We now need to define a few different nodes in our graph.
In `langgraph`, a node can be either a function or a [runnable](https://python.langchain.com/docs/expression_language/).
There are two main nodes we need for this:

1. The agent: responsible for deciding what (if any) actions to take.
2. A function to invoke tools: if the agent decides to take an action, this node will then execute that action.

We will also need to define some edges.
Some of these edges may be conditional.
The reason they are conditional is that based on the output of a node, one of several paths may be taken.
The path that is taken is not known until that node is run (the LLM decides).

1. Conditional Edge: after the agent is called, we should either:
   a. If the agent said to take an action, then the function to invoke tools should be called
   b. If the agent said that it was finished, then it should finish
2. Normal Edge: after the tools are invoked, it should always go back to the agent to decide what to do next

Let's define the nodes, as well as a function to decide how what conditional edge to take.

```python
from langchain_core.agents import AgentFinish
from langgraph.prebuilt.tool_executor import ToolExecutor

# This a helper class we have that is useful for running tools
# It takes in an agent action and calls that tool and returns the result
tool_executor = ToolExecutor(tools)

# Define the agent
def run_agent(data):
    agent_outcome = agent_runnable.invoke(data)
    return {"agent_outcome": agent_outcome}

# Define the function to execute tools
def execute_tools(data):
    # Get the most recent agent_outcome - this is the key added in the `agent` above
    agent_action = data['agent_outcome']
    output = tool_executor.invoke(agent_action)
    return {"intermediate_steps": [(agent_action, str(output))]}

# Define logic that will be used to determine which conditional edge to go down
def should_continue(data):
    # If the agent outcome is an AgentFinish, then we return `exit` string
    # This will be used when setting up the graph to define the flow
    if isinstance(data['agent_outcome'], AgentFinish):
        return "end"
    # Otherwise, an AgentAction is returned
    # Here we return `continue` string
    # This will be used when setting up the graph to define the flow
    else:
        return "continue"
```

### Define the graph

We can now put it all together and define the graph!

```python
from langgraph.graph import END, StateGraph

# Define a new graph
 workflow = StateGraph(AgentState)

 # Define the two nodes we will cycle between
 workflow.add_node("agent", run_agent)
 workflow.add_node("action", execute_tools)

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
     # Finally we pass in a mapping.
     # The keys are strings, and the values are other nodes.
     # END is a special node marking that the graph should finish.
     # What will happen is we will call `should_continue`, and then the output of that
     # will be matched against the keys in this mapping.
     # Based on which one it matches, that node will then be called.
     {
         # If `tools`, then we call the tool node.
         "continue": "action",
         # Otherwise we finish.
         "end": END
     }
 )

 # We now add a normal edge from `tools` to `agent`.
 # This means that after `tools` is called, `agent` node is called next.
 workflow.add_edge('action', 'agent')

 # Finally, we compile it!
 # This compiles it into a LangChain Runnable,
 # meaning you can use it as you would any other runnable
 chain = workflow.compile()
```

### Use it!

We can now use it!
This now exposes the [same interface](https://python.langchain.com/docs/expression_language/) as all other LangChain runnables

```python
chain.invoke({"input": "what is the weather in sf"})
```

## Streaming

LangGraph has support for several different types of streaming.

### Streaming Node Output

One of the benefits of using LangGraph is that it is easy to stream output as it's produced by each node.

```python
for output in chain.stream(
    {"input": "what is the weather in sf"}
):
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
{'agent_outcome': AgentActionMessageLog(tool='tavily_search_results_json', tool_input={'query': 'weather in San Francisco'}, log="\nInvoking: `tavily_search_results_json` with `{'query': 'weather in San Francisco'}`\n\n\n", message_log=[AIMessage(content='', additional_kwargs={'function_call': {'arguments': '{"query":"weather in San Francisco"}', 'name': 'tavily_search_results_json'}})])}

---

Output from node 'action':
---
{'intermediate_steps': [(AgentActionMessageLog(tool='tavily_search_results_json', tool_input={'query': 'weather in San Francisco'}, log="\nInvoking: `tavily_search_results_json` with `{'query': 'weather in San Francisco'}`\n\n\n", message_log=[AIMessage(content='', additional_kwargs={'function_call': {'arguments': '{"query":"weather in San Francisco"}', 'name': 'tavily_search_results_json'}})]), "[{'url': 'https://weatherspark.com/h/m/557/2024/1/Historical-Weather-in-January-2024-in-San-Francisco-California-United-States', 'content': 'January 2024 Weather History in San Francisco California, United States  Daily Precipitation in January 2024 in San Francisco Observed Weather in January 2024 in San Francisco  San Francisco Temperature History January 2024 Hourly Temperature in January 2024 in San Francisco  Hours of Daylight and Twilight in January 2024 in San FranciscoThis report shows the past weather for San Francisco, providing a weather history for January 2024. It features all historical weather data series we have available, including the San Francisco temperature history for January 2024. You can drill down from year to month and even day level reports by clicking on the graphs.'}]")]}

---

Output from node 'agent':
---
{'agent_outcome': AgentFinish(return_values={'output': "It seems that I couldn't retrieve the current weather in San Francisco. However, you can easily check the current weather in San Francisco by using a weather website or app."}, log="It seems that I couldn't retrieve the current weather in San Francisco. However, you can easily check the current weather in San Francisco by using a weather website or app.")}

---

Output from node '__end__':
---
{'input': 'what is the weather in sf', 'chat_history': [], 'agent_outcome': AgentFinish(return_values={'output': "It seems that I couldn't retrieve the current weather in San Francisco. However, you can easily check the current weather in San Francisco by using a weather website or app."}, log="It seems that I couldn't retrieve the current weather in San Francisco. However, you can easily check the current weather in San Francisco by using a weather website or app."), 'intermediate_steps': [(AgentActionMessageLog(tool='tavily_search_results_json', tool_input={'query': 'weather in San Francisco'}, log="\nInvoking: `tavily_search_results_json` with `{'query': 'weather in San Francisco'}`\n\n\n", message_log=[AIMessage(content='', additional_kwargs={'function_call': {'arguments': '{"query":"weather in San Francisco"}', 'name': 'tavily_search_results_json'}})]), "[{'url': 'https://weatherspark.com/h/m/557/2024/1/Historical-Weather-in-January-2024-in-San-Francisco-California-United-States', 'content': 'January 2024 Weather History in San Francisco California, United States  Daily Precipitation in January 2024 in San Francisco Observed Weather in January 2024 in San Francisco  San Francisco Temperature History January 2024 Hourly Temperature in January 2024 in San Francisco  Hours of Daylight and Twilight in January 2024 in San FranciscoThis report shows the past weather for San Francisco, providing a weather history for January 2024. It features all historical weather data series we have available, including the San Francisco temperature history for January 2024. You can drill down from year to month and even day level reports by clicking on the graphs.'}]")]}

---
```

### Streaming LLM Tokens

You can also access the LLM tokens as they are produced by each node. 
In this case only the "agent" node produces LLM tokens.
In order for this to work properly, you must be using an LLM that supports streaming as well as have set it when constructing the LLM (e.g. `ChatOpenAI(model="gpt-3.5-turbo-1106", streaming=True)`)

```python
async for output in chain.astream_log(
    {"input": "what is the weather in sf", "intermediate_steps": []},
    include_types=["llm"],
):
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
content='' additional_kwargs={'function_call': {'arguments': '{"', 'name': ''}}
content='' additional_kwargs={'function_call': {'arguments': 'query', 'name': ''}}
content='' additional_kwargs={'function_call': {'arguments': '":"', 'name': ''}}
content='' additional_kwargs={'function_call': {'arguments': 'current', 'name': ''}}
content='' additional_kwargs={'function_call': {'arguments': ' weather', 'name': ''}}
content='' additional_kwargs={'function_call': {'arguments': ' in', 'name': ''}}
content='' additional_kwargs={'function_call': {'arguments': ' San', 'name': ''}}
content='' additional_kwargs={'function_call': {'arguments': ' Francisco', 'name': ''}}
content='' additional_kwargs={'function_call': {'arguments': '"}', 'name': ''}}
content=''
content=''
content='I'
content=' found'
content=' a'
content=' website'
content=' that'
content=' provides'
content=' detailed'
content=' weather'
content=' information'
content=' for'
content=' San'
content=' Francisco'
content='.'
content=' You'
content=' can'
content=' visit'
content=' the'
content=' following'
content=' link'
content=' for'
content=' the'
content=' current'
content=' weather'
content=' report'
content=':'
content=' ['
content='San'
content=' Francisco'
content=' Weather'
content=' Report'
content=']('
content='https'
content='://'
content='www'
content='.weather'
content='25'
content='.com'
content='/n'
content='orth'
content='-'
content='amer'
content='ica'
content='/'
content='usa'
content='/cal'
content='ifornia'
content='/s'
content='an'
content='-fr'
content='anc'
content='isco'
content=')'
content=''
```





## When to Use

When should you use this versus [LangChain Expression Language](https://python.langchain.com/docs/expression_language/)?

If you need cycles.

Langchain Expression Language allows you to easily define chains (DAGs) but does not have a good mechanism for adding in cycles.
`langgraph` adds that syntax.

## Examples


### ChatExecutor: with function calling

### AgentExecutor


## Documentation

There are only a few new APIs to use.

### StateGraph

The main entrypoint is `StateGraph`.

```python
from langgraph.graph import StateGraph
```

This class is responsible for constructing the graph.
It exposes an interface inspired by [NetworkX](https://networkx.org/documentation/latest/).
This graph is parameterized by a state object that it passes around to each node.


#### `__init__`

```python
    def __init__(self, schema: Type[Any]) -> None:
```

When constructing the graph, you need to pass in a schema for a state.
Each node then returns operations to update that state.
These operations can either SET specific attributes on the state (e.g. overwrite the existing values) or ADD to the existing attribute.
Whether to set or add is denoted by annotating the state object you construct the graph with.

The recommended way to specify the schema is with a typed dictionary: `from typing import TypedDict`

You can then annotate the different attributes using `from typing imoport Annotated`.
Currently, the only supported annotation is `import operator; operator.add`.
This annotation will make it so that any node that returns this attribute ADDS that new result to the existing value.

Let's take a look at an example:

```python
from typing import TypedDict, Annotated, Union
from langchain_core.agents import AgentAction, AgentFinish
import operator


class AgentState(TypedDict):
   # The input string
   input: str
   # The outcome of a given call to the agent
   # Needs `None` as a valid type, since this is what this will start as
   agent_outcome: Union[AgentAction, AgentFinish, None]
   # List of actions and corresponding observations
   # Here we annotate this with `operator.add` to indicate that operations to
   # this state should be ADDED to the existing values (not overwrite it)
   intermediate_steps: Annotated[list[tuple[AgentAction, str]], operator.add]

```

We can then use this like:

```python
# Initialize the StateGraph with this state
graph = StateGraph(AgentState)
# Create nodes and edges
...
# Compile the graph
app = graph.compile()

# The inputs should be a dictionary, because the state is a TypedDict
inputs = {
   # Let's assume this the input
   "input": "hi"
   # Let's assume agent_outcome is set by the graph as some point
   # It doesn't need to be provided, and it will be None by default
   # Let's assume `intermediate_steps` is built up over time by the graph
   # It doesn't need to provided, and it will be empty list by default
   # The reason `intermediate_steps` is an empty list and not `None` is because
   # it's annotated with `operator.add`
}
```

#### `.add_node`

```python
    def add_node(self, key: str, action: RunnableLike) -> None:
```

This method adds a node to the graph.
It takes two arguments:

- `key`: A string representing the name of the node. This must be unique.
- `action`: The action to take when this node is called. This should either be a function or a runnable.

#### `.add_edge`

```python
    def add_edge(self, start_key: str, end_key: str) -> None:
```

Creates an edge from one node to the next.
This means that output of the first node will be passed to the next node.
It takes two arguments.

- `start_key`: A string representing the name of the start node. This key must have already been registered in the graph.
- `end_key`: A string representing the name of the end node. This key must have already been registered in the graph.

#### `.add_conditional_edges`

```python
    def add_conditional_edges(
        self,
        start_key: str,
        condition: Callable[..., str],
        conditional_edge_mapping: Dict[str, str],
    ) -> None:
```

This method adds conditional edges.
What this means is that only one of the downstream edges will be taken, and which one that is depends on the results of the start node.
This takes three arguments:

- `start_key`: A string representing the name of the start node. This key must have already been registered in the graph.
- `condition`: A function to call to decide what to do next. The input will be the output of the start node. It should return a string that is present in `conditional_edge_mapping` and represents the edge to take.
- `conditional_edge_mapping`: A mapping of string to string. The keys should be strings that may be returned by `condition`. The values should be the downstream node to call if that condition is returned.

#### `.set_entry_point`

```python
    def set_entry_point(self, key: str) -> None:
```

The entrypoint to the graph.
This is the node that is first called.
It only takes one argument:

- `key`: The name of the node that should be called first.

#### `.set_finish_point`

```python
    def set_finish_point(self, key: str) -> None:
```

This is the exit point of the graph.
When this node is called, the results will be the final result from the graph.
It only has one argument:

- `key`: The name of the node that, when called, will return the results of calling it as the final output

Note: This does not need to be called if at any point you previously created an edge (conditional or normal) to `END`

### Graph

```python
from langgraph.graph import Graph

graph = Graph()
```

This has the same interface as `StateGraph` with the exception that it doesn't update a state object over time, and rather relies on passing around the full state from each step.
This means that whatever is returned from one node is the input to the next as is.

### `END`

```python
from langgraph.graph import END
```

This is a special node representing the end of the graph.
This means that anything passed to this node will be the final output of the graph.
It can be used in two places:

- As the `end_key` in `add_edge`
- As a value in `conditional_edge_mapping` as passed to `add_conditional_edges`


## Prebuilt Examples

There are also a few methods we've added to make it easy to use common, prebuilt graphs and components.

### ToolExecutor

```python
from langgraph.prebuilt import ToolExecutor
```

This is a simple helper class to help with calling tools.
It is parameterized by a list of tools:

```python
tools = [...]
tool_executor = ToolExecutor(tools)
```

It then exposes a [runnable interface](https://python.langchain.com/docs/expression_language/interface).
It can be used to call tools: you can pass in an [AgentAction](https://python.langchain.com/docs/modules/agents/concepts#agentaction) and it will look up the relevant tool and call it with the appropriate input.

### chat_executor.create_function_calling_executor

```python
from langgraph.prebuilt import chat_executor
```

This is a helper function for creating a graph that works with a chat model that utilizes function calling.
Can be created by passing in a model and a list of tools.
The model must be one that supports OpenAI function calling.

```python
from langchain_openai import ChatOpenAI
from langchain_community.tools.tavily_search import TavilySearchResults
from langgraph.prebuilt import chat_executor
from langchain_core.messages import HumanMessage

tools = [TavilySearchResults(max_results=1)]
model = ChatOpenAI()

app = chat_executor.create_function_calling_executor(model, tools)

inputs = {"messages": [HumanMessage(content="what is the weather in sf")]}
for s in app.stream(inputs):
    print(list(s.values())[0])
    print("----")
```

### create_agent_executor

```python
from langgraph.prebuilt import create_agent_executor
```

This is a helper function for creating a graph that works with [LangChain Agents](https://python.langchain.com/docs/modules/agents/).
Can be created by passing in an agent and a list of tools.

```python
from langgraph.prebuilt import create_agent_executor
from langchain_openai import ChatOpenAI
from langchain import hub
from langchain.agents import create_openai_functions_agent
from langchain_community.tools.tavily_search import TavilySearchResults

tools = [TavilySearchResults(max_results=1)]

# Get the prompt to use - you can modify this!
prompt = hub.pull("hwchase17/openai-functions-agent")

# Choose the LLM that will drive the agent
llm = ChatOpenAI(model="gpt-3.5-turbo-1106")

# Construct the OpenAI Functions agent
agent_runnable = create_openai_functions_agent(llm, tools, prompt)

app = create_agent_executor(agent_runnable, tools)

inputs = {"input": "what is the weather in sf", "chat_history": []}
for s in app.stream(inputs):
    print(list(s.values())[0])
    print("----")
```
