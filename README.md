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
llm = ChatOpenAI(model="gpt-3.5-turbo-1106")

# Construct the OpenAI Functions agent
agent_runnable = create_openai_functions_agent(llm, tools, prompt)
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
from langchain_core.runnables import RunnablePassthrough
from langchain_core.agents import AgentFinish


# Define the agent
# Note that here, we are using `.assign` to add the output of the agent to the dictionary
# This dictionary will be returned from the node
# The reason we don't want to return just the result of `agent_runnable` from this node is
# that we want to continue passing around all the other inputs
agent = RunnablePassthrough.assign(
    agent_outcome = agent_runnable
)

# Define the function to execute tools
def execute_tools(data):
    # Get the most recent agent_outcome - this is the key added in the `agent` above
    agent_action = data.pop('agent_outcome')
    # Get the tool to use
    tool_to_use = {t.name: t for t in tools}[agent_action.tool]
    # Call that tool on the input
    observation = tool_to_use.invoke(agent_action.tool_input)
    # We now add in the action and the observation to the `intermediate_steps` list
    # This is the list of all previous actions taken and their output
    data['intermediate_steps'].append((agent_action, observation))
    return data

# Define logic that will be used to determine which conditional edge to go down
def should_continue(data):
    # If the agent outcome is an AgentFinish, then we return `exit` string
    # This will be used when setting up the graph to define the flow
    if isinstance(data['agent_outcome'], AgentFinish):
        return "exit"
    # Otherwise, an AgentAction is returned
    # Here we return `continue` string
    # This will be used when setting up the graph to define the flow
    else:
        return "continue"
```

### Define the graph

We can now put it alltogether and define the graph!

```python
from langgraph.graph import END, Graph

workflow = Graph()

# Add the agent node, we give it name `agent` which we will use later
workflow.add_node("agent", agent)
# Add the tools node, we give it name `tools` which we will use later
workflow.add_node("tools", execute_tools)

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
        "continue": "tools",
        # Otherwise we finish.
        "exit": END
    }
)

# We now add a normal edge from `tools` to `agent`.
# This means that after `tools` is called, `agent` node is called next.
workflow.add_edge('tools', 'agent')

# Finally, we compile it!
# This compiles it into a LangChain Runnable,
# meaning you can use it as you would any other runnable
chain = workflow.compile()
```

### Use it!

We can now use it!
This now exposes the [same interface](https://python.langchain.com/docs/expression_language/) as all other LangChain runnables

```python
chain.invoke({"input": "what is the weather in sf", "intermediate_steps": []})
```

### Streaming

One of the benefits of using LangGraph is that it is easy to stream output as it's produced by each node.

```python
for output in chain.stream(
    {"input": "what is the weather in sf", "intermediate_steps": []}
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
{'agent_outcome': AgentActionMessageLog(tool='tavily_search_results_json', tool_input={'query': 'weather in San Francisco'}, log="\nInvoking: `tavily_search_results_json` with `{'query': 'weather in San Francisco'}`\n\n\n", message_log=[AIMessage(content='', additional_kwargs={'function_call': {'arguments': '{"query":"weather in San Francisco"}', 'name': 'tavily_search_results_json'}})]),
 'input': 'what is the weather in sf',
 'intermediate_steps': []}

---

Output from node 'tools':
---
{'input': 'what is the weather in sf',
 'intermediate_steps': [(AgentActionMessageLog(tool='tavily_search_results_json', tool_input={'query': 'weather in San Francisco'}, log="\nInvoking: `tavily_search_results_json` with `{'query': 'weather in San Francisco'}`\n\n\n", message_log=[AIMessage(content='', additional_kwargs={'function_call': {'arguments': '{"query":"weather in San Francisco"}', 'name': 'tavily_search_results_json'}})]),
                         [{'content': 'Best time to go to San Francisco? '
                                      'Weather in San Francisco in january '
                                      '2024  How was the weather last january? '
                                      'Here is the day by day recorded weather '
                                      'in San Francisco in january 2023:  '
                                      'Seasonal average climate and '
                                      'temperature of San Francisco in '
                                      'january  8% 46% 29% 12% 8% Evolution of '
                                      'daily average temperature and '
                                      'precipitation in San Francisco in '
                                      'januaryWeather in San Francisco in '
                                      'january 2024. The weather in San '
                                      'Francisco in january comes from '
                                      'statistical datas on the past years. '
                                      'You can view the weather statistics the '
                                      'entire month, but also by using the '
                                      'tabs for the beginning, the middle and '
                                      'the end of the month. ... 08-01-2023 '
                                      '52°F to 58°F. 09-01-2023 54°F to 61°F. '
                                      '10-01-2023 52°F to ...',
                           'url': 'https://www.whereandwhen.net/when/north-america/california/san-francisco-ca/january/'}])]}

---

Output from node 'agent':
---
{'agent_outcome': AgentFinish(return_values={'output': 'The weather in San Francisco in January ranges from 52°F to 61°F. For more detailed and current weather information, you may want to check a reliable weather website or app.'}, log='The weather in San Francisco in January ranges from 52°F to 61°F. For more detailed and current weather information, you may want to check a reliable weather website or app.'),
 'input': 'what is the weather in sf',
 'intermediate_steps': [(AgentActionMessageLog(tool='tavily_search_results_json', tool_input={'query': 'weather in San Francisco'}, log="\nInvoking: `tavily_search_results_json` with `{'query': 'weather in San Francisco'}`\n\n\n", message_log=[AIMessage(content='', additional_kwargs={'function_call': {'arguments': '{"query":"weather in San Francisco"}', 'name': 'tavily_search_results_json'}})]),
                         [{'content': 'Best time to go to San Francisco? '
                                      'Weather in San Francisco in january '
                                      '2024  How was the weather last january? '
                                      'Here is the day by day recorded weather '
                                      'in San Francisco in january 2023:  '
                                      'Seasonal average climate and '
                                      'temperature of San Francisco in '
                                      'january  8% 46% 29% 12% 8% Evolution of '
                                      'daily average temperature and '
                                      'precipitation in San Francisco in '
                                      'januaryWeather in San Francisco in '
                                      'january 2024. The weather in San '
                                      'Francisco in january comes from '
                                      'statistical datas on the past years. '
                                      'You can view the weather statistics the '
                                      'entire month, but also by using the '
                                      'tabs for the beginning, the middle and '
                                      'the end of the month. ... 08-01-2023 '
                                      '52°F to 58°F. 09-01-2023 54°F to 61°F. '
                                      '10-01-2023 52°F to ...',
                           'url': 'https://www.whereandwhen.net/when/north-america/california/san-francisco-ca/january/'}])]}

---

Output from node '__end__':
---
{'agent_outcome': AgentFinish(return_values={'output': 'The weather in San Francisco in January ranges from 52°F to 61°F. For more detailed and current weather information, you may want to check a reliable weather website or app.'}, log='The weather in San Francisco in January ranges from 52°F to 61°F. For more detailed and current weather information, you may want to check a reliable weather website or app.'),
 'input': 'what is the weather in sf',
 'intermediate_steps': [(AgentActionMessageLog(tool='tavily_search_results_json', tool_input={'query': 'weather in San Francisco'}, log="\nInvoking: `tavily_search_results_json` with `{'query': 'weather in San Francisco'}`\n\n\n", message_log=[AIMessage(content='', additional_kwargs={'function_call': {'arguments': '{"query":"weather in San Francisco"}', 'name': 'tavily_search_results_json'}})]),
                         [{'content': 'Best time to go to San Francisco? '
                                      'Weather in San Francisco in january '
                                      '2024  How was the weather last january? '
                                      'Here is the day by day recorded weather '
                                      'in San Francisco in january 2023:  '
                                      'Seasonal average climate and '
                                      'temperature of San Francisco in '
                                      'january  8% 46% 29% 12% 8% Evolution of '
                                      'daily average temperature and '
                                      'precipitation in San Francisco in '
                                      'januaryWeather in San Francisco in '
                                      'january 2024. The weather in San '
                                      'Francisco in january comes from '
                                      'statistical datas on the past years. '
                                      'You can view the weather statistics the '
                                      'entire month, but also by using the '
                                      'tabs for the beginning, the middle and '
                                      'the end of the month. ... 08-01-2023 '
                                      '52°F to 58°F. 09-01-2023 54°F to 61°F. '
                                      '10-01-2023 52°F to ...',
                           'url': 'https://www.whereandwhen.net/when/north-america/california/san-francisco-ca/january/'}])]}

---
```

### Streaming LLM Tokens

You can also access the LLM tokens as they are produced by each node. In this case only the "agent" node produces LLM tokens.

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
            # these are tokens from the LLM
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

## Documentation

There are only a few new APIs to use.

The main new class is `Graph`.

```python
from langgraph.graph import Graph
```

This class is responsible for constructing the graph.
It exposes an interface inspired by [NetworkX](https://networkx.org/documentation/latest/).

### `.add_node`

```python
    def add_node(self, key: str, action: RunnableLike) -> None:
```

This method adds a node to the graph.
It takes two arguments:

- `key`: A string representing the name of the node. This must be unique.
- `action`: The action to take when this node is called. This should either be a function or a runnable.

### `.add_edge`

```python
    def add_edge(self, start_key: str, end_key: str) -> None:
```

Creates an edge from one node to the next.
This means that output of the first node will be passed to the next node.
It takes two arguments.

- `start_key`: A string representing the name of the start node. This key must have already been registered in the graph.
- `end_key`: A string representing the name of the end node. This key must have already been registered in the graph.

### `.add_conditional_edges`

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

### `.set_entry_point`

```python
    def set_entry_point(self, key: str) -> None:
```

The entrypoint to the graph.
This is the node that is first called.
It only takes one argument:

- `key`: The name of the node that should be called first.

### `.set_finish_point`

```python
    def set_finish_point(self, key: str) -> None:
```

This is the exit point of the graph.
When this node is called, the results will be the final result from the graph.
It only has one argument:

- `key`: The name of the node that, when called, will return the results of calling it as the final output

Note: This does not need to be called if at any point you previously created an edge (conditional or normal) to `END`

### `END`

```python
from langgraph.graph import END
```

This is a special node representing the end of the graph.
This means that anything passed to this node will be the final output of the graph.
It can be used in two places:

- As the `end_key` in `add_edge`
- As a value in `conditional_edge_mapping` as passed to `add_conditional_edges`

## When to Use

When should you use this versus [LangChain Expression Language](https://python.langchain.com/docs/expression_language/)?

If you need cycles.

Langchain Expression Language allows you to easily define chains (DAGs) but does not have a good mechanism for adding in cycles.
`langgraph` adds that syntax.

## Examples

### AgentExecutor

See the above Quick Start for an example of re-creating the LangChain [`AgentExecutor`](https://python.langchain.com/docs/modules/agents/concepts#agentexecutor) class.

### Forced Function Calling

One simple modification of the above Graph is to modify it such that a certain tool is always called first.
This can be useful if you want to enforce a certain tool is called, but still want to enable agentic behavior after the fact.

Assuming you have done the above Quick Start, you can build off it like:

#### Define the first tool call

Here, we manually define the first tool call that we will make.
Notice that it does that same thing as `agent` would have done (adds the `agent_outcome` key).
This is so that we can easily plug it in.

```python
from langchain_core.agents import AgentActionMessageLog

def first_agent(inputs):
    action = AgentActionMessageLog(
      # We force call this tool
      tool="tavily_search_results_json",
      # We just pass in the `input` key to this tool
      tool_input=inputs["input"],
      log="",
      message_log=[]
    )
    inputs["agent_outcome"] = action
    return inputs
```

#### Create the graph

We can now create a new graph with this new node

```python
workflow = Graph()

# Add the same nodes as before, plus this "first agent"
workflow.add_node("first_agent", first_agent)
workflow.add_node("agent", agent)
workflow.add_node("tools", execute_tools)

# We now set the entry point to be this first agent
workflow.set_entry_point("first_agent")

# We define the same edges as before
workflow.add_conditional_edges(
    "agent",
    should_continue,
    {
        "continue": "tools",
        "exit": END
    }
)
workflow.add_edge('tools', 'agent')

# We also define a new edge, from the "first agent" to the tools node
# This is so that we can call the tool
workflow.add_edge('first_agent', 'tools')

# We now compile the graph as before
chain = workflow.compile()
```

#### Use it!

We can now use it as before!
Depending on whether or not the first tool call is actually useful, this may save you an LLM call or two.

```python
chain.invoke({"input": "what is the weather in sf", "intermediate_steps": []})
```
