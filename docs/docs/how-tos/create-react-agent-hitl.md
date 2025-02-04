# How to add human-in-the-loop processes to the prebuilt ReAct agent

<div class="admonition tip">
    <p class="admonition-title">Prerequisites</p>
    <p>
        This guide assumes familiarity with the following:
        <ul>
            <li>            
                <a href="https://langchain-ai.github.io/langgraph/concepts/human_in_the_loop/">
                    Human-in-the-loop
                </a>
            </li>
            <li>
                <a href="https://langchain-ai.github.io/langgraph/concepts/agentic_concepts/">
                    Agent Architectures
                </a>                   
            </li>
            <li>
                <a href="https://python.langchain.com/docs/concepts/chat_models/">
                    Chat Models
                </a>
            </li>
            <li>
                <a href="https://python.langchain.com/docs/concepts/tools/">
                    Tools
                </a>
            </li>            
        </ul>
    </p>
</div> 

This guide will show how to add human-in-the-loop processes to the prebuilt ReAct agent. Please see [this tutorial](../create-react-agent.md) for how to get started with the prebuilt ReAct agent

You can add a a breakpoint before tools are called by passing `interrupt_before=["tools"]` to `create_react_agent`. Note that you need to be using a checkpointer for this to work.

## Setup

First, let's install the required packages and set our API keys


```
%%capture --no-stderr
%pip install -U langgraph langchain-openai
```


```python
import getpass
import os


def _set_env(var: str):
    if not os.environ.get(var):
        os.environ[var] = getpass.getpass(f"{var}: ")


_set_env("OPENAI_API_KEY")
```

<div class="admonition tip">
    <p class="admonition-title">Set up <a href="https://smith.langchain.com">LangSmith</a> for LangGraph development</p>
    <p style="padding-top: 5px;">
        Sign up for LangSmith to quickly spot issues and improve the performance of your LangGraph projects. LangSmith lets you use trace data to debug, test, and monitor your LLM apps built with LangGraph — read more about how to get started <a href="https://docs.smith.langchain.com">here</a>. 
    </p>
</div>

## Code


```python
# First we initialize the model we want to use.
from langchain_openai import ChatOpenAI

model = ChatOpenAI(model="gpt-4o", temperature=0)


# For this tutorial we will use custom tool that returns pre-defined values for weather in two cities (NYC & SF)
from typing import Literal

from langchain_core.tools import tool


@tool
def get_weather(location: str):
    """Use this to get weather information from a given location."""
    if location.lower() in ["nyc", "new york"]:
        return "It might be cloudy in nyc"
    elif location.lower() in ["sf", "san francisco"]:
        return "It's always sunny in sf"
    else:
        raise AssertionError("Unknown Location")


tools = [get_weather]

# We need a checkpointer to enable human-in-the-loop patterns
from langgraph.checkpoint.memory import MemorySaver

memory = MemorySaver()

# Define the graph

from langgraph.prebuilt import create_react_agent

graph = create_react_agent(
    model, tools=tools, interrupt_before=["tools"], checkpointer=memory
)
```

## Usage



```python
def print_stream(stream):
    """A utility to pretty print the stream."""
    for s in stream:
        message = s["messages"][-1]
        if isinstance(message, tuple):
            print(message)
        else:
            message.pretty_print()
```


```python
from langchain_core.messages import HumanMessage

config = {"configurable": {"thread_id": "42"}}
inputs = {"messages": [("user", "what is the weather in SF, CA?")]}

print_stream(graph.stream(inputs, config, stream_mode="values"))
```

We can verify that our graph stopped at the right place:


```python
snapshot = graph.get_state(config)
print("Next step: ", snapshot.next)
```

Now we can either approve or edit the tool call before proceeding to the next node. If we wanted to approve the tool call, we would simply continue streaming the graph with `None` input. If we wanted to edit the tool call we need to update the state to have the correct tool call, and then after the update has been applied we can continue.

We can try resuming and we will see an error arise:


```python
print_stream(graph.stream(None, config, stream_mode="values"))
```

This error arose because our tool argument of "San Francisco, CA" is not a location our tool recognizes.

Let's show how we would edit the tool call to search for "San Francisco" instead of "San Francisco, CA" - since our tool as written treats "San Francisco, CA" as an unknown location. We will update the state and then resume streaming the graph and should see no errors arise:


```python
state = graph.get_state(config)

last_message = state.values["messages"][-1]
last_message.tool_calls[0]["args"] = {"location": "San Francisco"}

graph.update_state(config, {"messages": [last_message]})
```







```python
print_stream(graph.stream(None, config, stream_mode="values"))
```

Fantastic! Our graph updated properly to query the weather in San Francisco and got the correct "It's always sunny in sf" response from the tool, and then responded to the user accordingly.
