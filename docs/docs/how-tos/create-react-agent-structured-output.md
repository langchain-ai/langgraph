# How to return structured output from the prebuilt ReAct agent

!!! info "Prerequisites"
    This guide assumes familiarity with the following:
    
    - [Agent Architectures](../../concepts/agentic_concepts/.md)
    - [Chat Models](https://python.langchain.com/docs/concepts/chat_models/)
    - [Tools](https://python.langchain.com/docs/concepts/tools/)
    - [Structured Output](https://python.langchain.com/docs/concepts/structured_outputs/)

To return structured output from the prebuilt ReAct agent you can provide a `response_format` parameter with the desired output schema to [create_react_agent][langgraph.prebuilt.chat_agent_executor.create_react_agent]:

```python
class ResponseFormat(BaseModel):
    """Respond to the user in this format."""
    my_special_output: str


graph = create_react_agent(
    model,
    tools=tools,
    # specify the schema for the structured output using `response_format` parameter
    response_format=ResponseFormat
)
```

Prebuilt ReAct makes an additional LLM call at the end of the ReAct loop to produce a structured output response. Please see [this guide](../react-agent-structured-output.md) to learn about other  strategies for returning structured outputs from a tool-calling agent.

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
def get_weather(city: Literal["nyc", "sf"]):
    """Use this to get weather information."""
    if city == "nyc":
        return "It might be cloudy in nyc"
    elif city == "sf":
        return "It's always sunny in sf"
    else:
        raise AssertionError("Unknown city")


tools = [get_weather]

# Define the structured output schema

from pydantic import BaseModel, Field


class WeatherResponse(BaseModel):
    """Respond to the user in this format."""

    conditions: str = Field(description="Weather conditions")


# Define the graph

from langgraph.prebuilt import create_react_agent

graph = create_react_agent(
    model,
    tools=tools,
    # specify the schema for the structured output using `response_format` parameter
    response_format=WeatherResponse,
)
```

## Usage

Let's now test our agent:


```python
inputs = {"messages": [("user", "What's the weather in NYC?")]}
response = graph.invoke(inputs)
```

You can see that the agent output contains a `structured_response` key with the structured output conforming to the specified `WeatherResponse` schema, in addition to the message history under `messages` key.


```python
response["structured_response"]
```






### Customizing prompt

You might need to further customize the second LLM call for the structured output generation and provide a system prompt. To do so, you can pass a tuple (prompt, schema):


```python
graph = create_react_agent(
    model,
    tools=tools,
    # specify both the system prompt and the schema for the structured output
    response_format=("Always return capitalized weather conditions", WeatherResponse),
)

inputs = {"messages": [("user", "What's the weather in NYC?")]}
response = graph.invoke(inputs)
```

You can verify that the structured response now contains a capitalized value:


```python
response["structured_response"]
```





