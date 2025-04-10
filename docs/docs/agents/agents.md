# Agents

An agent is a system that uses an LLM to control the flow of an application. The most common form of an agent is an LLM that calls tools in a loop.

Use [`create_react_agent`][langgraph.prebuilt.chat_agent_executor.create_react_agent] to create a tool-calling agent.

## Quickstart

The most important things to configure when you create an agent are:

- `model`: a [chat model](https://python.langchain.com/docs/concepts/chat_models/) that the agent will use.
- `tools`: a list of [tools](https://python.langchain.com/docs/concepts/tools/) for the agent to use.
- `prompt`: string or system message with instructions for the agent.

```python
from langgraph.prebuilt import create_react_agent

def get_weather(city: str) -> str:
    """Get weather for a given city."""
    return f"It's always sunny in {city}!"

agent = create_react_agent(
    model="anthropic:claude-3-7-sonnet-latest",
    tools=[get_weather],
    prompt="You are a helpful assistant"
)

# Run the agent
agent.invoke({"messages": "what is the weather in sf"})
```

## Customize models

If you want to customize the model parameters, you can pass a `ChatModel` instance as `model`:

```python
from langchain_anthropic import ChatAnthropic
# highlight-next-line
model = ChatAnthropic(model="claude-3-7-sonnet-latest", temperature=0.7)

agent = create_react_agent(
    # highlight-next-line
    model=model,
    tools=[get_weather],
)
```

## Customize prompt

You can include context (e.g., user information) in the prompt:

```python
from langgraph.prebuilt.chat_agent_executor import AgentState
from langchain_core.runnables import RunnableConfig

def prompt(state: AgentState, config: RunnableConfig):
    user_name = config.get("configurable", {}).get("user_name")
    system_msg = f"You are a helpful assistant. Address the user as {user_name}."
    return [{"role": "system", "content": system_msg}] + state["messages"]

agent = create_react_agent(
    model="anthropic:claude-3-7-sonnet-latest",
    tools=[get_weather],
    # highlight-next-line
    prompt=prompt
)

agent.invoke(
    {"messages": "what is the weather in sf"},
    # highlight-next-line
    config={"configurable": {"user_name": "John Smith"}}
)
```

See more on how to manage context in your agent [here](./context.md).

## Structured output

By default, the final agent response is simply an AI message with text content. To return the agent's response in a structured output that conforms to a given schema, you can provide the schema via the `response_format` parameter. The schema can be a Pydantic model or a `TypedDict` object:

```python
from langgraph.prebuilt.chat_agent_executor import AgentState
from langchain_core.runnables import RunnableConfig
from pydantic import BaseModel

class WeatherResponse(BaseModel):
    conditions: str

agent = create_react_agent(
    model="anthropic:claude-3-7-sonnet-latest",
    tools=[get_weather],
    # highlight-next-line
    response_format=WeatherResponse
)

response = agent.invoke({"messages": "what is the weather in sf"})

# highlight-next-line
response["structured_response"]
```
!!! Note
    To return structured output, `create_react_agent` makes an additional LLM call at the end of the tool-calling loop.
